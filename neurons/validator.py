from os import urandom
import json
import copy
import time
import random
import asyncio
from neurons.base import BaseNeuron, NeuronType
from targon.dataset import create_query_prompt, create_search_prompt
from targon.utils import normalize, print_info, safe_mean, InferenceStats, check_tokens, setup_db, add_records
import traceback
import math
import numpy as np
import pandas as pd
import bittensor as bt

from typing import List
from targon import (
    protocol,
    __spec_version__ as spec_version,
)
from bittensor.utils.weight_utils import (
    process_weights_for_netuid,
)


class Validator(BaseNeuron):
    neuron_type = NeuronType.Validator

    def __init__(self, config=None):
        super().__init__(config)
        ## Typesafety
        assert self.config.netuid
        assert self.config.neuron
        assert self.config.axon

        ## BITTENSOR INITIALIZATION
        self.dendrite = bt.dendrite(wallet=self.wallet)

        ## CHECK IF REGG'D
        if not self.metagraph.validator_permit[self.uid]:
            bt.logging.error("Validator does not have vpermit")
            exit()

        ## SET MISC PARAMS
        self.hotkeys = self.metagraph.hotkeys
        self.next_forward_block = None
        self.last_posted_weights = self.metagraph.last_update[self.uid]

        ## STATS
        miners = self.get_miner_uids()
        self.miner_tps = {miner: [] for miner in miners}

        ## SET DATASET
        self.dataset = pd.read_json(
            "hf://datasets/pinecone/dl-doc-search/train.jsonl", lines=True
        )
        bt.logging.info(
            "\N{grinning face with smiling eyes}", "Successfully Initialized!"
        )

        if(self.config.database_url):
            asyncio.run(setup_db(self.config.database_url))
            bt.logging.info("Succesfully created DB")

    def create_ground_truth(self, messages, sampling_params):
        assert self.config.neuron
        ground_truth_tokens = []
        stream = self.client.chat.completions.create(
            model=self.config.neuron.model_name,
            messages=messages,
            stream=True,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            seed=sampling_params.seed,
            timeout=5,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if not token:
                continue
            ground_truth_tokens.append(token)
        ground_truth_output = "".join(ground_truth_tokens)
        return ground_truth_output

    async def handle_inference(self, messages, sampling_params, uid, ground_truth):
        assert self.config.neuron
        stats = InferenceStats(
            time_to_first_token=0,
            time_for_all_tokens=0,
            tokens_per_second=0,
            total_time=0,
            tokens=[],
            response="",
            verified=False,
        )
        try:
            synapse = protocol.Inference(
                messages=json.dumps(messages),
                sampling_params=sampling_params,
            )
            response_tokens = []
            token_count = 0
            start_send_message_time = time.time()
            end_send_message_time = None
            start_token_time = 0
            async for token in await self.dendrite(
                self.metagraph.axons[uid],
                synapse,
                deserialize=False,
                timeout=self.config.neuron.timeout,
                streaming=True,
            ):
                if token_count == 1:
                    end_send_message_time = time.time()
                    start_token_time = time.time()
                if isinstance(token, protocol.Inference):
                    continue
                response_tokens.append(token)
                token_count += 1

            if end_send_message_time is None:
                end_send_message_time = time.time()
                start_token_time = end_send_message_time
            end_token_time = time.time()
            time_to_first_token = end_send_message_time - start_send_message_time
            time_for_all_tokens = end_token_time - start_token_time
            tokens_per_second_partial = (
                token_count / time_for_all_tokens
                if token_count > 0 and time_for_all_tokens > 0
                else 0
            )
            tokens_per_second = tokens_per_second_partial
            response = "".join(response_tokens)

            stats.verified = check_tokens(response.split(" "), ground_truth.split(" "))
            stats.time_to_first_token = time_to_first_token
            stats.time_for_all_tokens = time_for_all_tokens
            stats.total_time = end_token_time - start_send_message_time
            stats.response = response
            stats.tokens = response_tokens
            stats.tokens_per_second = tokens_per_second
            # check if the response was pregenerated, meaning the time it takes to get the first token is much longer than the total generation
            return uid, stats
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            bt.logging.error(traceback.format_exc())
            return uid, stats

    def score(self, uid, stats: InferenceStats):
        bt.logging.trace(f"{uid}: {stats.verified} | {stats.total_time}")
        if stats.verified and stats.total_time != 0:
            self.miner_tps[uid].append(
                len(stats.response.split(" ")) / stats.total_time
            )
            return
        self.miner_tps[uid].append(0)

    async def process_uids(self, uids, messages, sampling_params, ground_truth):
        assert self.config.neuron
        try:
            bt.logging.info(
                f"Forward Block: {self.subtensor.block} |  Blocks till Set Weights: { (self.subtensor.block - self.metagraph.last_update[self.uid]) - self.config.neuron.epoch_length }"
            )
            tasks = []
            for uid in uids:
                tasks.append(
                    asyncio.create_task(
                        self.handle_inference(
                            messages, sampling_params, uid, ground_truth
                        )
                    )
                )
            stats = await asyncio.gather(*tasks)
            
            if(self.config.database_url):
                records = []
                for uid, stat in stats:
                    self.score(uid, stat)
                    records.append((uid, stat.response, "v2.0.0"))
                
                await add_records(records, self.config.database_url)

            else:
                for uid, stat in stats:
                    self.score(uid, stat)

            # TODO: insert records into db in one transaction

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            time.sleep(12)

    def run(self):
        assert self.config.subtensor
        assert self.config.neuron
        if self.sync_metagraph():
            self.resync_hotkeys()
        bt.logging.info(
            f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.subtensor.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        while not self.should_exit:
            # If startup / first loop
            if not self.next_forward_block:
                self.next_forward_block = self.subtensor.block

            # Wait random amount
            if self.next_forward_block > self.subtensor.block:
                continue

            # Declare next forward block a random time in the future so that not all valis query at the same time
            self.next_forward_block = random.randint(12, 36) + self.subtensor.block

            bt.logging.info(
                print_info(
                    self.metagraph,
                    self.wallet.hotkey.ss58_address,
                    self.subtensor.block,
                    isMiner=False,
                )
            )
            # get all miner uids
            miner_uids = self.get_miner_uids()

            # randomize miner_uids
            random.shuffle(miner_uids)

            # reduce down to 16 miners
            miner_uids = miner_uids[: self.config.neuron.sample_size]
            try:
                messages, sampling_params = asyncio.run(self.generate_question())
                ground_truth_tokens = []
                stream = self.client.chat.completions.create(
                    model=self.config.neuron.model_name,
                    messages=messages,
                    stream=True,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    seed=sampling_params.seed,
                    timeout=5,
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if not token:
                        continue
                    ground_truth_tokens.append(token)

                ground_truth = "".join(ground_truth_tokens)
            except Exception as e:
                bt.logging.error(f"Error generating dataset: {e}")
                bt.logging.error(traceback.format_exc())
                continue
            self.loop.run_until_complete(
                self.process_uids(miner_uids, messages, sampling_params, ground_truth)
            )  # Adjust batch_size as needed

            # Sync metagraph and potentially set weights.
            if self.sync_metagraph():
                self.resync_hotkeys()

            # Check if we should set weights
            if (
                self.last_posted_weights + self.config.neuron.epoch_length
                < self.subtensor.block
            ):
                self.last_posted_weights = self.subtensor.block
                self.set_weights()

    async def generate_question(self):
        assert self.config.neuron
        # Generate a random seed for reproducibility in sampling and text generation
        random.seed(urandom(100))
        seed = random.randint(10000, 10000000)

        # Determine the maximum number of new tokens to generate
        max_new_tokens = random.randint(16, 1024)

        # Create sampling parameters using the generated seed and token limit
        sampling_params = protocol.InferenceSamplingParams(
            seed=seed, max_new_tokens=max_new_tokens
        )

        # Sample a random row from the dataset and extract the text
        random_row_text = self.dataset.sample(n=1)["text"].iloc[0]

        # Generate a query from the sampled text and perform text generation
        messages = create_query_prompt(random_row_text)

        res = self.client.chat.completions.create(
            model=self.config.neuron.model_name,
            messages=messages,
            stream=False,
            temperature=0.5,
            top_p=sampling_params.top_p,
            seed=sampling_params.seed,
        )

        # Create a final search prompt using the query and sources
        completion = res.choices[0].message.content
        if completion is None:
            print(res)
            raise Exception("No completion")
        prompt = create_search_prompt(completion)

        return prompt, sampling_params

    def set_weights(self):
        assert self.config.netuid

        tokens_per_second = {
            miner: safe_mean(self.miner_tps[miner][-30:]) for miner in self.miner_tps
        }
        tps_list = list(tokens_per_second.values())
        if len(tps_list) == 0:
            bt.logging.warning("Not setting weights, no responses from miners")
            return
        top_tps = max(tps_list)
        percentile_threshold = top_tps * 0.8
        for i, v in enumerate(tps_list):
            if v < percentile_threshold:
                tps_list[i] = 0

        range_tps = top_tps - min(tps_list)
        avg_tps = np.average(tps_list)

        rewards = {}
        for uid, tps in tokens_per_second.items():
            reward_multiplier = 1
            if tps < percentile_threshold:
                tps = 0
            if tps > 0:
                normalized_difference = (tps - avg_tps) / range_tps
                reward_multiplier = math.exp(
                    normalized_difference * 10
                )  # Scale the difference to enhance reward disparity

            rewards[uid] = reward_multiplier * tps
        uids: List[int] = sorted(rewards.keys())
        rewards = [rewards[uid] for uid in uids]

        bt.logging.info(f"All tps: {tokens_per_second}")
        if sum(rewards) == 0:
            bt.logging.warning("No one gave responses worth scoring")
            return
        raw_weights = normalize(rewards)
        bt.logging.info(f"Raw Weights: {raw_weights}")

        # Set the weights on chain via our subtensor connection.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=np.asarray(uids),
            weights=np.asarray(raw_weights),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        bt.logging.info("Setting Weights: " + str(processed_weights))
        bt.logging.info("Weight Uids: " + str(processed_weight_uids))
        result, message = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=processed_weight_uids,  # type: ignore
            weights=processed_weights,
            wait_for_finalization=True,
            wait_for_inclusion=True,
            version_key=spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error(f"set_weights failed {message}")

    def resync_hotkeys(self):
        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if self.miner_tps.get(uid) == None:
                self.miner_tps[uid] = []
            if hotkey != self.metagraph.hotkeys[uid]:
                self.miner_tps[uid] = []

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def get_miner_uids(self) -> List[int]:
        """Returns all available uids from the metagraph
        Returns:
            uids (np.ndarray): Array of available uids.
        """
        available_uids = []
        assert self.config.neuron

        for uid in range(int(self.metagraph.n.item())):
            if uid == self.uid:
                continue

            # Filter non serving axons.
            if not self.metagraph.axons[uid].is_serving:
                continue
            # Filter validator permit > 1024 stake.
            if self.metagraph.validator_permit[uid]:
                if self.metagraph.S[uid] > self.config.neuron.vpermit_tao_limit:
                    continue
            available_uids.append(uid)
            continue
        return available_uids


if __name__ == "__main__":
    try:
        validator = Validator()
        validator.run()
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
