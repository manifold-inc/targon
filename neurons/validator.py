import sys
import json
import copy
import time
import random
import asyncio
from targon.utils import normalize, print_info, safe_mean, InferenceStats, check_tokens
import traceback
import math
import argparse
import numpy as np
import pandas as pd
import bittensor as bt
import signal
from openai import OpenAI

from typing import List
from transformers import AutoTokenizer
from targon import (
    generate_dataset,
    add_args,
    add_validator_args,
    protocol,
    validate_config_and_neuron_path,
    __spec_version__ as spec_version,
)
from bittensor.utils.weight_utils import (
    process_weights_for_netuid,
)


class Validator:
    neuron_type = "ValidatorNeuron"
    config: "bt.config"

    def exit_gracefully(self, *_):
        if self.should_exit:
            bt.logging.info("Forcefully exiting")
            exit()
        bt.logging.info("Exiting Gracefully at end of cycle")
        self.should_exit = True

    def __init__(self, config=None):
        ## ADD CONFIG

        parser = argparse.ArgumentParser()
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.axon.add_args(parser)
        add_args(parser)
        add_validator_args(parser)
        self.config = bt.config(parser)
        if config:
            base_config = copy.deepcopy(config)
            self.config.merge(base_config)
        validate_config_and_neuron_path(self.config)

        ## Typesafety
        assert self.config.netuid
        assert self.config.neuron
        assert self.config.logging
        assert self.config.axon

        ## Add kill signals
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        ## LOGGING
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.on()
        if self.config.logging.debug:
            bt.logging.set_debug(True)
        if self.config.logging.trace:
            bt.logging.set_trace(True)
        bt.turn_console_on()

        ## BITTENSOR INITIALIZATION
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.loop = asyncio.get_event_loop()
        bt.logging.debug(f"Wallet: {self.wallet}")
        bt.logging.debug(f"Subtensor: {self.subtensor}")
        bt.logging.debug(f"Metagraph: {self.metagraph}")

        ## CHECK IF REGG'D
        self.check_registered()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        if not self.metagraph.validator_permit[self.uid]:
            bt.logging.error("Validator does not have vpermit")

        ## SET MISC PARAMS
        self.should_exit = False
        self.last_synced_block = None
        self.hotkeys = self.metagraph.hotkeys
        self.last_forward_block = None
        self.last_posted_weights = self.subtensor.block

        ## STATS
        miners = self.get_miner_uids()
        self.time_to_first_token = {miner: [] for miner in miners}
        self.time_for_all_tokens = {miner: [] for miner in miners}
        self.tokens_per_second = {miner: [] for miner in miners}
        self.verified_success = {miner: [] for miner in miners}
        self.top_verified_tps = 0
        self.top_unverified_tps = 0

        ## SET DATASET
        self.dataset = pd.read_json(
            "hf://datasets/pinecone/dl-doc-search/train.jsonl", lines=True
        )

        ## SET CLIENT
        self.client = OpenAI(
            base_url=self.config.neuron.model_endpoint,
            api_key=self.config.neuron.api_key,
        )

        ## SET PROMPT TOKENIZER
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(
            self.config.neuron.default_tokenizer
        )
        bt.logging.info(
            "\N{grinning face with smiling eyes}", "Successfully Initialized!"
        )

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
                for t in token:
                    response_tokens.append(t)
                token_count += 1
            if token_count <= 1 or len(response_tokens) <= 1:
                return None
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

            # check if the response was pregenerated, meaning the time it takes to get the first token is much longer than the total generation
            verified = True
            if time_to_first_token > 1.8 * time_for_all_tokens:
                verified = False
                tokens_per_second = 0
            if verified:
                verified = check_tokens(response, ground_truth)
            stats = InferenceStats(
                time_to_first_token=time_to_first_token,
                time_for_all_tokens=time_for_all_tokens,
                tokens_per_second=tokens_per_second,
                tokens=response_tokens,
                response=response,
                verified=verified,
                uid=uid,
            )
            return stats
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            bt.logging.error(traceback.format_exc())
            return None

    def score(self, stats):
        if stats is None:
            return
        self.top_unverified_tps = max(self.top_unverified_tps, stats.tokens_per_second)
        if not stats.verified:
            return
        self.time_to_first_token[stats.uid].append(stats.time_to_first_token)
        self.time_for_all_tokens[stats.uid].append(stats.time_for_all_tokens)
        self.tokens_per_second[stats.uid].append(stats.tokens_per_second)
        self.verified_success[stats.uid].append(stats.verified)
        self.top_verified_tps = max(self.top_verified_tps, stats.tokens_per_second)

    def stats(self):
        time_to_first_token_stats = {
            miner: safe_mean(self.time_to_first_token[miner][:30])
            for miner in self.time_to_first_token
        }
        time_for_all_tokens_stats = {
            miner: safe_mean(self.time_for_all_tokens[miner][:30])
            for miner in self.time_for_all_tokens
        }
        tokens_per_second_stats = {
            miner: safe_mean(self.tokens_per_second[miner][:30])
            for miner in self.tokens_per_second
        }
        verified_success_stats = {
            miner: safe_mean(self.verified_success[miner][:30])
            for miner in self.verified_success
        }

        mean_tps_dict = {
            uid: safe_mean(self.tokens_per_second[uid][:30])
            for uid in self.tokens_per_second
        }
        mean_tps_dict = dict(
            sorted(mean_tps_dict.items(), key=lambda item: item[1], reverse=True)
        )
        top_20_uids = dict(list(mean_tps_dict.items())[:20])
        return {
            "top_verified_tps": self.top_verified_tps,
            "top_unverified_tps": self.top_unverified_tps,
            "top_uids": top_20_uids,
            "miners": {
                uid: {
                    "top_tps": max(self.tokens_per_second[uid])
                    if len(self.tokens_per_second[uid]) > 0
                    else 0,
                    "mean_time_to_first_token": time_to_first_token_stats[uid],
                    "mean_time_for_all_tokens": time_for_all_tokens_stats[uid],
                    "mean_tokens_per_second": tokens_per_second_stats[uid],
                    "mean_verified_success": verified_success_stats[uid],
                }
                for uid in self.time_to_first_token
            },
        }

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

            for stat in stats:
                self.score(stat)

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            time.sleep(12)

    def run(self):
        assert self.config.subtensor
        assert self.config.neuron
        self.sync()
        bt.logging.info(
            f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.subtensor.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        while not self.should_exit:
            # Print Vali Info every few blocks
            if self.last_forward_block == self.subtensor.block:
                continue
            if not self.subtensor.block % 12 == 0:
                continue
            bt.logging.info(
                print_info(
                    self.metagraph,
                    self.wallet.hotkey.ss58_address,
                    self.subtensor.block,
                    isMiner=False,
                )
            )
            self.last_forward_block = self.subtensor.block
            # get all miner uids
            miner_uids = self.get_miner_uids()

            # randomize miner_uids
            random.shuffle(miner_uids)

            # reduce down to 16 miners
            miner_uids = miner_uids[: self.config.neuron.sample_size]
            try:
                messages, sampling_params = asyncio.run(generate_dataset(self))
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
                time.sleep(12)
                continue
            self.loop.run_until_complete(
                self.process_uids(miner_uids, messages, sampling_params, ground_truth)
            )  # Adjust batch_size as needed

            # Sync metagraph and potentially set weights.
            self.sync()

    def should_sync_metagraph(self):
        assert self.config.neuron
        if self.last_synced_block is None:
            self.last_synced_block = self.subtensor.block
            return True

        if self.subtensor.block % 90 != 0:
            return False
        if self.last_synced_block == self.subtensor.block:
            return False
        self.last_synced_block = self.subtensor.block
        return True

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        assert self.config.neuron
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        # If metagraph is out of date
        if (
            (self.subtensor.block - self.metagraph.last_update[self.uid])
            > self.config.neuron.epoch_length
            and self.neuron_type != "MinerNeuron"
            and self.last_posted_weights + 20 < self.subtensor.block
        ):
            self.set_weights()
            self.resync_metagraph()

    def set_weights(self):
        assert self.config.neuron
        assert self.config.netuid

        tokens_per_second = {
            miner: safe_mean(self.tokens_per_second[miner][:30])
            for miner in self.tokens_per_second
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

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # TODO
        if sum(rewards) == 0:
            bt.logging.warning("No one gave responses worth scoring")
            return
        raw_weights = normalize(rewards)

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
            uids=processed_weight_uids, #type: ignore
            weights=processed_weights,
            wait_for_finalization=True,
            wait_for_inclusion=True,
            version_key=spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error(f"set_weights failed {message}")

    def check_registered(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit()

    def resync_metagraph(self):
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.time_to_first_token[uid] = []
                self.time_for_all_tokens[uid] = []
                self.tokens_per_second[uid] = []
                self.verified_success[uid] = []

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

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
                bt.logging.debug(f"uid: {uid} is not serving")
                continue
            # Filter validator permit > 1024 stake.
            if self.metagraph.validator_permit[uid]:
                bt.logging.debug(f"uid: {uid} has validator permit")
                if self.metagraph.S[uid] > self.config.neuron.vpermit_tao_limit:
                    bt.logging.debug(
                        f"uid: {uid} has stake ({self.metagraph.S[uid]}) > {self.config.neuron.vpermit_tao_limit}"
                    )
                    continue
            available_uids.append(uid)
            continue
        return available_uids


if __name__ == "__main__":
    validator = Validator()
    validator.run()
    exit()
