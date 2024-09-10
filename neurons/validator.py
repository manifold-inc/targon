from os import urandom
from openai.types.chat import ChatCompletionMessageParam
from requests import post
import json
import copy
import time
import random
import asyncio

from asyncpg.connection import asyncpg
from bittensor.dendrite import aiohttp
import openai
from neurons.base import BaseNeuron, NeuronType
from targon.dataset import create_query_prompt, create_search_prompt
from targon.epistula import generate_header
from targon.updater import autoupdate
from targon.utils import (
    fail_with_none,
    normalize,
    print_info,
    safe_mean_score,
)
from targon.protocol import Endpoints, InferenceSamplingParams, InferenceStats
import traceback
import numpy as np
import dask.dataframe as dd
import bittensor as bt
from nanoid import generate

from typing import Any, Dict, List, Optional, Tuple, Union
from targon import (
    protocol,
    __version__,
    __spec_version__ as spec_version,
)
from bittensor.utils.weight_utils import (
    process_weights_for_netuid,
)

INGESTOR_URL = "http://177.54.155.247:8000"


class Validator(BaseNeuron):
    miner_tps: Dict[int, Any]
    db_organics: Optional[asyncpg.Connection]
    neuron_type = NeuronType.Validator

    def __init__(self, config=None):
        super().__init__(config)
        ## Typesafety
        assert self.config.netuid
        assert self.config.neuron
        assert self.config.axon
        assert self.config.database

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
        bt.logging.info(f"Last updated at block {self.last_posted_weights}")

        ## STATS
        self.miner_tps = {}

        try:
            with open(self.config.neuron.cache_file, "r") as file:
                loaded_data: Dict[str, Any] = json.load(file)
                # Only load cache if fresh
                if loaded_data.get("block_saved", 0) > self.subtensor.block - 360:
                    miner_tps: Dict[str, List[float]] = loaded_data.get("miner_tps", {})
                    self.miner_tps = dict([(int(k), v) for k, v in miner_tps.items()])
                    bt.logging.info("Loading cached data")
                    bt.logging.trace(str(self.miner_tps))
        except IOError:
            bt.logging.info("No cache file found")
        except EOFError:
            bt.logging.warning("Curropted pickle file")
        except Exception as e:
            bt.logging.error(f"Failed reading cache file: {e}")
            bt.logging.error(traceback.format_exc())

        miners = self.get_miner_uids()
        for miner in miners:
            if self.miner_tps.get(miner) == None:
                self.miner_tps[miner] = []

        ## SET DATASET
        bt.logging.info("⌛️", "Loading dataset")
        # @CARRO / @josh todo
        # choose multiple datasets
        df = dd.read_parquet(
            "hf://datasets/manifoldlabs/Infinity-Instruct/7M/*.parquet"
        )
        self.dataset = df.compute()

        bt.logging.info(
            "\N{grinning face with smiling eyes}", "Successfully Initialized!"
        )
        try:
            self.db_organics = None
            if self.config.database.organics_url:
                self.db_organics = self.loop.run_until_complete(
                    asyncpg.connect(self.config.database.organics_url)
                )
        except Exception as e:
            bt.logging.error(f"Failed to initialize organics database: {e}")

    async def send_stats_to_ingestor(self, stats, sampling_params, messages, endpoint):
        try:
            r_nanoid = generate(size=48)
            # @SAROKAN make sure the new fields line up w/ DB and injestor.
            responses = [
                {
                    "r_nanoid": r_nanoid,
                    "hotkey": self.metagraph.axons[uid].hotkey,
                    "coldkey": self.metagraph.axons[uid].coldkey,
                    "uid": int(uid),
                    "stats": stat.model_dump(),
                }
                for uid, stat in stats
            ]
            vali_request = {"prompt": messages}
            if isinstance(messages, str):
                vali_request = {"messages": messages}
            request = {
                "r_nanoid": r_nanoid,
                "block": self.subtensor.block,
                "sampling_params": sampling_params.model_dump(),
                "request": vali_request,
                "request_endpoint": endpoint,
                "version": spec_version,
                "hotkey": self.wallet.hotkey.ss58_address,
            }
            # Prepare the data
            body = {
                "request": request,
                "responses": responses,
            }
            headers = generate_header(self.wallet.hotkey, body)
            # Send request to the FastAPI server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{INGESTOR_URL}/ingest", headers=headers, json=body
                ) as response:
                    if response.status == 200:
                        bt.logging.info("Records ingested successfully.")
                    else:
                        error_detail = await response.text()
                        bt.logging.error(
                            f"Error sending records: {response.status} - {error_detail}"
                        )

        except Exception as e:
            bt.logging.error(f"Error in send_stats_to_ingestor: {e}")
            bt.logging.error(traceback.format_exc())

    def run(self):
        assert self.config.subtensor
        assert self.config.neuron
        assert self.config.database
        if self.sync_metagraph():
            self.resync_hotkeys()
        bt.logging.info(
            f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.subtensor.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        step = 0
        miner_subset = 36
        miner_uids = self.get_miner_uids()
        random.shuffle(miner_uids)
        miner_uids = miner_uids[:miner_subset]
        while not self.should_exit:
            blocks_till = self.config.neuron.epoch_length - (
                self.subtensor.block % self.config.neuron.epoch_length
            )
            if self.last_posted_weights != self.subtensor.block:
                bt.logging.info(
                    f"Forward Block: {self.subtensor.block} | Step {step} |  Blocks till Set Weights: {blocks_till}"
                )

            # Set weights
            # Gives wiggle room for out of sync validators
            if (
                self.subtensor.block % self.config.neuron.epoch_length == 0
                or self.last_posted_weights + (self.config.neuron.epoch_length * 2)
                < self.subtensor.block
            ):
                if self.last_posted_weights == self.subtensor.block:
                    continue
                bt.logging.info(
                    f"Last set weights: {self.last_posted_weights}, current: {self.subtensor.block}"
                )
                self.last_posted_weights = self.subtensor.block

                # Sync metagraph before setting weights
                if self.sync_metagraph():
                    self.resync_hotkeys()
                self.set_weights()

                # Only keep last 15 scores
                for uid in self.miner_tps.keys():
                    self.miner_tps[uid] = self.miner_tps[uid][-15:]

            # Stop querying if close to weight set block
            if blocks_till < 5 or blocks_till == self.config.neuron.epoch_length:
                continue

            # Sync metagraph if needed
            if self.sync_metagraph():
                self.resync_hotkeys()

            # Check to see if we need to update
            if self.config.autoupdate:
                autoupdate(branch="main")

            # Score organic queries every few steps
            # TODO: Readd organics
            # if not step % 25 and self.config.database.organics_url:
            # self.loop.run_until_complete(self.score_organic())

            print_info(
                self.metagraph,
                self.wallet.hotkey.ss58_address,
                self.subtensor.block,
                isMiner=False,
            )

            # get random set of miner uids every other step
            if step % 2:
                miner_uids = self.get_miner_uids()
                random.shuffle(miner_uids)
                miner_uids = miner_uids[:miner_subset]
            res = self.loop.run_until_complete(self.query_miners(miner_uids))
            self.loop.run_until_complete(self.send_stats_to_ingestor(*res))
            self.save_scores()
            step += 1

        # Exiting
        self.shutdown()

    async def query_miners(self, miner_uids):
        assert self.config.database
        endpoint = random.choice(list(Endpoints))
        messages, sampling_params = self.generate_question(endpoint)
        if messages is None or sampling_params is None:
            return None
        tasks = []
        async with aiohttp.ClientSession() as session:
            for uid in miner_uids:
                tasks.append(
                    asyncio.create_task(
                        self.handle_inference(messages, sampling_params, uid, session, endpoint)
                    )
                )
            stats: List[Tuple[int, InferenceStats]] = await asyncio.gather(*tasks)
        for uid, stat in stats:
            bt.logging.info(f"{uid}: {stat.verified} | {stat.total_time}")
            if stat.verified and stat.total_time != 0:
                self.miner_tps[uid].append(stat.tps)
                continue
            self.miner_tps[uid].append(None)
        return (
            stats,
            sampling_params,
            messages,
            endpoint
        )

    async def handle_inference(
        self,
        messages,
        sampling_params: InferenceSamplingParams,
        uid: int,
        session: aiohttp.ClientSession,
        endpoint: Endpoints
    ):
        assert self.config.neuron
        assert sampling_params.max_tokens
        stats = InferenceStats(
            time_to_first_token=0,
            time_for_all_tokens=0,
            tps=0,
            total_time=0,
            tokens=[],
            response="",
            verified=False,
        )
        try:
            response_tokens = []
            token_count = 0
            end_send_message_time = None
            start_token_time = 0

            axon_info = self.metagraph.axons[uid]
            match endpoint:
                case Endpoints.CHAT:
                    url = f"http://{axon_info.ip}:{axon_info.port}/v1/chat/completion"
                    body = {"messages": messages, **sampling_params.model_dump()}
                case Endpoints.COMPLETION:
                    url = f"http://{axon_info.ip}:{axon_info.port}/v1/completion"
                    body = {"prompt": messages, **sampling_params.model_dump()}
            headers = generate_header(self.wallet.hotkey, body, axon_info.hotkey)
            start_send_message_time = time.time()
            timeout = aiohttp.ClientTimeout(
                total=20, # Total time (s)
                sock_connect=5, # Time to Connect (s)
                sock_read=5 # Time to read first byte (s)
            )
            try:
                async with session.post(
                    url=url,
                    headers=headers,
                    json=body,
                    timeout=timeout,
                ) as r:
                    if r.status != 200:
                        raise Exception(f"{r.status}: {await r.text()}")
                    async for chunk in r.content.iter_any():
                        if token_count == 1:
                            end_send_message_time = time.time()
                            start_token_time = time.time()
                        print(chunk)
                        token = chunk.decode()
                        print("decode:", chunk)
                        response_tokens.append(token)
                        token_count += 1
            except Exception as e:
                bt.logging.trace(f"Miner failed request: {e}")

            if end_send_message_time is None:
                end_send_message_time = time.time()
                start_token_time = end_send_message_time
            end_token_time = time.time()
            time_to_first_token = end_send_message_time - start_send_message_time
            time_for_all_tokens = end_token_time - start_token_time
            response = "".join(response_tokens)

            verified = self.check_tokens(messages, stats.tokens, endpoint=endpoint)
            stats.verified = verified or False
            stats.time_to_first_token = time_to_first_token
            stats.time_for_all_tokens = time_for_all_tokens
            stats.total_time = end_token_time - start_send_message_time
            stats.response = response
            stats.tps = (
                min(len(stats.tokens), sampling_params.max_tokens)
                / stats.total_time
            )
            return uid, stats
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            bt.logging.error(traceback.format_exc())
            return uid, stats

    @fail_with_none("Failed writing to cache file")
    def save_scores(self):
        assert self.config.neuron
        with open(self.config.neuron.cache_file, "w") as file:
            bt.logging.info("Caching scores...")
            json.dump(
                {
                    "miner_tps": self.miner_tps,
                    "block_saved": self.subtensor.block,
                    "version": spec_version,
                },
                file,
            )
            file.flush()
            bt.logging.info("Cached")


    @fail_with_none("Failed to check tokens")
    def check_tokens(
        self,
        prompt: Union[str, List[ChatCompletionMessageParam]],
        response: List[Tuple[str, int]],
        endpoint: Endpoints = Endpoints.CHAT,
    ) -> Optional[bool]:
        assert self.config.neuron
        response_string = ""
        index = len(response)
        for i in range(index):
            response_string += response[i][0]
        powv = response[index][1]
        match endpoint:
            case Endpoints.CHAT:
                assert isinstance(prompt, dict)
                res = post(
                    self.config.neuron.model_endpoint + "/chat/completions/verify",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(
                        {
                            "messages": prompt,
                            "model": self.config.neuron.model_name,
                            "response": response_string,
                            "powv": powv,
                        }
                    ),
                )
                return res.json()
            case Endpoints.COMPLETION:
                assert isinstance(prompt, str)
                res = post(
                    self.config.neuron.model_endpoint + "/completions/verify",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(
                        {
                            "prompt": prompt,
                            "model": self.config.neuron.model_name,
                            "response": response_string,
                            "powv": powv,
                        }
                    ),
                )
                return res.json()
            case _:
                raise Exception(f"Unknown Endpoint {endpoint}")

    def shutdown(self):
        if self.db_organics:
            bt.logging.info("Closing organics db connection")
            self.loop.run_until_complete(self.db_organics.close())

    def generate_question(self, endpoint: Endpoints):
        try:
            assert self.config.neuron
            # Generate a random seed for reproducibility in sampling and text generation
            random.seed(urandom(100))
            seed = random.randint(10000, 10000000)
            temperature = random.random() * 2
            max_tokens= random.randint(1024, 1024 * 15)

            # Create sampling parameters using the generated seed and token limit
            sampling_params = protocol.InferenceSamplingParams(
                seed=seed, max_tokens=max_tokens, temperature=temperature
            )

            random_row_text = self.dataset.sample(n=1)["conversations"].iloc[0][0][
                "value"
            ]
            # Generate a query from the sampled text and perform text generation
            messages = create_query_prompt(random_row_text)

            # If this fails, it gets caught in the same try/catch as ground truth generation
            res = self.client.chat.completions.create(
                model=self.config.neuron.model_name,
                messages=messages,
                stream=False,
                temperature=0.5,
                top_p=sampling_params.top_p,
                seed=sampling_params.seed,
                max_tokens=random.randint(16, 64),
            )

            # Create a final search prompt using the query and sources
            completion = res.choices[0].message.content
            if completion is None:
                bt.logging.error(str(res))
                raise Exception("No completion")
            prompt = create_search_prompt(completion, endpoint)

            return prompt, sampling_params
        except openai.APIConnectionError as e:
            bt.logging.error(
                f"Failed to connect to LLM server with connection string {self.client.base_url}: {e.message}"
            )
            bt.logging.error(
                "Make sure an open ai compliant server is running at the above url, or fix --neuron.model_endpoint"
            )
            return None, None
        except Exception as e:
            bt.logging.error(f"Error generating dataset: {e}")
            bt.logging.error(traceback.format_exc())
            return None, None

    @fail_with_none("Failed getting Weights")
    def get_weights(self) -> Tuple[List[int], List[float]]:
        tps = {
            miner: safe_mean_score(self.miner_tps[miner][-15:])
            for miner in self.miner_tps
        }
        tps_list = list(tps.values())
        if len(tps_list) == 0:
            bt.logging.warning("Not setting weights, no responses from miners")
            return [], []
        uids: List[int] = sorted(tps.keys())
        rewards = [tps[uid] for uid in uids]

        bt.logging.info(f"All wps: {tps}")
        if sum(rewards) < 1 / 1e9:
            bt.logging.warning("No one gave responses worth scoring")
            return [], []
        raw_weights = normalize(rewards)
        bt.logging.info(f"Raw Weights: {raw_weights}")
        return uids, raw_weights

    @fail_with_none("Failed setting weights")
    def set_weights(self):
        assert self.config.netuid
        weights = self.get_weights()
        if weights is None:
            return None
        uids, raw_weights = weights
        if not len(uids):
            bt.logging.info("No UIDS to score")
            return

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
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=spec_version,
            max_retries=1,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error(f"set_weights failed {message}")

    @fail_with_none("Failed resyncing hotkeys")
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
