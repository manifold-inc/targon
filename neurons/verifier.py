import sys
import copy
import time
import random
import asyncio
import uvicorn
import argparse
import numpy as np
import pandas as pd
import bittensor as bt

from typing import List
from fastapi import FastAPI
from transformers import AutoTokenizer
from bittensor.axon import FastAPIThreadedServer
from huggingface_hub import AsyncInferenceClient
from targon import (
    generate_dataset,
    create_ground_truth,
    handle_inference,
    add_args,
    add_verifier_args,
    validate_config_and_neuron_path,
)


class Verifier:
    neuron_type = "VerifierNeuron"
    config: "bt.config"

    @classmethod
    def check_config(cls, config):
        validate_config_and_neuron_path(config)

    @classmethod
    def add_args(cls, parser):
        add_args(parser)
        add_verifier_args(parser)

    @classmethod
    def _config(cls):
        parser = argparse.ArgumentParser()
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.axon.add_args(parser)
        cls.add_args(parser)
        config = bt.config(parser)
        return config

    @property
    def block(self):
        return self.subtensor.block

    def __init__(self, config=None):
        ## ADD CONFIG
        self.config = self._config()
        if config:
            base_config = copy.deepcopy(config)
            self.config.merge(base_config)
        self.check_config(self.config)
        print(self.config)

        ## Typesafety
        assert self.config.full_path
        assert self.config.netuid
        assert self.config.neuron
        assert self.config.logging

        ## LOGGING
        bt.logging(config=self.config, logging_dir=self.config.full_path)
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
        self.axon = bt.axon(config=self.config)

        bt.logging.debug(f"Wallet: {self.wallet}")
        bt.logging.debug(f"Subtensor: {self.subtensor}")
        bt.logging.debug(f"Metagraph: {self.metagraph}")

        ## CHECK IF REGG'D
        self.check_registered()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        ## SET MISC PARAMS
        self.step = 0
        self.should_exit = False

        self.hotkeys = self.metagraph.hotkeys

        ## STATS
        miners = self.get_miner_uids()
        self.time_to_first_token = {miner: [] for miner in miners}
        self.time_for_all_tokens = {miner: [] for miner in miners}
        self.tokens_per_second = {miner: [] for miner in miners}

        self.verified_success = {miner: [] for miner in miners}

        self.top_verified_tps = 0
        self.top_unverified_tps = 0

        ## STATS SERVER
        self.app = FastAPI()
        self.app.router.add_api_route("/api/stats", self.stats, methods=["GET"])
        self.fast_config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.config.neuron.proxy.port,
            loop="asyncio",
        )
        self.fast_server = FastAPIThreadedServer(config=self.fast_config)
        self.fast_server.start()

        ## SET DATASET
        self.dataset = pd.read_json(
            "hf://datasets/pinecone/dl-doc-search/train.jsonl", lines=True
        )

        ## SET CLIENT
        self.client = AsyncInferenceClient(self.config.neuron.model_endpoint)

        ## SET PROMPT TOKENIZER
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(
            self.config.neuron.default_tokenizer
        )

    async def forward(self, uids, prompt, sampling_params, ground_truth):
        """
        Verifier forward pass. Consists of:
        - Generating the query
        - Querying the provers
        - Getting the responses
        - Rewarding the provers
        - Updating the scores
        """
        print("forward()")
        assert self.config.neuron
        if self.config.neuron.api_only:
            bt.logging.info("Running in API only mode, sleeping for 12 seconds.")
            time.sleep(12)
            return
        try:
            bt.logging.info(f"forward block: {self.block} step: {self.step}")
            tasks = []
            for uid in uids:
                tasks.append(
                    asyncio.create_task(
                        handle_inference(
                            self, prompt, sampling_params, uid, ground_truth
                        )
                    )
                )
            # stats = await handle_inference(self, prompt, sampling_params, uid, ground_truth)
            stats = await asyncio.gather(*tasks)

            for stat in stats:
                self.score(stat)

            bt.logging.info(str(stats))
            # await self.score(stats)

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            time.sleep(12)

    def score(self, stats):
        if stats.verified:
            self.time_to_first_token[stats.uid].append(stats.time_to_first_token)
            self.time_for_all_tokens[stats.uid].append(stats.time_for_all_tokens)
            self.tokens_per_second[stats.uid].append(stats.tokens_per_second)

            self.verified_success[stats.uid].append(stats.verified)

            self.top_verified_tps = max(self.top_verified_tps, stats.tokens_per_second)

        else:
            self.time_to_first_token[stats.uid].append(0)
            self.time_for_all_tokens[stats.uid].append(0)
            self.tokens_per_second[stats.uid].append(0)

            self.verified_success[stats.uid].append(False)
            self.top_unverified_tps = max(
                self.top_unverified_tps, stats.tokens_per_second
            )

    def stats(self):
        def safe_mean(data):
            mean_value = np.mean(data)
            if np.isnan(mean_value) or np.isinf(mean_value):
                return 0.0
            return float(mean_value)

        time_to_first_token_stats = {
            miner: safe_mean(
                self.time_to_first_token[miner][:30]
                if len(self.time_to_first_token[miner]) > 30
                else self.time_to_first_token[miner]
            )
            for miner in self.time_to_first_token
        }
        time_for_all_tokens_stats = {
            miner: safe_mean(
                self.time_for_all_tokens[miner][:30]
                if len(self.time_for_all_tokens[miner]) > 30
                else self.time_for_all_tokens[miner]
            )
            for miner in self.time_for_all_tokens
        }
        tokens_per_second_stats = {
            miner: safe_mean(
                self.tokens_per_second[miner][:30]
                if len(self.tokens_per_second[miner]) > 30
                else self.tokens_per_second[miner]
            )
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

    async def process_uids(self, uids, prompt, sampling_params, ground_truth):
        try:
            await self.forward(uids, prompt, sampling_params, ground_truth)
        except Exception as e:
            bt.logging.error(f"Error processing uids: {e}")

    def run(self):
        assert self.config.subtensor
        assert self.config.neuron
        self.sync()
        bt.logging.info(
            f"Running verifier {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Verifier starting at block: {self.block}")

        # This loop maintains the verifier's operations until intentionally stopped.
        while not self.should_exit:
            # get all miner uids
            miner_uids = self.get_miner_uids()
            # miner_uids = [245]
            # randomize miner_uids
            random.shuffle(miner_uids)

            # reduce down to 16 miners
            miner_uids = miner_uids[: self.config.neuron.sample_size]
            # miner_uids = [107, 122]
            try:
                prompt, sampling_params = asyncio.run(generate_dataset(self))
                ground_truth = asyncio.run(
                    create_ground_truth(self, prompt, sampling_params)
                )
            except Exception as e:
                bt.logging.error(f"Error generating dataset: {e}")
                time.sleep(12)
                continue
            bt.logging.info(f"number of uids to sample: {len(miner_uids)}")
            self.loop.run_until_complete(
                self.process_uids(miner_uids, prompt, sampling_params, ground_truth)
            )  # Adjust batch_size as needed

            # Sync metagraph and potentially set weights.
            self.sync()

            self.step += 1

    def __enter__(self):
        self.run()

    def __exit__(self, *_):
        pass

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given prover or verifier.
        """
        # Ensure prover or verifier hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        # if self.should_set_weights():
        #     self.set_weights()

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        assert self.config.neuron
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        assert self.config.neuron
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length and self.neuron_type != "ProverNeuron"

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
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
        # if len(self.hotkeys) < len(self.metagraph.hotkeys):
        #     # Update the size of the moving average scores.
        #     new_moving_average = np.zeros((self.metagraph.n))
        #     min_len = min(len(self.hotkeys), len(self.scores))
        #     new_moving_average[:min_len] = self.scores[:min_len]
        #     self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def check_uid_availability(
        self,
        metagraph: "bt.metagraph",
        uid: int,
        vpermit_tao_limit: int,
        mock: bool = False,
    ) -> bool:
        """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
        Args:
            metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
            uid (int): uid to be checked
            vpermit_tao_limit (int): Verifier permit tao limit
        Returns:
            bool: True if uid is available, False otherwise
        """
        if not mock:
            # Filter non serving axons.
            if not metagraph.axons[uid].is_serving:
                bt.logging.debug(f"uid: {uid} is not serving")
                return False
            # Filter verifier permit > 1024 stake.
            if metagraph.validator_permit[uid]:
                bt.logging.debug(f"uid: {uid} has verifier permit")
                if metagraph.S[uid] > vpermit_tao_limit:
                    bt.logging.debug(
                        f"uid: {uid} has stake ({metagraph.S[uid]}) > {vpermit_tao_limit}"
                    )
                    return False
        else:
            return True

        # Available otherwise.
        return True

    def get_miner_uids(self) -> List[int]:
        """Returns all available uids from the metagraph
        Returns:
            uids (np.ndarray): Array of available uids.
        """
        available_uids = []
        assert self.config.neuron
        assert self.config.mock

        for uid in range(int(self.metagraph.n.item())):
            if uid == self.uid:
                continue
            uid_is_available = self.check_uid_availability(
                self.metagraph,
                uid,
                self.config.neuron.vpermit_tao_limit,
                self.config.mock,
            )
            if uid_is_available:
                available_uids.append(uid)
        return available_uids


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # Verifier.add_args(parser)
    # args = parser.parse_args()
    with Verifier() as verifier:
        while True:
            bt.logging.info("Verifier running...", str(time.time()))
            time.sleep(5)
