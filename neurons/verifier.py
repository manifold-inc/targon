import sys
import copy
import time
import random
import asyncio
import uvicorn
import argparse
import numpy as np
import pandas as pd
import plotext as plt
import bittensor as bt

from typing import List
from fastapi import FastAPI
from transformers import AutoTokenizer
from bittensor.axon import FastAPIThreadedServer
from huggingface_hub import AsyncInferenceClient
from targon import config, check_config, add_args, add_verifier_args
from targon import generate_dataset, create_ground_truth, handle_inference, InferenceStats


class Verifier:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)
        add_verifier_args(cls, parser) 

    @classmethod
    def _config(cls):
        parser = argparse.ArgumentParser()
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.axon.add_args(parser)
        cls.add_args(parser)
        return bt.config(parser)
    
    @property
    def block(self):
        return self.subtensor.block
    
    def __init__(self, config=None):

        ## ADD CONFIG
        base_config = copy.deepcopy(config or self._config())
        self.config = self._config()
        self.config.merge(base_config)
        self.check_config(self.config)
        print(self.config)

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
        self.app.router.add_api_route(
            "/api/stats", self.stats, methods=["GET"]
        )
        self.fast_config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.config.neuron.proxy.port,
            loop="asyncio",
        )
        self.fast_server = FastAPIThreadedServer(config=self.fast_config)
        self.fast_server.start()

        ## SET DATASET
        self.dataset = pd.read_json("hf://datasets/pinecone/dl-doc-search/train.jsonl", lines=True)


        ## SET TGI CLIENT
        self.client = AsyncInferenceClient(self.config.neuron.tgi_endpoint)


        ## SET PROMPT TOKENIZER
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(self.config.neuron.default_tokenizer)
        



    async def forward(self, uid, prompt, sampling_params, ground_truth):
        """
        Verifier forward pass. Consists of:
        - Generating the query
        - Querying the provers
        - Getting the responses
        - Rewarding the provers
        - Updating the scores
        """
        print("forward()")
        if self.config.neuron.api_only:
            bt.logging.info("Running in API only mode, sleeping for 12 seconds.")
            time.sleep(12)
            return
        try:
            bt.logging.info(f"forward block: {self.block if not self.config.mock else self.block_number} step: {self.step}")

            stats = await handle_inference(self, prompt, sampling_params, uid, ground_truth)


            bt.logging.info(stats)
            await self.score(stats)


            
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            time.sleep(12)
            pass

    async def score(self, stats):
        if stats.verified:
            self.time_to_first_token[stats.uid].append(stats.time_to_first_token)
            self.time_for_all_tokens[stats.uid].append(stats.time_for_all_tokens)
            self.tokens_per_second[stats.uid].append(stats.tokens_per_second)

            self.verified_success[stats.uid].append(stats.verified)

            self.top_verified_tps = max(self.top_verified_tps, stats.tokens_per_second)
        
        self.top_unverified_tps = max(self.top_unverified_tps, stats.tokens_per_second)
        # else:
        #     self.time_to_first_token[stats.uid].append(0)
        #     self.time_for_all_tokens[stats.uid].append(0)
        #     self.tokens_per_second[stats.uid].append(0)

        #     self.verified_success[stats.uid].append(False)
        


    async def stats(self):
        def safe_mean(data):
            mean_value = np.mean(data)
            if np.isnan(mean_value) or np.isinf(mean_value):
                return 0.0
            return float(mean_value)
        
        time_to_first_token_stats = {miner: safe_mean(self.time_to_first_token[miner]) for miner in self.time_to_first_token}
        time_for_all_tokens_stats = {miner: safe_mean(self.time_for_all_tokens[miner]) for miner in self.time_for_all_tokens}
        tokens_per_second_stats = {miner: safe_mean(self.tokens_per_second[miner]) for miner in self.tokens_per_second}
        verified_success_stats = {miner: safe_mean(self.verified_success[miner]) for miner in self.verified_success}
        
        return {
            "top_verified_tps": self.top_verified_tps,
            "top_unverified_tps": self.top_unverified_tps,
            "miners": {uid: 
                        {
                            "top_tps": max(self.tokens_per_second[uid]) if len(self.tokens_per_second[uid]) > 0 else 0,
                            "mean_time_to_first_token": time_to_first_token_stats[uid], 
                            "mean_time_for_all_tokens": time_for_all_tokens_stats[uid], 
                            "mean_tokens_per_second": tokens_per_second_stats[uid], 
                            "mean_verified_success": verified_success_stats[uid]
                        } for uid in self.time_to_first_token}
            # "time_to_first_token": time_to_first_token_stats,
            # "time_for_all_tokens": time_for_all_tokens_stats,
            # "tokens_per_second": tokens_per_second_stats,
            # "verified_success": verified_success_stats
        }

    def plot(self, data):
        plt.scatter(data)
        plt.title("Sorted Tokens per Second")
        plt.xlabel("UID (sorted)")
        plt.ylabel("Reward Score")
        plt.plotsize(100, 20)
        plt.show()
        plt.clf()
        

    async def concurrent_forward(self, uid, concurrent_forwards):
        coroutines = [
            self.forward(uid)
            for _ in range(concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    async def process_uids(self, uids, batch_size):
        for i in range(0, len(uids), batch_size):
            batch = uids[i:i+batch_size]
            await asyncio.gather(*(self.process_uid(uid) for uid in batch))

    async def process_uid(self, uid):
        bt.logging.info(f"miner uid: {uid}")
        for i in [1, 2, 4]:
            prompt, sampling_params = await generate_dataset(self)
            ground_truth = await create_ground_truth(self, prompt, sampling_params)
            await self.forward(uid, prompt, sampling_params, ground_truth)
        self.plot(self.tokens_per_second[uid])

    def run(self):
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

            bt.logging.info(f"number of uids to sample: {len(miner_uids)}")
            self.loop.run_until_complete(self.process_uids(miner_uids, batch_size=1))  # Adjust batch_size as needed

            # Sync metagraph and potentially set weights.
            self.sync()

            self.step += 1

    def __enter__(self):
        self.run()

    def __exit__(self, exc_type, exc_value, traceback):
        pass



    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given prover or verifier.
        """
        # Ensure prover or verifier hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.set_weights()


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
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
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
                self.scores[uid] = 0  # hotkey has been replaced

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

    def check_uid_availability(
        self, metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int, mock: bool = False
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
                    bt.logging.debug(f"uid: {uid} has stake ({metagraph.S[uid]}) > {vpermit_tao_limit}")
                    return False
        else:
            return True

        # Available otherwise.
        return True

    def get_miner_uids(
        self, exclude: List[int] = None
    ) -> np.ndarray:
        """Returns all available uids from the metagraph, excluding specified uids.
        Args:
            exclude (List[int]): List of uids to exclude from the result.
        Returns:
            uids (np.ndarray): Array of available uids not excluded.
        """
        available_uids = []

        for uid in range(self.metagraph.n.item()):
            if uid == self.uid:
                continue
            uid_is_available = self.check_uid_availability(
                self.metagraph, uid, self.config.neuron.vpermit_tao_limit, self.config.mock
            )
            uid_is_not_excluded = exclude is None or uid not in exclude

            if uid_is_available and uid_is_not_excluded:
                available_uids.append(uid)

        return available_uids


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # Verifier.add_args(parser)
    # args = parser.parse_args()
    with Verifier() as verifier:
        while True:
            bt.logging.info("Verifier running...", time.time())
            time.sleep(5)