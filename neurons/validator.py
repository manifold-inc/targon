import sys
import copy
import time
import random
import asyncio
from targon.utils import print_info
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
    create_ground_truth,
    handle_inference,
    add_args,
    add_validator_args,
    validate_config_and_neuron_path,
    __spec_version__ as spec_version,
)


def normalize(arr: List[float], t_min=0, t_max=1) -> List[float]:
    """
    Normalizes a list of floats to a specified range [t_min, t_max].

    This function scales the input list of floats such that the minimum value in the list
    is mapped to t_min and the maximum value in the list is mapped to t_max. The values
    in between are scaled proportionally.

    Args:
    arr (List[float]): The list of floats to be normalized.
    t_min (float): The minimum value of the target range. Default is 0.
    t_max (float): The maximum value of the target range. Default is 1.

    Returns:
    List[float]: A new list containing the normalized values.
    """
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def safe_mean(data):
    """
    Computes the mean of a list of numbers, returning 0.0 if the list is empty or if the
    computed mean is NaN or infinite.

    This function ensures that the mean calculation is safe by handling edge cases where
    the input list is empty or the mean is not a finite number.

    Args:
    data (list): A list of numbers to compute the mean of.

    Returns:
    float: The mean of the list if it's a valid number, otherwise 0.0.
    """
    if len(data) == 0:
        return 0.0
    mean_value = np.mean(data)
    if np.isnan(mean_value) or np.isinf(mean_value):
        return 0.0
    return float(mean_value)


class Validator:
    neuron_type = "ValidatorNeuron"
    config: "bt.config"

    @property
    def block(self):
        return self.subtensor.block

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
        self.axon = bt.axon(
            wallet=self.wallet,
            port=self.config.axon.port,
            external_ip=self.config.axon.external_ip,
            config=self.config,
        )

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

    async def forward(self, uids, messages, sampling_params, ground_truth):
        """
        Performs the forward pass of the validator, which includes the following steps:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores

        Args:
        uids (list): A list of user IDs to query.
        messages (list): A list of messages to send in the queries.
        sampling_params (dict): Parameters for sampling during inference.
        ground_truth (list): Ground truth data for validating responses.

        Returns:
        None
        """
        assert self.config.neuron
        try:
            bt.logging.info(
                f"Forward Block: {self.block} |  Blocks till Set Weights: { abs((self.block - self.metagraph.last_update[self.uid]) - self.config.neuron.epoch_length) }"
            )
            tasks = []
            for uid in uids:
                tasks.append(
                    asyncio.create_task(
                        handle_inference(
                            self, messages, sampling_params, uid, ground_truth
                        )
                    )
                )
            stats = await asyncio.gather(*tasks)

            for stat in stats:
                self.score(stat)

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            time.sleep(12)

    def score(self, stats):
        """
        Updates various statistics based on the provided stats object. This includes:
        - Updating the top unverified tokens per second
        - Appending times to lists for first token and all tokens
        - Appending tokens per second
        - Appending verification success
        - Updating the top verified tokens per second

        Args:
        stats (object): An object containing various statistics to update.

        Returns:
        None
        """
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
        """
        Computes and returns various statistics for the miners, including:
        - Mean times to the first token and for all tokens
        - Mean tokens per second
        - Mean verification success rates
        - Top verified and unverified tokens per second
        - Top 20 UIDs based on mean tokens per second

        The statistics are calculated using the most recent 30 entries for each miner.

        Args:
        None

        Returns:
        dict: A dictionary containing the computed statistics.
        """
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
        """
        Processes a list of user IDs by performing a forward pass of the validator.

        This function attempts to perform the forward pass with the given parameters.
        If an exception occurs during the process, it logs the error.

        Args:
        uids (list): A list of user IDs to be processed.
        messages (list): A list of messages to be sent in the queries.
        sampling_params (dict): Parameters for sampling during inference.
        ground_truth (list): Ground truth data for validating responses.

        Returns:
        None
        """
        try:
            await self.forward(uids, messages, sampling_params, ground_truth)
        except Exception as e:
            bt.logging.error(f"Error processing uids: {e}")

    def run(self):
        """
        Runs the validator, performing the following steps:
        - Asserts necessary configurations are set.
        - Synchronizes the initial state.
        - Logs the validator's startup information.
        - Enters a loop to maintain operations until intentionally stopped.
        - Logs validator information every few blocks.
        - Retrieves and shuffles miner UIDs.
        - Reduces the list of miner UIDs to a sample size.
        - Generates messages and sampling parameters.
        - Creates ground truth data.
        - Processes the UIDs asynchronously.
        - Synchronizes the metagraph and updates weights.
        - Increments the step counter.

        The loop continues until the `should_exit` flag is set to True.
        """
        assert self.config.subtensor
        assert self.config.neuron
        self.sync()
        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

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
                ground_truth = asyncio.run(
                    create_ground_truth(self, messages, sampling_params)
                )
            except Exception as e:
                bt.logging.error(f"Error generating dataset: {e}")
                time.sleep(12)
                continue
            self.loop.run_until_complete(
                self.process_uids(miner_uids, messages, sampling_params, ground_truth)
            )  # Adjust batch_size as needed

            # Sync metagraph and potentially set weights.
            self.sync()

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.set_weights()

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores received from the miners.
        The weights determine the trust and incentive levels the validator assigns to miner nodes on the network.

        This function calculates the weights by:
        - Computing the mean tokens per second for each miner.
        - Identifying the top tokens per second and setting a threshold for valid responses.
        - Normalizing the rewards based on the performance of the miners.
        - Processing the raw weights according to subtensor limitations.
        - Converting the weights and UIDs to the appropriate format.
        - Setting the weights on the chain via the subtensor connection.

        Args:
        None

        Returns:
        None
        """
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

        bt.logging.info("Setting weights")
        bt.logging.info("Processed Weights: " + str(raw_weights))
        bt.logging.info("Processed Weight Uids: " + str(uids))

        # Set the weights on chain via our subtensor connection.
        result, message = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uids,
            weights=raw_weights,
            wait_for_finalization=True,
            wait_for_inclusion=True,
            version_key=spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error(f"set_weights failed {message}")

    def check_registered(self):
        """
        Checks if the wallet's hotkey is registered on the specified subnet (netuid).
        If the hotkey is not registered, an error message is logged, and the program exits.

        This method is used to ensure that the hotkey required for operations is properly registered
        before proceeding with any further actions.

        Args:
        None

        Returns:
        None
        """
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
        Determines if the metagraph should be synchronized based on the number of epoch blocks
        that have elapsed since the last checkpoint.

        This method checks whether enough blocks have passed to warrant a sync and updates
        the last synced block accordingly.

        Args:
        None

        Returns:
        bool: True if the metagraph should be synchronized, False otherwise.
        """
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

    def should_set_weights(self) -> bool:
        """
        Determines whether the validator should set the weights based on the current step,
        block number, and the epoch length defined in the configuration.

        This function ensures that weights are not set during initialization and that the
        appropriate number of blocks have passed since the last weight update before
        setting new weights.

        Args:
        None

        Returns:
        bool: True if weights should be set, False otherwise.
        """
        assert self.config.neuron

        # Define appropriate logic for when set weights.
        return (
            (self.block - self.metagraph.last_update[self.uid])
            > self.config.neuron.epoch_length
            and self.neuron_type != "MinerNeuron"
            and self.last_posted_weights + 20 < self.subtensor.block
        )

    def resync_metagraph(self):
        """
        Resynchronizes the metagraph by copying its state, syncing it with the subtensor,
        and updating hotkeys and moving averages if there are changes.

        This function performs the following steps:
        - Logs the start of the resynchronization process.
        - Copies the current state of the metagraph.
        - Syncs the metagraph with the subtensor.
        - Checks if there are any changes in the axon information of the metagraph.
        - If there are changes, it zeroes out all hotkeys that have been replaced and updates
        the hotkeys and moving averages accordingly.
        - Ensures that new hotkeys and moving averages are added if the metagraph size has changed.

        Args:
        None

        Returns:
        None
        """
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

    def check_uid_availability(
        self,
        uid: int,
        vpermit_tao_limit: int,
        mock: bool = False,
    ) -> bool:
        """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
        Args:
            metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
            uid (int): uid to be checked
            vpermit_tao_limit (int): Validator permit tao limit
        Returns:
            bool: True if uid is available, False otherwise
        """
        if not mock:
            # Filter non serving axons.
            if not self.metagraph.axons[uid].is_serving:
                bt.logging.debug(f"uid: {uid} is not serving")
                return False
            # Filter validator permit > 1024 stake.
            if self.metagraph.validator_permit[uid]:
                bt.logging.debug(f"uid: {uid} has validator permit")
                if self.metagraph.S[uid] > vpermit_tao_limit:
                    bt.logging.debug(
                        f"uid: {uid} has stake ({self.metagraph.S[uid]}) > {vpermit_tao_limit}"
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

        for uid in range(int(self.metagraph.n.item())):
            if uid == self.uid:
                continue
            uid_is_available = self.check_uid_availability(
                uid,
                self.config.neuron.vpermit_tao_limit,
                self.config.mock if self.config.mock else False,
            )
            if uid_is_available:
                available_uids.append(uid)
        return available_uids


if __name__ == "__main__":
    validator = Validator()
    validator.run()
    exit()
