import sys
import copy
import time
import asyncio
import argparse
import numpy as np
import bittensor as bt

from targon import config, check_config, add_args, add_verifier_args

class Validator:
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


        ## SET STEPS
        self.step = 0


    async def forward(self):
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
            start_time = time.time()
            bt.logging.info(f"forward block: {self.block if not self.config.mock else self.block_number} step: {self.step}")

            # --- Perform coin flip

            # --- Generate the query.
            event = await inference_data(self)
            

            if not self.config.mock:
                try:
                    block = self.substrate.subscribe_block_headers(self.subscription_handler)
                except:
                    sleep_time = 12 - (time.time() - start_time)
                    if sleep_time > 0:
                        bt.logging.info(f"Sleeping for {sleep_time} seconds")
                        await asyncio.sleep(sleep_time)
            else:
                time.sleep(1)
            
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            time.sleep(12)
            pass


    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        self.sync()
        bt.logging.info(
            f"Running verifier {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Verifier starting at block: {self.block}")

        # This loop maintains the verifier's operations until intentionally stopped.
        while True:
            bt.logging.info(f"step({self.step}) block({self.block})")

            # Run multiple forwards concurrently.
            self.loop.run_until_complete(self.concurrent_forward())

            # Check if we should exit.
            if self.should_exit:
                break

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



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # Validator.add_args(parser)
    # args = parser.parse_args()
    with Validator() as verifier:
        while True:
            bt.logging.info("Verifier running...", time.time())
            time.sleep(5)