# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2024 Manifold Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import torch
import argparse
import asyncio
import threading
import traceback
import bittensor as bt
from targon.base.neuron import BaseNeuron
from targon.utils.config import add_prover_args


class BaseProverNeuron(BaseNeuron):
    """
    Base class for Bittensor provers.
    """
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)  
        add_prover_args(cls, parser)    


    def __init__(self, config=None):
        super().__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_verifier_permit:
            bt.logging.warning(
                "You are allowing non-verifiers to send requests to your prover. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your prover. This is a security risk."
            )

        # The axon handles request processing, allowing verifiers to send this prover requests.
        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        # Attach deterprovers which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to prover axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        bt.logging.info(f"Axon created: {self.axon}")      

        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if str(e).startswith('There is no current event loop in thread'):
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            else:
                raise e

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def run(self):
        """
        Initiates and manages the main loop for the prover on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the prover's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The prover continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the prover waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the prover remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the prover.
            - The prover's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the prover is stopped by a manual interruption.
            Exception: For unforeseen errors during the prover's operation, which are logged for diagnosis.
        """

        # Check that prover is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving prover axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the prover's axon, making it active on the network.
        a = self.axon.start()
        asyncio.run(a)

        bt.logging.info(f"Prover starting at block: {self.block}")

        # This loop maintains the prover's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while (
                    self.block - self.metagraph.last_update[self.uid]
                    < self.config.neuron.epoch_length
                ):
                    # Wait before checking again.
                    time.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1

        # If someone intentionally stops the prover, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Prover killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the prover will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        """
        Starts the prover's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting prover in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the prover's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping prover in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the prover's operations in a background thread upon entering the context.
        This method facilitates the use of the prover in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the prover's background operations upon exiting the context.
        This method facilitates the use of the prover in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def set_weights(self):
        """
        Self-assigns a weight of 1 to the current prover (identified by its UID) and
        a weight of 0 to all other peers in the network. The weights determine the trust level the prover assigns to other nodes on the network.

        Raises:
            Exception: If there's an error while setting weights, the exception is logged for diagnosis.
        """
        try:
            # --- query the chain for the most current number of peers on the network
            chain_weights = torch.zeros(
                self.subtensor.subnetwork_n(netuid=self.metagraph.netuid)
            )
            chain_weights[self.uid] = 1

            # --- Set weights.
            self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.metagraph.netuid,
                uids=torch.arange(0, len(chain_weights)),
                weights=chain_weights.to("cpu"),
                wait_for_inclusion=False,
                version_key=self.spec_version,
            )

        except Exception as e:
            bt.logging.error(
                f"Failed to set weights on chain with exception: { e }"
            )

        bt.logging.info(f"Set weights: {chain_weights}")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)