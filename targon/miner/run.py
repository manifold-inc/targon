# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2023 Manifold Labs

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
import wandb
import asyncio
import bittensor as bt
import traceback
from targon.protocol import TargonSearchResultStream
from .set_weights import set_weights


def run(self):
    """
    Initiates and manages the main loop for the miner on the Bittensor network.

    This function performs the following primary tasks:
    1. Optionally registers the miner's wallet with the network.
    2. Attaches the miner's forward, blacklist, and priority functions to its axon.
    3. Starts the miner's axon, making it active on the network.
    4. Regularly updates the metagraph with the latest network state.
    5. Optionally sets weights on the network, defining how much trust to assign to other nodes.
    6. Handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

    The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
    During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
    knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
    and up-to-date with the network's latest state.

    Note:
        - The function leverages the global configurations set during the initialization of the miner.
        - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

    Raises:
        KeyboardInterrupt: If the miner is stopped by a manual interruption.
        Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
    """

    # --- Optionally register the wallet.
    if not self.config.miner.no_register:
        bt.logging.info(
            f"Registering wallet: {self.wallet} on netuid {self.config.netuid}"
        )
        self.subtensor.register(netuid=self.config.netuid, wallet=self.wallet)

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving axon {TargonSearchResultStream} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
    )
    self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
    self.axon.start()

    # --- Run until should_exit = True.
    self.last_epoch_block = self.subtensor.get_current_block()
    bt.logging.info(f"Miner starting at block: {self.last_epoch_block}")

    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    try:
        while not self.should_exit:
            start_epoch = time.time()

            # --- Wait until next epoch.
            current_block = self.subtensor.get_current_block()
            while (
                current_block - self.last_epoch_block
                < self.config.miner.blocks_per_epoch
            ):
                # --- Wait for next bloc.
                time.sleep(1)
                current_block = self.subtensor.get_current_block()

                # --- Check if we should exit.
                if self.should_exit:
                    break

            # --- Update the metagraph with the latest network state.
            self.last_epoch_block = self.subtensor.get_current_block()

            metagraph = self.subtensor.metagraph(
                netuid=self.config.netuid,
                lite=True,
                block=self.last_epoch_block,
            )
            log = (
                f"Step:{step} | "
                f"Block:{metagraph.block.item()} | "
                f"Stake:{metagraph.S[self.my_subnet_uid]} | "
                f"Rank:{metagraph.R[self.my_subnet_uid]} | "
                f"Trust:{metagraph.T[self.my_subnet_uid]} | "
                f"Consensus:{metagraph.C[self.my_subnet_uid] } | "
                f"Incentive:{metagraph.I[self.my_subnet_uid]} | "
                f"Emission:{metagraph.E[self.my_subnet_uid]}"
            )
            bt.logging.info(log)
            if self.config.wandb.on:
                wandb.log(log)

            # --- Set weights.
            if not self.config.miner.no_set_weights:
                set_weights(
                    self.subtensor,
                    self.config.netuid,
                    self.my_subnet_uid,
                    self.wallet,
                    self.config.wandb.on,
                )
            step += 1

    # If someone intentionally stops the miner, it'll safely terminate operations.
    except KeyboardInterrupt:
        self.axon.stop()
        bt.logging.success("Miner killed by keyboard interrupt.")
        exit()

    # In case of unforeseen errors, the miner will log the error and continue operations.
    except Exception as e:
        bt.logging.error(traceback.format_exc())
