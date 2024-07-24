import copy
from functools import partial
import traceback
import time
import sys
import asyncio
from typing import Tuple
from starlette.types import Send
import json
from targon.utils import print_info
import uvicorn
import argparse
import bittensor as bt
import signal
from openai import OpenAI

from bittensor.axon import FastAPIThreadedServer
from targon import (
    add_args,
    add_miner_args,
    validate_config_and_neuron_path,
)
from targon.protocol import Inference


class Miner:
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
        add_miner_args(parser)
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
        self.loop = asyncio.get_event_loop()
        self.axon = bt.axon(
            wallet=self.wallet,
            port=self.config.axon.port,
            external_ip=self.config.axon.external_ip,
            config=self.config,
        )
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

        bt.logging.debug(f"Wallet: {self.wallet}")
        bt.logging.debug(f"Subtensor: {self.subtensor}")
        bt.logging.debug(f"Metagraph: {self.metagraph}")

        ## CHECK IF REGG'D
        self.check_registered()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        ## SET MISC PARAMS
        self.step = 0
        self.should_exit = False
        self.last_synced_block = None

        ## Open AI init
        self.client = OpenAI(
            base_url=self.config.neuron.model_endpoint,
            api_key=self.config.neuron.api_key,
        )
        bt.logging.info("\N{grinning face with smiling eyes}", "Successfully Initialized!")

    async def blacklist(self, synapse: Inference) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (PromptingSynapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        assert synapse.dendrite
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: Inference) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (PromptingSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        assert synapse.dendrite
        assert synapse.dendrite.hotkey
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", str(priority)
        )
        return priority

    async def forward(self, synapse: Inference):
        """
        Handles the forwarding of an inference request by initiating a streaming chat completion
        using the specified model and sampling parameters. The response is streamed back to the client
        as it is generated.

        Args:
        synapse (Inference): The inference request containing the messages and sampling parameters.

        Returns:
        A streaming response with the generated chat completion.
        """
        bt.logging.info(u'\u2713' ,"Getting Inference request!")

        async def _prompt(synapse: Inference, send: Send) -> None:
            assert self.config.neuron
            assert synapse.sampling_params
            messages = json.loads(synapse.messages)
            stream = self.client.chat.completions.create(
                model=self.config.neuron.model_name,
                messages=messages,
                stream=True,
                temperature=synapse.sampling_params.temperature,
                top_p=synapse.sampling_params.top_p,
                seed=synapse.sampling_params.seed,
                timeout=5,
            )
            full_text = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_text += token
                    await send(
                        {
                            "type": "http.response.body",
                            "body": token.encode("utf-8"),
                            "more_body": True,
                        }
                    )
            bt.logging.info("\N{grinning face}" , "Successful Prompt");

        token_streamer = partial(_prompt, synapse)
        return synapse.create_streaming_response(token_streamer)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

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

        assert self.config.netuid
        assert self.config.subtensor
        assert self.config.axon
        assert self.config.neuron
        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the miner's axon, making it active on the network.

        # change the config in the axon
        fast_config = uvicorn.Config(
            self.axon.app,
            host="0.0.0.0",
            port=self.config.axon.port,
            log_level='info',
            loop="asyncio",
        )
        self.axon.fast_server = FastAPIThreadedServer(config=fast_config)

        self.axon.start()

        bt.logging.info(f"Miner starting at block: {self.subtensor.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                # Print Logs for Miner
                bt.logging.info(
                    print_info(
                        self.metagraph,
                        self.wallet.hotkey.ss58_address,
                        self.subtensor.block,
                    )
                )
                # Wait before checking again.
                time.sleep(12)

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1
        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception:
            bt.logging.error(traceback.format_exc())

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            bt.logging.info("Resyncing Metagraph")
            self.metagraph.sync(subtensor=self.subtensor)

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
        if self.step == 0:
            self.last_synced_block = self.subtensor.block
            return True

        if self.subtensor.block % 90 != 0:
            return False
        if self.last_synced_block == self.subtensor.block:
            return False
        self.last_synced_block = self.subtensor.block
        return True


if __name__ == "__main__":
    miner = Miner()
    miner.run()
    exit()
