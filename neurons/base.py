import argparse
import bittensor as bt
import copy

from nest_asyncio import asyncio
from openai import AsyncOpenAI
from targon import (
    add_args,
    add_validator_args,
    validate_config_and_neuron_path,
)
from targon.config import add_miner_args
from enum import Enum
import signal

from targon import (
    __spec_version__ as spec_version,
)


class NeuronType(Enum):
    Validator = 'VALIDATOR'
    Miner = "MINER"

class BaseNeuron:
    config: "bt.config"
    neuron_type: NeuronType
    should_exit = False
    next_sync_block = None

    def exit_gracefully(self, *_):
        if self.should_exit:
            bt.logging.info("Forcefully exiting")
            exit()
        bt.logging.info("Exiting Gracefully at end of cycle")
        self.should_exit = True

    def check_registered(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        if self.next_sync_block is None:
            self.next_sync_block = self.subtensor.block + 30
            return True

        if self.next_sync_block > self.subtensor.block:
            return False
        self.next_sync_block = self.subtensor.block + 30
        return True

    def sync_metagraph(self):
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()
        if self.should_sync_metagraph():
            bt.logging.info("Resyncing Metagraph")
            self.metagraph.sync(subtensor=self.subtensor)
            return True
        return False


    def __init__(self, config=None):
        # Add parser args
        bt.logging.info(f"Targon version {spec_version}")
        parser = argparse.ArgumentParser()
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.axon.add_args(parser)
        add_args(parser)
        if self.neuron_type == NeuronType.Validator:
            add_validator_args(parser)
        if self.neuron_type == NeuronType.Miner:
            add_miner_args(parser)
        self.config = bt.config(parser)
        if config:
            base_config = copy.deepcopy(config)
            self.config.merge(base_config)
        validate_config_and_neuron_path(self.config)

        ## Add kill signals
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        ## Typesafety
        assert self.config.logging
        assert self.config.neuron
        assert self.config.netuid
        assert self.config.axon

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
        bt.logging.debug(f"Wallet: {self.wallet}")
        bt.logging.debug(f"Subtensor: {self.subtensor}")
        bt.logging.debug(f"Metagraph: {self.metagraph}")

        ## CHECK IF REGG'D
        self.check_registered()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            base_url=self.config.neuron.model_endpoint,
            api_key=self.config.neuron.api_key,
        )
