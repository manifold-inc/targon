import argparse
from threading import Thread
from typing import Callable, List
import bittensor as bt
import copy

from nest_asyncio import asyncio
from substrateinterface import SubstrateInterface
from targon import (
    add_args,
    add_validator_args,
    validate_config_and_neuron_path,
)
from targon.config import add_miner_args, load_config_file
from enum import Enum
import signal

from targon import (
    __spec_version__ as spec_version,
)
from targon.metagraph import run_block_callback_thread
from targon.utils import ExitContext
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
import inspect


class NeuronType(Enum):
    Validator = "VALIDATOR"
    Miner = "MINER"


class BaseNeuron:
    config: "bt.config"
    neuron_type: NeuronType
    exit_context = ExitContext()
    next_sync_block = None
    block_callbacks: List[Callable] = []
    substrate_thread: Thread

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

    def maybe_sync_metagraph(self, block):
        assert self.config.neuron
        if block % self.config.epoch_length:
            return False

        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()
        bt.logging.info("Resyncing Metagraph")
        self.metagraph.sync(subtensor=self.subtensor)
        return True

    async def run_callbacks(self, block):
        for callback in self.block_callbacks:
            try:
                res = callback(block)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                bt.logging.error(
                    f"Failed running callback {callback.__name__}: {str(e)}"
                )

    def __init__(self, config=None):
        self.config_file = load_config_file(self.neuron_type)
        print(f"Bittensor Version: {bt.__version__}")
        print(self.config_file)
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
        signal.signal(signal.SIGINT, self.exit_context.startExit)
        signal.signal(signal.SIGTERM, self.exit_context.startExit)

        ## Typesafety
        assert self.config.logging
        assert self.config.neuron
        assert self.config.netuid
        assert self.config.axon
        assert self.config.subtensor

        ## LOGGING
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.set_info()
        if self.config.logging.debug:
            bt.logging.set_debug(True)
        if self.config.logging.trace:
            bt.logging.set_trace(True)

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

        ## Substrate, Subtensor and Metagraph
        self.substrate = SubstrateInterface(
            ss58_format=SS58_FORMAT,
            use_remote_preset=True,
            url=self.config.subtensor.chain_endpoint,
            type_registry=TYPE_REGISTRY,
        )
        self.block_callbacks.append(self.maybe_sync_metagraph)
        self.substrate_thread = run_block_callback_thread(
            self.substrate, self.run_callbacks
        )
