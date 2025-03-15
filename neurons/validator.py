import json
import numpy as np
import asyncio
import sys
from threading import Thread
from time import sleep

import aiohttp
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
from bittensor.utils.weight_utils import process_weights_for_netuid
from substrateinterface import SubstrateInterface
from targon.types import NeuronType
from neurons.base import BaseNeuron
from targon.broadcast import broadcast
from targon.cache import load_organics, save_organics
from targon.config import (
    AUTO_UPDATE,
    HEARTBEAT,
    IS_TESTNET,
    get_models_from_config,
    get_models_from_endpoint,
)
from targon.docker import load_docker, load_existing_images, sync_output_checkers
from targon.jugo import (
    get_global_stats,
    score_organics,
    send_organics_to_jugo,
    send_uid_info_to_jugo,
)
from targon.math import get_weights
from targon.metagraph import (
    create_set_weights,
    get_miner_uids,
    run_block_callback_thread,
)
from targon.types import ValidatorConfig
from targon.updater import autoupdate
from targon.utils import (
    print_info,
)
import traceback
import bittensor as bt

from typing import Any, Dict, List, Tuple
from targon import (
    __version__,
    __spec_version__ as spec_version,
)
from dotenv import load_dotenv

load_dotenv()


class Validator(BaseNeuron):
    config_file: ValidatorConfig
    neuron_type = NeuronType.Validator
    miner_models: Dict[int, Dict[str, int]]
    verification_ports: Dict[str, Dict[str, Any]]
    models: List[str]
    lock_waiting = False
    lock_halt = False
    is_runing = False
    organics = {}
    last_bucket_id = None
    heartbeat_thread: Thread
    step = 0
    starting_docker = True
    tool_dataset = None
    skip_next_weightset = False

    def __init__(self, config=None, run_init=True, standalone=False):
        super().__init__(config=config, standalone=standalone)
        ## Typesafety
        self.set_weights = create_set_weights(spec_version, 4)

        ## CHECK IF REGG'D
        if not self.metagraph.validator_permit[self.uid] and not IS_TESTNET:
            bt.logging.error("Validator does not have vpermit")
            exit()
        if run_init:
            self.init()

    def init(self):
        assert self.config.netuid
        assert self.config.cache_file
        assert self.config.vpermit_tao_limit
        assert self.config.subtensor
        ## LOAD DOCKER
        self.client = load_docker()

        ## SET MISC PARAMS
        bt.logging.info(f"Last updated at block {self.metagraph.last_update[self.uid]}")

        ## LOAD MINER SCORES CACHE
        self.organics = load_organics()
        bt.logging.info(json.dumps(self.organics, indent=2))

        ## REGISTER BLOCK CALLBACKS
        self.block_callbacks.extend(
            [
                self.log_on_block,
                self.set_weights_on_interval,
                self.sync_output_checkers_on_interval,
                self.send_models_to_miners_on_interval,
                self.score_organics_on_block,
            ]
        )

        # Setup heartbeat thread
        if HEARTBEAT:
            self.heartbeat_thread = Thread(name="heartbeat", target=self.heartbeat)
            self.heartbeat_thread.start()

        ## DONE
        bt.logging.info(
            "\N{GRINNING FACE WITH SMILING EYES}", "Successfully Initialized!"
        )

    def heartbeat(self):
        bt.logging.info("Starting Heartbeat")
        last_step = self.step
        stuck_count = 0
        while True:
            while self.lock_halt:
                sleep(5)
            sleep(60)
            if last_step == self.step:
                stuck_count += 1
            if last_step != self.step:
                stuck_count = 0
            if stuck_count >= 5:
                bt.logging.error(
                    "Heartbeat detecting main process hang, attempting restart"
                )
                autoupdate(force=True)
                sys.exit(0)
            last_step = self.step
            bt.logging.info("Heartbeat")

    async def send_models_to_miners_on_interval(self, block):
        assert self.config.vpermit_tao_limit
        if block % self.config.epoch_length:
            return

        if block != 0 and not self.is_runing:
            return
        miner_uids = get_miner_uids(
            self.metagraph, self.uid, self.config.vpermit_tao_limit
        )
        self.miner_models = {}
        bt.logging.info("Broadcasting models to all miners")
        body = self.models
        post_tasks = []
        post_results = []
        async with aiohttp.ClientSession() as session:
            for uid in miner_uids:
                bt.logging.info(f"Broadcasting models {uid}")
                axon_info = self.metagraph.axons[uid]
                post_tasks.append(
                    broadcast(
                        uid,
                        body,
                        axon_info,
                        session,
                        self.wallet.hotkey,
                    )
                )
            if len(post_tasks) != 0:
                responses = await asyncio.gather(*post_tasks)
                post_results.extend(responses)
                post_tasks = []

        for uid, miner_models, err in post_results:
            if err != "":
                bt.logging.info(f"broadcast {uid}: {err}")
            self.miner_models[uid] = miner_models
        bt.logging.info(json.dumps(self.miner_models, indent=2))

    def sync_output_checkers_on_interval(self, block):
        if not self.is_runing:
            return
        if block % self.config.epoch_length:
            return
        self.lock_halt = True
        while not self.lock_waiting:
            sleep(1)
        try:
            models, extra = self.get_models()
            self.models = list(set([m["model"] for m in models] + extra))
            self.verification_ports = sync_output_checkers(
                self.client, models, self.config_file, extra
            )
        finally:
            self.lock_halt = False

    async def score_organics_on_block(self, block):
        if not self.is_runing:
            return
        blocks_till = self.config.epoch_length - (block % self.config.epoch_length)
        if blocks_till < 15:
            return
        bt.logging.info(str(self.verification_ports))
        bucket_id, organic_stats = await score_organics(
            self.last_bucket_id,
            self.verification_ports,
            self.wallet,
            self.organics,
            self.subtensor,
            self.config.epoch_length,
        )
        save_organics(self.organics)

        bt.logging.info("Scored Organics")
        if bucket_id == None:
            return
        self.last_bucket_id = bucket_id
        if organic_stats == None:
            return
        bt.logging.info("Sending organics to jugo")
        await send_organics_to_jugo(self.wallet, organic_stats)
        bt.logging.info("Sent organics to jugo")

    async def set_weights_on_interval(self, block):
        if block % self.config.epoch_length:
            return
        if self.skip_next_weightset == True:
            self.skip_next_weightset = False
            bt.logging.info("Skipping weightset due to startup config")
            return
        self.lock_halt = True
        while not self.lock_waiting and block != 0:
            sleep(1)

        self.subtensor = bt.subtensor(config=self.config)
        organic_metadata = await get_global_stats(self.wallet)
        if organic_metadata is None:
            bt.logging.error("Cannot set weights, failed getting metadata from jugo")
            return
        uids, weights, jugo_data = get_weights(
            self.miner_models, self.organics, organic_metadata
        )
        if not self.config_file.skip_weight_set:
            async with aiohttp.ClientSession() as session:
                await send_uid_info_to_jugo(self.wallet.hotkey, session, jugo_data)
            self.set_weights(
                self.wallet, self.metagraph, self.subtensor, (uids, weights)
            )

        self.lock_halt = False
        self.organics = {}

    async def log_on_block(self, block):
        blocks_till = self.config.epoch_length - (block % self.config.epoch_length)
        print_info(
            self.metagraph,
            self.wallet.hotkey.ss58_address,
            block,
        )
        bt.logging.info(
            f"Forward Block: {self.subtensor.block} | Blocks till Set Weights: {blocks_till}"
        )
        if block % 10 != 0:
            return
        organic_metadata = await get_global_stats(self.wallet)
        if organic_metadata is None:
            bt.logging.error("Cannot get weights, failed getting metadata from jugo")
            return
        uids, raw_weights, _ = get_weights(
            self.miner_models, self.organics, organic_metadata
        )
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=np.asarray(uids),
            weights=np.asarray(raw_weights),
            netuid=4,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        processed_weights = [float(x) for x in processed_weights]
        processed_weight_uids = [int(x) for x in processed_weight_uids]
        final = {}
        for uid, w in zip(processed_weight_uids, processed_weights):
            final[uid] = w
        bt.logging.info("Final Weights: " + json.dumps(final, indent=2))

    async def run(self):
        assert self.config.subtensor
        assert self.config.neuron
        assert self.config.vpermit_tao_limit
        bt.logging.info(
            f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.subtensor.block}")

        # Ensure everything is setup
        models, extra = self.get_models()
        self.models = list(set([m["model"] for m in models] + extra))
        try:
            self.lock_halt = True
            existing, self.verification_ports = load_existing_images(
                self.client, self.config_file
            )
            if not existing:
                self.verification_ports = sync_output_checkers(
                    self.client, models, self.config_file, extra
                )
        except Exception as e:
            bt.logging.error(f"Failed starting up output checkers: {e}")
        finally:
            self.lock_halt = False
        bt.logging.info(str(self.verification_ports))
        await self.send_models_to_miners_on_interval(0)

        if self.config_file and self.config_file.set_weights_on_start:
            try:
                await self.set_weights_on_interval(0)
                self.skip_next_weightset = True
            except Exception as e:
                bt.logging.error(f"Failed setting weights on startup: {str(e)}")

        self.is_runing = True
        while not self.exit_context.isExiting:
            self.step += 1
            if self.config.autoupdate and not AUTO_UPDATE:
                bt.logging.info("Checking autoupdate")
                autoupdate(branch="main")
            # Make sure our substrate thread is alive
            if not self.substrate_thread.is_alive():
                bt.logging.info("Restarting substrate interface due to killed node")
                self.substrate = SubstrateInterface(
                    ss58_format=SS58_FORMAT,
                    use_remote_preset=True,
                    url=self.config.subtensor.chain_endpoint,
                    type_registry=TYPE_REGISTRY,
                )
                self.substrate_thread = run_block_callback_thread(
                    self.substrate, self.run_callbacks
                )

            # Mutex for setting weights
            if self.lock_halt:
                bt.logging.info("Halting for organics")
                self.lock_waiting = True
                while self.lock_halt:
                    bt.logging.info("Waiting for lock to release")
                    sleep(5)
                self.lock_waiting = False
            await asyncio.sleep(1)
        # Exiting
        self.shutdown()

    def shutdown(self):
        pass

    def get_models(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        List of models and sizes of models
        Miners are scored based
        - How large is the model
        - How many models are we testing them on
        - How fast

        Ask miners what models they are running
        score based on what models valis want
        Let valis inspect what most popular models are
        - Top valis manually decide via model leasing
        - Minor valis follow along for consensus
        """
        assert self.config.models
        models_from_config = []
        if self.config_file and self.config_file.verification_ports:
            models_from_config = list(self.config_file.verification_ports.keys())
        match self.config.models.mode:
            case "config":
                models = get_models_from_config()
                if not models:
                    raise Exception("No models")
            case _:
                models = get_models_from_endpoint(self.config.models.endpoint)
                if not models:
                    raise Exception("No models")

        return models, models_from_config


if __name__ == "__main__":
    try:
        validator = Validator()
        asyncio.run(validator.run())
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
