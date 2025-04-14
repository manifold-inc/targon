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
from targon.broadcast import cvm_attest, cvm_healthcheck
from targon.config import (
    AUTO_UPDATE,
    HEARTBEAT,
    IS_TESTNET,
)
from targon.jugo import (
    send_uid_info_to_jugo,
    score_cvm_attestations,
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

from typing import Dict, List, Tuple
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
    cvm_nodes: Dict[int, Tuple[str, List[str]]]
    lock_waiting = False
    lock_halt = False
    is_runing = False
    heartbeat_thread: Thread
    step = 0
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

        ## SET MISC PARAMS
        bt.logging.info(f"Last updated at block {self.metagraph.last_update[self.uid]}")

        ## Initialize CVM nodes tracking
        self.cvm_nodes = {}

        ## Initialize cvm_attestations, might do a similar load as organics
        self.cvm_attestations = {}

        ## REGISTER BLOCK CALLBACKS
        self.block_callbacks.extend(
            [
                self.log_on_block,
                self.set_weights_on_interval,
                self.check_cvm_nodes_health,
                self.verify_cvm_nodes,
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

    async def check_cvm_nodes_health(self, block):
        assert self.config.vpermit_tao_limit

        # skip if this is a verification block (every 60)
        if block % 60 == 0:
            return

        # check every 15 blocks
        if block % 30 != 0:
            return

        if block != 0 and not self.is_runing:
            return

        miner_uids = get_miner_uids(
            self.metagraph, self.uid, self.config.vpermit_tao_limit
        )
        self.cvm_nodes = {}
        bt.logging.info("Checking miner cvm nodes health")
        tasks = []
        res = []
        async with aiohttp.ClientSession() as session:
            for uid in miner_uids:
                tasks.append(
                    cvm_healthcheck(self.metagraph, uid, session, self.wallet.hotkey)
                )

            if len(tasks) != 0:
                res = await asyncio.gather(*tasks)

        # Count total healthy nodes across all miners
        for h, u, nodes in res:
            self.cvm_nodes[u] = (h, nodes)

        total_healthy_nodes = sum(len(nodes) for nodes in self.cvm_nodes.values())
        bt.logging.info(
            f"Verified health of {total_healthy_nodes} cvm nodes across {len(self.cvm_nodes)} miners"
        )

    async def verify_cvm_nodes(self, block):
        assert self.config.vpermit_tao_limit
        if block % 60 != 0:
            return

        if block != 0 and not self.is_runing:
            return

        bt.logging.info(f"Verifying {len(self.cvm_nodes)} cvm nodes")

        tasks = []
        res = []
        async with aiohttp.ClientSession() as session:
            for uid, (h, nodes) in self.cvm_nodes.items():
                for node_url in nodes:
                    tasks.append(
                        cvm_attest(node_url, uid, session, h, self.wallet.hotkey)
                    )

            if len(tasks) != 0:
                res = await asyncio.gather(*tasks)

        for r in res:
            if r is None:
                continue
            uid, node_url, result = r
            if uid not in self.cvm_attestations:
                self.cvm_attestations[uid] = {}
            if node_url not in self.cvm_attestations[uid]:
                self.cvm_attestations[uid][node_url] = []

            self.cvm_attestations[uid][node_url].append(result)

        attestation_stats = await score_cvm_attestations(
            self.cvm_attestations,
        )

        if attestation_stats is None:
            return

        bt.logging.info("Sending attestations to jugo")
        async with aiohttp.ClientSession() as jugo_session:
            await send_uid_info_to_jugo(
                self.wallet.hotkey, jugo_session, attestation_stats
            )
        bt.logging.info("Sent attestations to jugo")

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
        assert self.config.subtensor

        self.subtensor = bt.subtensor(
            config=self.config,
            network=self.config.subtensor.chain_endpoint,
        )
        uids, weights, jugo_data = get_weights(self.cvm_attestations)
        try:
            if not self.config_file.skip_weight_set:
                async with aiohttp.ClientSession() as session:
                    await send_uid_info_to_jugo(self.wallet.hotkey, session, jugo_data)
                self.set_weights(
                    self.wallet, self.metagraph, self.subtensor, (uids, weights)
                )
        except Exception as e:
            bt.logging.error(
                f"Failed sending uid info to jugo, or failed setting weights: {e}"
            )

        self.lock_halt = False
        # clear up attestations after scoring
        self.cvm_attestations = {}

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
        uids, raw_weights, _ = get_weights(self.cvm_attestations)
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


if __name__ == "__main__":
    try:
        validator = Validator()
        asyncio.run(validator.run())
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
