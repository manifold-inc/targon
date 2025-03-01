import json
import uuid
import random
import asyncio
import sys
from threading import Thread
from time import sleep

import aiohttp
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
import httpx
from substrateinterface import SubstrateInterface
from neurons.base import BaseNeuron, NeuronType
from targon.broadcast import broadcast
from targon.cache import load_cache, load_organics, save_organics
from targon.config import (
    AUTO_UPDATE,
    HEARTBEAT,
    IS_TESTNET,
    SLIDING_WINDOW,
    get_models_from_config,
    get_models_from_endpoint,
    load_public_key,
)
from targon.dataset import download_dataset, download_tool_dataset
from targon.docker import load_docker, load_existing_images, sync_output_checkers
from targon.jugo import score_organics, send_organics_to_jugo, send_stats_to_jugo
from targon.math import get_weights
from targon.metagraph import (
    create_set_weights,
    get_miner_uids,
    resync_hotkeys,
    run_block_callback_thread,
)
from targon.request import check_tokens, generate_request, handle_inference
from targon.updater import autoupdate
from targon.utils import (
    fail_with_none,
    print_info,
)
from targon.types import Endpoints, InferenceStats
import traceback
import bittensor as bt

from typing import Any, Dict, List, Optional, Tuple
from targon import (
    __version__,
    __spec_version__ as spec_version,
)
from dotenv import load_dotenv

load_dotenv()


class Validator(BaseNeuron):
    neuron_type = NeuronType.Validator
    miner_tps: Dict[int, Dict[str, List[Optional[float]]]]
    miner_models: Dict[int, List[str]]
    miner_nodes: Dict[int, bool] = {}
    verification_ports: Dict[str, Dict[str, Any]]
    models: List[str]
    lock_waiting = False
    lock_halt = False
    is_runing = False
    organics = {}
    last_bucket_id = None
    heartbeat_thread: Thread
    step = 0
    dataset = None
    starting_docker = True
    tool_dataset = None
    skip_next_weightset = False

    def __init__(self, config=None, run_init=True):
        super().__init__(config)
        ## Typesafety
        self.set_weights = create_set_weights(spec_version, 4)
        self.public_key = load_public_key()

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
        assert self.config.database
        assert self.config.subtensor
        ## LOAD DOCKER
        self.client = load_docker()

        ## SET MISC PARAMS
        self.next_forward_block = None
        self.last_posted_weights = self.metagraph.last_update[self.uid]
        bt.logging.info(f"Last updated at block {self.last_posted_weights}")

        ## LOAD MINER SCORES CACHE
        miners = get_miner_uids(self.metagraph, self.uid, self.config.vpermit_tao_limit)
        self.miner_tps = load_cache(
            self.config.cache_file, self.subtensor.block, miners
        )
        self.organics = load_organics()

        ## LOAD DATASETS
        bt.logging.info("⌛️", "Loading datasets")
        self.dataset = download_dataset()
        self.tool_dataset = download_tool_dataset()

        ## REGISTER BLOCK CALLBACKS
        self.block_callbacks.extend(
            [
                self.log_on_block,
                self.set_weights_on_interval,
                self.sync_output_checkers_on_interval,
                self.resync_hotkeys_on_interval,
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
        gpu_ids = set()
        post_tasks = []
        async with aiohttp.ClientSession() as session:
            for uid in miner_uids:
                bt.logging.info(f"Broadcasting models {uid}")
                axon_info = self.metagraph.axons[uid]
                post_tasks.append(
                    broadcast(
                        self.miner_nodes,
                        self.miner_models,
                        uid,
                        body,
                        axon_info,
                        self.public_key,
                        session,
                        self.wallet.hotkey,
                    )
                )
            all_gpus = await asyncio.gather(*post_tasks)
        bt.logging.info(str(all_gpus))

        bt.logging.info("Miner models: " + str(self.miner_models))

    def resync_hotkeys_on_interval(self, block):
        if not self.is_runing:
            return
        if block % self.config.epoch_length:
            return
        resync_hotkeys(self.metagraph, self.miner_tps)

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

    def score_organics_on_block(self, block):
        if not self.is_runing:
            return
        blocks_till = self.config.epoch_length - (block % self.config.epoch_length)
        if block % 5 or blocks_till < 35:
            return
        bt.logging.info(str(self.verification_ports))
        bucket_id, organic_stats = asyncio.run(
            score_organics(
                self.last_bucket_id,
                self.verification_ports,
                self.wallet,
                self.organics,
            )
        )
        save_organics(self.organics)

        bt.logging.info("Scored Organics")
        if bucket_id == None:
            return
        self.last_bucket_id = bucket_id
        if organic_stats == None:
            return
        bt.logging.info(
            json.dumps(
                get_weights(
                    self.miner_models,
                    self.miner_tps,
                    self.organics,
                    self.models,
                    self.miner_nodes,
                )
            ),
        )
        bt.logging.info("Sending organics to jugo")
        asyncio.run(send_organics_to_jugo(self.wallet, organic_stats))
        bt.logging.info("Sent organics to jugo")

    def set_weights_on_interval(self, block):
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
        weights = get_weights(
            self.miner_models,
            self.miner_tps,
            self.organics,
            self.models,
            self.miner_nodes,
        )

        if not self.config_file.skip_weight_set:
            self.set_weights(self.wallet, self.metagraph, self.subtensor, weights)

        # Only keep last 30 scores
        for uid in self.miner_tps:
            for model in self.miner_tps[uid]:
                self.miner_tps[uid][model] = self.miner_tps[uid][model][
                    -SLIDING_WINDOW:
                ]
        self.lock_halt = False
        self.organics = {}

    def log_on_block(self, block):
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
        weights = get_weights(
            self.miner_models,
            self.miner_tps,
            self.organics,
            self.models,
            self.miner_nodes,
        )
        weight_json = json.dumps(weights)
        bt.logging.info(weight_json)

    async def run(self):
        assert self.config.subtensor
        assert self.config.neuron
        assert self.config.database
        assert self.config.vpermit_tao_limit
        bt.logging.info(
            f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.subtensor.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        miner_subset = 36

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
        resync_hotkeys(self.metagraph, self.miner_tps)
        await self.send_models_to_miners_on_interval(0)

        if self.config_file and self.config_file.set_weights_on_start:
            try:
                self.set_weights_on_interval(0)
                self.skip_next_weightset = True
            except Exception as e:
                bt.logging.error(f"Failed setting weights on startup: {str(e)}")

        self.is_runing = True
        while not self.exit_context.isExiting:
            bt.logging.info("Running next synthetic iteration")
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
                    sleep(1)
                self.lock_waiting = False

            bt.logging.info("Selecting model")
            # Random model, but every three is a model we are verifying for sure
            model_name = random.choice(self.models)
            if self.step % 3 == 0:
                model_name = random.choice(list(self.verification_ports.keys()))

            models, extra_models = self.get_models()
            models = [m["model"] for m in models]
            generator_model_name = random.choice(
                list(
                    (set(models) - set(extra_models))
                    & set(list(self.verification_ports.keys()))
                )
            )
            if self.verification_ports.get(model_name) != None:
                endpoint = random.choice(
                    self.verification_ports[model_name]["endpoints"]
                )
            else:
                endpoint = random.choice(
                    self.verification_ports[generator_model_name]["endpoints"]
                )

            uids = get_miner_uids(
                self.metagraph, self.uid, self.config.vpermit_tao_limit
            )
            random.shuffle(uids)
            miner_uids = []
            for uid in uids:
                if len(miner_uids) > miner_subset:
                    break

                # Make sure tps array exists
                if self.miner_tps[uid].get(model_name) is None:
                    self.miner_tps[uid][model_name] = []

                if model_name not in self.miner_models.get(uid, []):
                    self.miner_tps[uid][model_name].append(None)
                    continue
                miner_uids.append(uid)

            # Skip if no miners running this model
            if not len(miner_uids):
                bt.logging.info("No miners for this model")
                continue

            bt.logging.info(
                f"Querying Miners for model {model_name} using {generator_model_name}"
            )

            res = await self.query_miners(
                miner_uids,
                model_name,
                endpoint,
                generator_model_name,
                model_name in list(self.verification_ports.keys()),
            )
            self.save_scores()
            if res is not None:
                bt.logging.info("About to end stats to jugo")
                await send_stats_to_jugo(
                    self.metagraph,
                    self.subtensor,
                    self.wallet,
                    *res,
                    spec_version,
                    self.models,
                    self.miner_tps,
                )

        # Exiting
        self.shutdown()

    async def verify_response(self, uid, request, endpoint, stat: InferenceStats):
        if stat.error or stat.cause:
            return uid, stat
        # We do this out of the handle_inference loop to not block other requests
        verification_port = self.verification_ports.get(request["model"], {}).get(
            "port"
        )
        verification_url = self.verification_ports.get(request["model"], {}).get("url")
        if verification_port is None or verification_url is None:
            bt.logging.error(
                "Send request to a miner without verification port for model"
            )
            return uid, None
        verified, err = await check_tokens(
            request,
            stat.tokens,
            endpoint=endpoint,
            port=verification_port,
            url=verification_url,
        )
        if err is not None or verified is None:
            bt.logging.error(
                f"Failed checking tokens for {uid} on model {request['model']}: {err}"
            )
            return uid, None
        stat.verified = (
            verified.get("verified", False) if verified is not None else False
        )
        if stat.verified:
            tokencount = min(len(stat.tokens), request["max_tokens"])
            response_tokens = int(verified.get("response_tokens", 0))
            if response_tokens:
                tokencount = min(tokencount, response_tokens)
            stat.tps = tokencount / stat.total_time
            stat.gpus = verified.get("gpus", 1)
        if stat.error is None and not stat.verified:
            stat.error = verified.get("error")
            stat.cause = verified.get("cause")
        return uid, stat

    async def query_miners(
        self,
        miner_uids: List[int],
        model_name: str,
        endpoint: Endpoints,
        generator_model_name: str,
        should_score: bool,
    ):
        assert self.config.database

        request = generate_request(
            self.dataset,
            self.tool_dataset,
            model_name,
            endpoint,
            self.verification_ports.get(generator_model_name),
        )
        bt.logging.info("Generated Request")
        if not request:
            bt.logging.info("No request was generated")
            return None

        bt.logging.info(f"{model_name} - {endpoint}: {request}")

        # We do these in separate groups for better response timings
        tasks = []
        try:
            for uid in miner_uids:
                tasks.append(
                    asyncio.create_task(
                        handle_inference(
                            self.metagraph, self.wallet, request, uid, endpoint
                        )
                    )
                )
            responses: List[Tuple[int, InferenceStats]] = await asyncio.gather(*tasks)

            # Skip scoring if we arent running that model
            if not should_score:
                return None

            tasks = []
            for uid, stat in responses:
                tasks.append(
                    asyncio.create_task(
                        self.verify_response(uid, request, endpoint, stat)
                    )
                )
            stats: List[Tuple[int, Optional[InferenceStats]]] = await asyncio.gather(
                *tasks
            )
        except Exception:
            bt.logging.error(f"Failed sending requests: {traceback.format_exc()}")
            stats = []
        processed_stats = []
        for uid, stat in stats:
            if not stat:
                continue
            processed_stats.append((uid, stat))
            bt.logging.info(f"{uid}: {stat.verified} | {stat.total_time}")
            if not stat.verified and stat.error:
                bt.logging.info(str(stat.cause))

            # UID is not in our miner tps list
            if self.miner_tps.get(uid) is None:
                self.miner_tps[uid] = {request["model"]: []}
            # This uid doesnt have reccords of this model
            if self.miner_tps[uid].get(request["model"]) is None:
                self.miner_tps[uid][request["model"]] = []

            if stat.verified and stat.total_time != 0:
                self.miner_tps[uid][request["model"]].append(stat.gpus)
                continue
            self.miner_tps[uid][request["model"]].append(None)
        return (processed_stats, request, endpoint)

    @fail_with_none("Failed writing to cache file")
    def save_scores(self):
        assert self.config.cache_file
        if self.exit_context.isExiting:
            return
        try:
            with open(self.config.cache_file, "w") as file:
                bt.logging.info("Caching scores...")
                json.dump(
                    {
                        "miner_tps": self.miner_tps,
                        "block_saved": self.subtensor.block,
                        "version": spec_version,
                    },
                    file,
                )
                file.flush()
                bt.logging.info("Cached")
        except Exception as e:
            bt.logging.error(f"Failed writing to cache file: {e}")

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
