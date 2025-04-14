import traceback
import time
from bittensor.core.axon import FastAPIThreadedServer
from bittensor.core.extrinsics.serving import serve_extrinsic
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
import netaddr
import requests

from targon.types import NeuronType
from neurons.base import BaseNeuron
from targon.epistula import verify_signature
from targon.types import MinerConfig
from targon.utils import print_info
import uvicorn
import bittensor as bt


class Miner(BaseNeuron):
    config_file: MinerConfig
    neuron_type = NeuronType.Miner
    fast_api: FastAPIThreadedServer

    def shutdown(self):
        if self.fast_api:
            self.fast_api.stop()

    def log_on_block(self, block):
        print_info(
            self.metagraph,
            self.wallet.hotkey.ss58_address,
            block,
        )

    def __init__(self, config=None):
        super().__init__(config)
        bt.logging.set_info()
        ## Typesafety
        assert self.config.netuid
        assert self.config.logging

        # Register log callback
        self.block_callbacks.append(self.log_on_block)

        ## BITTENSOR INITIALIZATION
        bt.logging.info(
            "\N{grinning face with smiling eyes}", "Successfully Initialized!"
        )

        assert self.config_file
        assert self.config_file.cvm_nodes

    async def list_cvm_nodes(self, _: Request):
        return self.get_cvm_nodes()

    def get_cvm_nodes(self):
        # TODO
        # Miners need to return the list of cvm nodes they are using
        assert self.config_file
        assert self.config_file.cvm_nodes
        return self.config_file.cvm_nodes

    async def determine_epistula_version_and_verify(self, request: Request):
        version = request.headers.get("Epistula-Version")
        if version == "2":
            await self.verify_request(request)
            return
        raise HTTPException(status_code=400, detail="Unknown Epistula version")

    async def verify_request(
        self,
        request: Request,
    ):
        # We do this as early as possible so that now has a lesser chance
        # of causing a stale request
        now = round(time.time() * 1000)

        # We need to check the signature of the body as bytes
        # But use some specific fields from the body
        signed_by = request.headers.get("Epistula-Signed-By")
        signed_for = request.headers.get("Epistula-Signed-For")
        if signed_for != self.wallet.hotkey.ss58_address:
            raise HTTPException(
                status_code=400, detail="Bad Request, message is not intended for self"
            )
        if signed_by not in self.metagraph.hotkeys:
            raise HTTPException(status_code=401, detail="Signer not in metagraph")

        uid = self.metagraph.hotkeys.index(signed_by)
        stake = self.metagraph.S[uid].item()
        if not self.config.no_force_validator_permit and stake < 10000:
            bt.logging.warning(
                f"Blacklisting request from {signed_by} [uid={uid}], not enough stake -- {stake}"
            )
            raise HTTPException(status_code=401, detail="Stake below minimum: {stake}")

        # If anything is returned here, we can throw
        body = await request.body()
        err = verify_signature(
            request.headers.get("Epistula-Request-Signature"),
            body,
            request.headers.get("Epistula-Timestamp"),
            request.headers.get("Epistula-Uuid"),
            signed_for,
            signed_by,
            now,
        )
        if err:
            bt.logging.error(err)
            raise HTTPException(status_code=400, detail=err)

    def run(self):
        assert self.config.netuid
        assert self.config.subtensor
        assert self.config.axon

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        external_ip = self.config.axon.external_ip or self.config.axon.ip
        if not external_ip or external_ip == "[::]":
            try:
                external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
                netaddr.IPAddress(external_ip)
            except Exception:
                bt.logging.error("Failed to get external IP")

        bt.logging.info(
            f"Serving miner endpoint {external_ip}:{self.config.axon.port} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        serve_success = serve_extrinsic(
            subtensor=self.subtensor,
            wallet=self.wallet,
            ip=external_ip,
            port=self.config.axon.port,
            protocol=4,
            netuid=self.config.netuid,
        )
        if not serve_success:
            bt.logging.error("Failed to serve endpoint")
            return

        # Start  starts the miner's endpoint, making it active on the network.
        # change the config in the axon
        app = FastAPI()
        router = APIRouter()
        router.add_api_route("/", ping, methods=["GET"])
        router.add_api_route(
            "/cvm",
            self.list_cvm_nodes,
            dependencies=[Depends(self.determine_epistula_version_and_verify)],
            methods=["GET"],
        )
        app.include_router(router)
        fast_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.config.axon.port,
            log_level="info",
            loop="asyncio",
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        bt.logging.info(f"Miner starting at block: {self.subtensor.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.exit_context.isExiting:
                time.sleep(1)
        except Exception as e:
            bt.logging.error(str(e))
            bt.logging.error(traceback.format_exc())
        self.shutdown()


def ping():
    return 200


if __name__ == "__main__":
    try:
        miner = Miner()
        miner.run()
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
