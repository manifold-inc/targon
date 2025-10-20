import os
import sys
from typing import List
from bittensor.core.axon import traceback, uvicorn
from pydantic import BaseModel
import bittensor as bt
import bittensor_wallet
from fastapi import FastAPI, HTTPException
import logging

# Initialize global subtensor connection

app = FastAPI(
    title="weights-api",
    description="API for setting weights on bittensor",
    version="1.0.0",
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CHAIN_ENDPOINT = os.getenv(
    "CHAIN_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443"
)
NETUID = int(os.getenv("NETUID", "4"))
HOTKEY_PHRASE = os.getenv("HOTKEY_PHRASE", "")
HOTKEY_PRIVATE_KEY = os.getenv("HOTKEY_PRIVATE_KEY", "")

if len(HOTKEY_PHRASE) == 0 and len(HOTKEY_PRIVATE_KEY) == 0:
    logger.error("No hotkey phrase or private key provided")
    sys.exit(1)


wallet = bittensor_wallet.Wallet()
if len(HOTKEY_PRIVATE_KEY) > 0:
    keypair = bt.Keypair.create_from_private_key(HOTKEY_PRIVATE_KEY)
    wallet.set_hotkey(keypair, overwrite=True)
else:
    wallet.regenerate_hotkey(HOTKEY_PHRASE, suppress=True, overwrite=True)

logger.info(
    f"Starting api with hotkey: {wallet.hotkey.ss58_address} on netuid: {NETUID}"
)


class WeightRequest(BaseModel):
    uids: List[int]
    weights: List[int]
    version: int


@app.post("/api/v1/set-weights")
async def post_set_weights(req: WeightRequest):
    try:
        # Subtensor is absurdly buggy and will silently fail after a few weight sets
        subtensor = bt.async_subtensor(CHAIN_ENDPOINT)
        logger.info(
            f"setting weights\n{req.uids}\n{req.weights}\nversion: {req.version}\nNetuid: {NETUID}"
        )
        res = await subtensor.set_weights(
            wallet=wallet,
            uids=req.uids,
            weights=req.weights,
            netuid=NETUID,
            version_key=req.version,
            wait_for_finalization=True,
            wait_for_inclusion=True,
            period=64
        )
        logger.info(f"weights set: {res}")
        await subtensor.close()
        return {"success": res[0], "msg": res[1]}
    except Exception as e:
        logger.error(f"Error: {str(e)}: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
