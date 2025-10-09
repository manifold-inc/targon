import os
import sys
from typing import List
from bittensor.core.axon import uvicorn
from pydantic import BaseModel
import bittensor as bt
import bittensor_wallet
from fastapi import FastAPI, HTTPException
import logging

# Initialize global subtensor connection
subtensor = bt.async_subtensor("ws://subtensor.sybil.com:9944")

app = FastAPI(
    title="weights-api",
    description="API for setting weights on bittensor",
    version="1.0.0",
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NETUID = int(os.getenv("NETUID", "4"))
HOTKEY_PHRASE = os.getenv("HOTKEY_PHRASE", "")
if len(HOTKEY_PHRASE) == 0:
    logger.error("No hotkey phrase")
    sys.exit(1)

wallet = bittensor_wallet.Wallet().regenerate_hotkey(
    HOTKEY_PHRASE, suppress=True, overwrite=True
)
logger.info(f"Starting api with hotkey: {wallet.hotkey.ss58_address} on netuid: {NETUID}")


class WeightRequest(BaseModel):
    uids: List[int]
    weights: List[int]
    version: int


@app.post("/api/v1/set-weights")
async def post_set_weights(req: WeightRequest):
    try:
        logger.info("setting weights")
        res = await subtensor.set_weights(
            wallet=wallet,
            uids=req.uids,
            weights=req.weights,
            netuid=NETUID,
            version_key=req.version,
        )
        logger.info(f"weights set: {res}")
        return {"success": res[0], "msg": res[1]}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
