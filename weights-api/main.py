import os
import sys
import math
import traceback
from typing import List, Union

import uvicorn
from pydantic import BaseModel, field_validator, model_validator
import bittensor as bt
import bittensor_wallet
from fastapi import FastAPI, HTTPException
import logging

# Maximum number of UIDs allowed in a single request
MAX_UIDS = 256

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


def validate_weight_value(value: Union[int, float]) -> Union[int, float]:
    """Validate that a weight value is finite and non-negative."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f"Weight value must be finite, got {value}")
    if value < 0:
        raise ValueError(f"Weight value must be non-negative, got {value}")
    return value


class WeightRequest(BaseModel):
    uids: List[int]
    weights: List[Union[int, float]]
    version: int

    @field_validator("uids")
    @classmethod
    def validate_uids(cls, v: List[int]) -> List[int]:
        if len(v) == 0:
            raise ValueError("uids list must not be empty")
        if len(v) > MAX_UIDS:
            raise ValueError(f"uids list exceeds maximum length of {MAX_UIDS}")
        if len(v) != len(set(v)):
            raise ValueError("uids list contains duplicate values")
        for uid in v:
            if uid < 0:
                raise ValueError(f"uid must be non-negative, got {uid}")
        return v

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: List[Union[int, float]]) -> List[Union[int, float]]:
        if len(v) == 0:
            raise ValueError("weights list must not be empty")
        for w in v:
            validate_weight_value(w)
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        if v < 0:
            raise ValueError("version must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_lengths_match(self) -> "WeightRequest":
        if len(self.uids) != len(self.weights):
            raise ValueError(
                f"uids and weights must have the same length, "
                f"got {len(self.uids)} uids and {len(self.weights)} weights"
            )
        return self


subtensor = bt.AsyncSubtensor(CHAIN_ENDPOINT)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "netuid": NETUID}


@app.post("/api/v1/set-weights")
async def post_set_weights(req: WeightRequest):
    try:
        logger.info(
            f"setting weights for {len(req.uids)} uids, "
            f"version: {req.version}, netuid: {NETUID}"
        )
        async with subtensor as sub:
            res = await sub.set_weights(
                wallet=wallet,
                uids=req.uids,
                weights=req.weights,
                netuid=NETUID,
                version_key=req.version,
                wait_for_finalization=True,
                wait_for_inclusion=True,
                period=64,
                max_retries=1,
                raise_error=True,
            )
        logger.info(f"weights set: {res}")
        return {"success": res[0], "msg": res[1]}
    except Exception as e:
        logger.error(f"Error: {str(e)}: {traceback.format_exc()}")
        # Only way to actually reconnect subtensor easily is to blow it up
        os._exit(1)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
