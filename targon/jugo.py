from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import traceback
from nanoid import generate

from targon.epistula import generate_header
from targon.types import Endpoints, InferenceStats
import bittensor as bt

JUGO_URL = "https://jugo.sybil.com"


async def send_stats_to_ingestor(
    metagraph: "bt.metagraph",
    subtensor: "bt.subtensor",
    wallet: "bt.wallet",
    stats: List[Tuple[int, Optional[InferenceStats]]],
    req: Dict[str, Any],
    endpoint: Endpoints,
    version: int,
    models: List[str],
):
    try:
        r_nanoid = generate(size=48)
        responses = [
            {
                "r_nanoid": r_nanoid,
                "hotkey": metagraph.axons[uid].hotkey,
                "coldkey": metagraph.axons[uid].coldkey,
                "uid": int(uid),
                "stats": stat and stat.model_dump(),
            }
            for uid, stat in stats
        ]
        request = {
            "r_nanoid": r_nanoid,
            "block": subtensor.block,
            "request": req,
            "request_endpoint": str(endpoint),
            "version": version,
            "hotkey": wallet.hotkey.ss58_address,
        }
        # Prepare the data
        body = {"request": request, "responses": responses, "models": models}
        headers = generate_header(wallet.hotkey, body)
        # Send request to the FastAPI server
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{JUGO_URL}/", headers=headers, json=body
            ) as response:
                if response.status == 200:
                    bt.logging.info("Records ingested successfully.")
                else:
                    error_detail = await response.text()
                    bt.logging.error(
                        f"Error sending records: {response.status} - {error_detail}"
                    )

    except aiohttp.ClientConnectionError:
        bt.logging.error("Error conecting to ingestor, offline.")
    except Exception as e:
        bt.logging.error(f"Error in send_stats_to_ingestor: {e}")
        bt.logging.error(traceback.format_exc())
