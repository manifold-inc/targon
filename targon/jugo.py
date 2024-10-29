from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import traceback
from nanoid import generate

from targon.epistula import generate_header
from targon.request import check_tokens
from targon.types import Endpoints, InferenceStats
import bittensor as bt

JUGO_URL = "https://jugo.sybil.com"


async def send_stats_to_jugo(
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
                    bt.logging.info("Records sent successfully.")
                else:
                    error_detail = await response.text()
                    bt.logging.error(
                        f"Error sending records: {response.status} - {error_detail}"
                    )

    except aiohttp.ClientConnectionError:
        bt.logging.error("Error conecting to jugo, offline.")
    except Exception as e:
        bt.logging.error(f"Error in send_stats_to_jugo: {e}")
        bt.logging.error(traceback.format_exc())


async def score_organics(last_bucket_id, ports, wallet):
    async with aiohttp.ClientSession() as session:
        headers = generate_header(wallet.hotkey, bytes())
        async with session.get(JUGO_URL, headers=headers) as res:
            if res.status != 200:
                return last_bucket_id
            organics = await res.json()
    bucket_id = organics.get("bucket_id")
    if last_bucket_id == bucket_id:
        return last_bucket_id
    records = organics.get("records")
    scores = {}
    for record in records:
        uid = record["uid"]
        if scores.get(uid) is None:
            scores[uid] = []
        if not record["success"]:
            scores[uid].append(-500)
            continue
        tokens = []
        for token in record["response"]:
            choice = token.get("choices", [{}])[0]
            text = ""
            logprob = -100
            match record["endpoint"]:
                case "CHAT":
                    text = choice.get("delta", {}).get("content")
                    logprobs = choice.get("logprobs", {})
                    logprob = logprobs.get("content", [{}])[0].get("logprob", -100)
                    token = logprobs.get("content", [{}])[0].get("token", "")
                case "COMPLETION":
                    text = choice.get("text")
                    logprobs = choice.get("logprobs", {})
                    logprob = logprobs.get("token_logprobs", -100)[0]
                    token = logprobs.get("tokens", [""])[0]

            token_id = -1
            if token.startswith("token_id:"):
                token_parts = token.split(":")
                if len(token_parts) > 1:
                    token_id = int(token_parts[1])

            tokens.append(
                {
                    "text": text,
                    "logprob": logprob,
                    "token_id": token_id,
                }
            )
            port = ports.get(record["request"]["model"], {}).get("port")
            if not port:
                continue
            res = await check_tokens(
                record["request"],
                tokens,
                record["uid"],
                Endpoints(record["endpoint"]),
                port,
            )
            if res is None:
                continue
            verified = res.get("verified")
            if verified:
                scores[uid].append(100)
            else:
                scores[uid].append(-100)

    return bucket_id, scores
