from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import traceback
from nanoid import generate

from targon.epistula import generate_header
from targon.request import check_tokens
from targon.types import Endpoints, InferenceStats, OrganicStats
import bittensor as bt

JUGO_URL = "https://jugo.targon.com"


async def send_organics_to_jugo(
    wallet: "bt.wallet",
    organics: List[OrganicStats],
):
    try:
        body = {"organics": [organic.model_dump() for organic in organics]}
        headers = generate_header(wallet.hotkey, body)
        # Send request to the FastAPI server
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{JUGO_URL}/organics/scores",
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(60),
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


async def send_stats_to_jugo(
    metagraph: "bt.metagraph",
    subtensor: "bt.subtensor",
    wallet: "bt.wallet",
    stats: List[Tuple[int, Optional[InferenceStats]]],
    req: Dict[str, Any],
    endpoint: Endpoints,
    version: int,
    models: List[str],
    miner_tps: Dict[int, Dict[str, List[Optional[float]]]],
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
        body = {
            "request": request,
            "responses": responses,
            "models": models,
            "scores": miner_tps,
        }
        headers = generate_header(wallet.hotkey, body)
        # Send request to the FastAPI server
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{JUGO_URL}/",
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(60),
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
    try:
        async with aiohttp.ClientSession() as session:
            body = list(ports.keys())
            headers = generate_header(wallet.hotkey, body)
            async with session.post(
                JUGO_URL + "/organics",
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(60),
            ) as res:
                if res.status != 200:
                    bt.logging.info(f"Error pinging jugo {res.text}")
                    return last_bucket_id, None, None
                res_body = await res.json()
        bucket_id = res_body.get("bucket_id")
        organics = res_body.get("organics")
        if last_bucket_id == bucket_id:
            bt.logging.info(f"Already seen this bucket id")
            return last_bucket_id, None, None
        scores = {}
        organic_stats = []
        bt.logging.info(f"Found {len(organics)} organics")
        for model, records in organics.items():
            for record in records:
                uid = record["uid"]
                if scores.get(uid) is None:
                    scores[uid] = []
                if not record["success"]:
                    scores[uid].append(-500)
                    continue
                # No response tokens
                if len(record["response"]) < 2:
                    scores[uid].append(-300)
                    continue

                port = ports.get(model, {}).get("port")
                url = ports.get(model, {}).get("url")
                if not port:
                    continue

                res = await check_tokens(
                    record["request"],
                    record["response"],
                    record["uid"],
                    Endpoints(record["endpoint"]),
                    port,
                    url=url,
                )
                bt.logging.info(str(res))
                if res is None:
                    continue
                verified = res.get("verified", False)
                tps = 0
                if verified:
                    try:
                        response_tokens_count = int(record.get("response_tokens", 0))

                        # This shouldnt happen
                        if response_tokens_count == 0:
                            continue

                        tps = min(
                            response_tokens_count, record["request"]["max_tokens"]
                        ) / (int(record.get("total_time")) / 1000)
                        scores[uid].append((tps * 100) + 500)
                    except Exception as e:
                        bt.logging.error("Error scoring record: " + str(e))
                        continue
                organic_stats.append(
                    OrganicStats(
                        time_to_first_token=int(record.get("time_to_first_token")),
                        time_for_all_tokens=int(record.get("total_time"))
                        - int(record.get("time_to_first_token")),
                        total_time=int(record.get("total_time")),
                        tps=tps,
                        tokens=[],
                        verified=verified,
                        error=res.get("error"),
                        cause=res.get("cause"),
                        model=model,
                        max_tokens=record.get("request").get("max_tokens"),
                        seed=record.get("request").get("seed"),
                        temperature=record.get("request").get("temperature"),
                        uid=uid,
                        hotkey=record.get("hotkey"),
                        coldkey=record.get("coldkey"),
                        endpoint=record.get("endpoint"),
                        total_tokens=record.get("response_tokens"),
                        pub_id=record.get("pub_id", ""),
                    )
                )
        bt.logging.info(f"{bucket_id}: {scores}")
        return bucket_id, scores, organic_stats
    except Exception as e:
        bt.logging.error(str(e))
        return None
