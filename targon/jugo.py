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
        bt.logging.info("Sending to Jugo...")
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


async def score_organics(last_bucket_id, ports, wallet, existing_scores):
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
                    return last_bucket_id, None
                res_body = await res.json()
        bucket_id = res_body.get("bucket_id")
        organics = res_body.get("organics")
        if last_bucket_id == bucket_id:
            bt.logging.info(f"Already seen this bucket id")
            return last_bucket_id, None
        scores = existing_scores
        organic_stats = []
        bt.logging.info(f"Found {len(organics)} organics")
        for model, records in organics.items():
            for record in records:
                pub_id = record.get("pub_id", "")
                uid = str(record["uid"])
                if scores.get(uid) is None:
                    scores[uid] = {}
                if scores[uid].get(model) is None:
                    scores[uid][model] = []
                if record.get("response") is None:
                    continue
                if not record.get("success") or len(record.get("response", [])) < 2:
                    continue

                port = ports.get(model, {}).get("port")
                url = ports.get(model, {}).get("url")
                if not port:
                    continue

                res, err = await check_tokens(
                    record["request"],
                    record["response"],
                    Endpoints(record["endpoint"]),
                    port,
                    url=url,
                )
                if err is not None or res is None:
                    bt.logging.info(
                        f"UID {uid} {pub_id}: Error validating organic on model {model}: {err} "
                    )
                    continue
                bt.logging.info(
                    f"UID {uid} {pub_id}: Verified organic: ({res}) model ({model}) at ({url}:{port})"
                )
                verified = res.get("verified", False)
                total_input_tokens = res.get("input_tokens", 0)
                tps = 0
                if verified:
                    try:
                        response_tokens_count = int(res.get("response_tokens", 0))

                        # This shouldnt happen
                        if response_tokens_count == 0:
                            continue

                        tps = min(
                            response_tokens_count, record["request"]["max_tokens"]
                        ) / (int(record.get("total_time")) / 1000)
                        context_modifier = 1 + min(
                            (((total_input_tokens / 2400) ** 2) / 1000), 1
                        )
                        gpu_required = res.get("gpus", 1)
                        if gpu_required >= 8:
                            # Large models get more weight, a lot more.
                            gpu_required = gpu_required * 2
                        scores[uid][model].append(gpu_required * context_modifier)
                    except Exception as e:
                        bt.logging.error(f"Error scoring record {pub_id}: {e}")
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
                        uid=int(uid),
                        hotkey=record.get("hotkey"),
                        coldkey=record.get("coldkey"),
                        endpoint=record.get("endpoint"),
                        total_tokens=record.get("response_tokens"),
                        pub_id=record.get("pub_id", ""),
                        gpus=res.get("gpus", 1),
                    )
                )
        bt.logging.info(f"{bucket_id}: {scores}")
        return bucket_id, organic_stats
    except Exception as e:
        bt.logging.error(str(e))
        return None, None
