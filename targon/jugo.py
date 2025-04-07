import asyncio
import time
from typing import Dict, List, Optional
import random
import json

import aiohttp
import traceback

from targon.epistula import generate_header
from targon.request import check_tokens
from targon.types import Endpoints, OrganicStats
import bittensor as bt

from targon.utils import fail_with_none

JUGO_URL = "https://jugo.targon.com"


async def send_organics_to_jugo(
    wallet,
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


@fail_with_none("Failed getting global stats")
async def get_global_stats(wallet):
    async with aiohttp.ClientSession() as session:
        headers = generate_header(wallet.hotkey, b"")
        async with session.get(
            JUGO_URL + "/organics/metadata",
            headers=headers,
            timeout=aiohttp.ClientTimeout(60),
        ) as res:
            if res.status != 200:
                bt.logging.info(f"Error pinging jugo {res.text}")
                return None
            res_body = await res.json()
    assert isinstance(res_body, dict)
    return res_body


async def send_uid_info_to_jugo(
    hotkey, session: aiohttp.ClientSession, data: List[Dict]
):
    req_bytes = json.dumps(
        data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    headers = generate_header(hotkey, req_bytes, "")
    async with session.post(
        url="https://jugo.targon.com/mongo", headers=headers, data=req_bytes
    ) as res:
        if res.status == 200:
            return
        text = await res.text()
    bt.logging.error(f"Failed sending to jugo: {text}")


async def score_organics(
    last_bucket_id,
    ports,
    wallet,
    existing_scores,
    subtensor,
    epoch_len,
    max_concurrent=2,
):
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

        # This takes some load off each verif so that not all
        # validators are hitting them at the same time at start of interval
        organics_list = list(organics.values())
        records = []
        for model_set in organics_list:
            records.extend(model_set)
        random.shuffle(records)

        bt.logging.info(
            f"Found {len(records)} organics. Running with {max_concurrent} concurrent"
        )

        running_tasks: List[asyncio.Task[Optional[OrganicStats]]] = []
        total_completed = 0
        start = time.time()
        for record in records:
            model = record.get("model_name")
            blocks_till = epoch_len - (subtensor.block % epoch_len)
            if blocks_till < 15:
                break

            port = ports.get(model, {}).get("port")
            url = ports.get(model, {}).get("url")
            if not url:
                continue

            if len(running_tasks) > max_concurrent:
                done, pending = await asyncio.wait(
                    running_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                running_tasks = list(pending)
                for task in list(done):
                    task_res = task.result()
                    if task_res is None:
                        continue
                    total_completed += 1
                    organic_stats.append(task_res)

            running_tasks.append(
                asyncio.create_task(
                    verify_record(
                        record,
                        scores,
                        port,
                        url,
                        api_key=ports.get(model, {}).get("api_key"),
                    )
                )
            )

        if len(running_tasks) != 0:
            done, _ = await asyncio.wait(
                running_tasks, return_when=asyncio.ALL_COMPLETED
            )
            for task in list(done):
                task_res = task.result()
                if task_res is None:
                    continue
                total_completed += 1
                organic_stats.append(task_res)

        bt.logging.info(
            f"Completed {total_completed} organics in {time.time() - start}s\n{bucket_id}: {scores}"
        )
        return bucket_id, organic_stats
    except Exception as e:
        bt.logging.error(traceback.format_exc())
        bt.logging.error(str(e))
        return None, None


async def verify_record(
    record, scores, port: Optional[int], url: str, api_key: Optional[str] = None
) -> Optional[OrganicStats]:
    model = record.get("model_name")
    pub_id = record.get("pub_id", "")
    uid = str(record["uid"])
    if scores.get(uid) is None:
        scores[uid] = {}
    if scores[uid].get(model) is None:
        scores[uid][model] = []
    if record.get("response") is None:
        return None
    if not record.get("success") or len(record.get("response", [])) < 2:
        return None

    res, err = await check_tokens(
        record["request"],
        record["response"],
        Endpoints(record["endpoint"]),
        port,
        url=url,
        request_id=record.get("pub_id"),
        api_key=api_key,
    )
    if err is not None or res is None:
        bt.logging.info(
            f"UID {uid} {pub_id}: Error validating organic on model {model}: {err} "
        )
        return None
    bt.logging.info(
        f"UID {uid} {pub_id}: Verified organic: ({res}) model ({model}) at ({url}:{port})"
    )
    verified = res.get("verified", False)
    # total_input_tokens = res.get("input_tokens", 0)
    tps = 0
    if not verified:
        scores[uid][model].append(None)
    if verified:
        try:
            response_tokens_count = int(res.get("response_tokens", 0))

            # This shouldnt happen
            if response_tokens_count == 0:
                return None

            tps = min(response_tokens_count, record["request"]["max_tokens"]) / (
                int(record.get("total_time")) / 1000
            )
            # context_modifier = (
            #    1
            #    + 0.5 * (total_input_tokens / 32000)
            #    + 0.25 * (total_input_tokens / 64000) ** 2
            # )
            gpu_required = res.get("gpus", 1)
            if gpu_required >= 8:
                # Large models get more weight, a lot more.
                gpu_required = gpu_required * 2
            # Bring back when we have more organics scored per interval
            # scores[uid][model].append(gpu_required * context_modifier)

            scores[uid][model].append(gpu_required)
        except Exception as e:
            bt.logging.error(f"Error scoring record {pub_id}: {e}")
            return None
    return OrganicStats(
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


async def score_cvm_attestations(attestations):
    attestation_stats = []

    for uid, nodes in attestations.items():
        for node_id, attestations_list in nodes.items():
            for attestation in attestations_list:
                try:
                    stats = {
                        "uid": uid,
                        "node_id": node_id,
                        "success": attestation.get("success", False),
                        "nonce": attestation.get("nonce", ""),
                        "token": attestation.get("token", ""),
                        "claims": attestation.get("claims", {}),
                        "validated": attestation.get("validated", False),
                        "gpus": attestation.get("gpus", []),
                        "error": attestation.get("error", None),
                        "input_tokens": attestation.get("input_tokens", 0),
                        "response_tokens": attestation.get("response_tokens", 0),
                    }

                    attestation_stats.append(stats)

                    if not attestation.get("success", False):
                        bt.logging.error(
                            f"Attestation failed for node {node_id} of miner {uid}: {attestation.get('error')}"
                        )
                    if not attestation.get("validated", False):
                        bt.logging.error(
                            f"Validation failed for node {node_id} of miner {uid}"
                        )

                except Exception as e:
                    bt.logging.error(
                        f"Error processing attestation for node {node_id} of miner {uid}: {e}"
                    )
                    continue

    return attestation_stats

