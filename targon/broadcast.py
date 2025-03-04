from typing import List
import json
import aiohttp
import uuid

from targon.epistula import generate_header
from targon.math import verify_signature


async def broadcast(
    uid,
    models,
    axon_info,
    public_key,
    session: aiohttp.ClientSession,
    hotkey,
) -> tuple[int, List[str], List[str], int, str]:
    try:
        req_bytes = json.dumps(
            models, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        headers = generate_header(hotkey, req_bytes, axon_info.hotkey)
        headers["Content-Type"] = "application/json"
        gpu_ids = set()
        async with session.post(
            f"http://{axon_info.ip}:{axon_info.port}/models",
            headers=headers,
            data=req_bytes,
            timeout=aiohttp.ClientTimeout(total=3),
        ) as res:
            if res.status != 200:
                return uid, [], [], False, f"Models response not 200: {res.status}"
            data = await res.json()
            if not isinstance(data, list):
                return uid, [], [], False, "Model Data not list"
            miner_models = list(set(data))
        nonce = str(uuid.uuid4())
        req_body = {"nonce": nonce}
        req_bytes = json.dumps(
            req_body, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        headers = generate_header(hotkey, req_bytes, axon_info.hotkey)
        headers["Content-Type"] = "application/json"
        async with session.post(
            f"http://{axon_info.ip}:{axon_info.port}/nodes",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
            data=req_bytes,
        ) as res:
            if res.status != 200:
                return uid, [], [], False, f"Response not 200: {res.status}"
            data = await res.json()
            if not isinstance(data, list):
                return uid, [], [], False, "Gpu data not list"
            parsed_any_gpus = False
            for node in data:
                if not isinstance(node, dict):
                    continue
                msg = node.get("msg")
                signature = node.get("signature")
                if not isinstance(msg, dict):
                    continue
                if not isinstance(signature, str):
                    continue
                miner_nonce = msg.get("nonce")
                if miner_nonce != nonce:
                    continue
                if not verify_signature(msg, signature, public_key):
                    continue

                # Make sure gpus are unique
                gpu_info = msg.get("gpu_info", [])
                for gpu in gpu_info:
                    if not isinstance(gpu, dict):
                        continue
                    gpu_id = gpu.get("id", None)
                    if gpu_id is None:
                        continue
                    if gpu_id in gpu_ids:
                        continue
                    gpu_ids.add(gpu_id)
                    parsed_any_gpus = True
            return uid, miner_models, list(gpu_ids), parsed_any_gpus, ""

    except Exception as e:
        return uid, [], [], False, f"Unknown error: {e}"
