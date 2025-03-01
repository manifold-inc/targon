import traceback
from typing import List
import json
import aiohttp
import uuid

from targon.epistula import generate_header
from targon.math import verify_signature


async def broadcast(
    miner_nodes,
    miner_models,
    uid,
    models,
    axon_info,
    public_key,
    session: aiohttp.ClientSession,
    hotkey,
) -> tuple[int, List[str]]:
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
                print(f"{uid} failed models request: {res.status}")
                miner_models[uid] = []
                return uid, []
            data = await res.json()
            if not isinstance(data, list):
                miner_models[uid] = []
                print(f"{uid} Data not list: {data}")
                return uid, []
            miner_models[uid] = list(set(data))
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
                miner_nodes[uid] = False
                return uid, []
            data = await res.json()
            if not isinstance(data, list):
                miner_nodes[uid] = False
                return uid, []
            for node in data:
                miner_nodes[uid] = False
                if not isinstance(node, dict):
                    break
                msg = node.get("msg")
                signature = node.get("signature")
                if not isinstance(msg, dict):
                    break
                if not isinstance(signature, str):
                    break
                miner_nonce = msg.get("nonce")
                if miner_nonce != nonce:
                    break
                if not verify_signature(msg, signature, public_key):
                    break

                # Make sure gpus are unique
                gpu_info = msg.get("gpu_info", [])
                parsed_gpus = False
                for gpu in gpu_info:
                    parsed_gpus = False
                    if not isinstance(gpu, dict):
                        break
                    gpu_id = gpu.get("id", None)
                    if gpu_id is None:
                        break
                    if gpu_id in gpu_ids:
                        break
                    gpu_ids.add(gpu_id)
                    parsed_gpus = True

                if not parsed_gpus:
                    break
                miner_nodes[uid] = True
            return uid, list(gpu_ids)

    except Exception as e:
        print(f"{uid} error broadcasting {e}")
        miner_nodes[uid] = False
        miner_models[uid] = []
        return uid, []
