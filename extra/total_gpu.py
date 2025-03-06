import asyncio
import uuid
from typing import List, Tuple
import os
import aiohttp
import bittensor as bt
import numpy
import json
from hashlib import sha256
from uuid import uuid4
from math import ceil
from typing import Any, Dict, List, Optional, Union
import time
from substrateinterface import Keypair
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import dotenv

dotenv.load_dotenv()


def load_public_key():
    try:
        with open("./../public_key.pem", "rb") as key_file:
            key = key_file.read()
            public_key = serialization.load_pem_public_key(key)
        return public_key
    except Exception as e:
        raise Exception(f"Error loading public key: {e}")


PUBKEY = load_public_key()


def verify_signature(msg: dict, signature: str, public_key):
    try:
        msg_bytes = json.dumps(msg, separators=(",", ":")).encode("utf-8")

        signature_bytes = base64.b64decode(signature)

        public_key.verify(
            signature_bytes,
            msg_bytes,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )

        return True
    except Exception:
        return False


async def get_gpus(session, hotkey, axon, uid) -> Tuple[int, int, int, str]:
    nonce = str(uuid.uuid4())
    req_body = {"nonce": nonce}
    req_bytes = json.dumps(
        req_body, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    headers = generate_header(hotkey, req_bytes, axon.hotkey)
    try:
        async with session.post(
            f"http://{axon.ip}:{axon.port}/nodes",
            headers=headers,
            data=req_bytes,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as res:
            if res.status != 200:
                return uid, 0, 0, f"Bad status code: {res.status}"
            nodes = await res.json()
            print(nodes)
            if not isinstance(nodes, list):
                return uid, 0, 0, f"response not list"
            h100s = 0
            h200s = 0
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                msg = node.get("msg")
                signature = node.get("signature")
                if not isinstance(msg, dict):
                    continue
                if not isinstance(signature, str):
                    continue
                if not verify_signature(msg, signature, PUBKEY):
                    continue
                miner_nonce = msg.get("nonce")
                if miner_nonce != nonce:
                    continue
                gpu_info = msg.get("gpu_info", [])
                if len(gpu_info) == 0:
                    continue
                is_h100 = "h100" in gpu_info[0].get("gpu_type", "").lower()
                is_h200 = "h200" in gpu_info[0].get("gpu_type", "").lower()
                if not is_h100 and not is_h200:
                    continue
                num_gpus = msg.get("no_of_gpus", 0)
                if is_h100:
                    h100s += 1 * num_gpus
                    continue
                h200s += 1 * num_gpus

            return uid, h100s, h200s, ""
    except Exception as e:
        return uid, 0, 0, f"Unknown error: {e}"


async def get_total_gpus():
    metagraph = subtensor.metagraph(netuid=4)

    # Get the corresponding uids
    uids_with_highest_incentives: List[int] = metagraph.uids.tolist()

    # get the axon of the uids
    axons: List[Tuple[bt.AxonInfo, int]] = [
        (metagraph.axons[uid], uid) for uid in uids_with_highest_incentives
    ]
    gpus = {
        "h100": 0,
        "h200": 0,
    }

    tasks = []
    async with aiohttp.ClientSession() as session:
        for axon, uid in axons:
            tasks.append(get_gpus(session, hotkey, axon, uid))
        responses = await asyncio.gather(*tasks)
    for uid, h100, h200, err in responses:
        gpus["h100"] += h100
        gpus["h200"] += h200
        if err == "":
            print(f"{uid}: {h100} h1, {h200} h2")
    print(f"total: {gpus}")


def generate_header(
    hotkey: Keypair,
    body: Union[Dict[Any, Any], List[Any], bytes],
    signed_for: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = round(time.time() * 1000)
    timestampInterval = ceil(timestamp / 1e4) * 1e4
    uuid = str(uuid4())
    req_hash = None
    if isinstance(body, bytes):
        req_hash = sha256(body).hexdigest()
    else:
        req_hash = sha256(json.dumps(body).encode("utf-8")).hexdigest()

    headers = {
        "Epistula-Version": str(2),
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x"
        + hotkey.sign(f"{req_hash}.{uuid}.{timestamp}.{signed_for or ''}").hex(),
    }
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        headers["Epistula-Secret-Signature-0"] = (
            "0x" + hotkey.sign(str(timestampInterval - 1) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-1"] = (
            "0x" + hotkey.sign(str(timestampInterval) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-2"] = (
            "0x" + hotkey.sign(str(timestampInterval + 1) + "." + signed_for).hex()
        )
    return headers


if __name__ == "__main__":
    hotkey = Keypair(
        ss58_address=os.getenv("HOTKEY", ""),
        public_key=os.getenv("PUBLIC_KEY", ""),
        private_key=os.getenv("PRIVATE_KEY", ""),
    )
    subtensor = bt.subtensor(
        os.getenv("SUBTENSOR_WS_ADDR", "ws://subtensor.sybil.com:9944")
    )
    asyncio.run(get_total_gpus())
