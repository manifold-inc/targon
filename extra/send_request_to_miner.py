from time import time
import bittensor as bt
from typing import Any, List, Optional
from openai.types.chat import ChatCompletionMessageParam
from substrateinterface import Keypair
from bittensor.subtensor import Dict, Union
from httpx import Timeout
from openai import DefaultHttpxClient, OpenAI
from math import ceil
import httpx
from hashlib import sha256
from uuid import uuid4
import json


def create_header_hook(hotkey, axon_hotkey, model):
    def add_headers(request: httpx.Request):
        for key, header in generate_header(hotkey, request.read(), axon_hotkey).items():
            request.headers[key] = header
        request.headers["X-Targon-Model"] = model

    return add_headers


def generate_header(
    hotkey: Keypair,
    body: Union[Dict[Any, Any], List[Any], bytes],
    signed_for: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = round(time() * 1000)
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


def make_client(miner_uid):
    wallet = bt.wallet()  # Set your wallet config
    subtensor = bt.subtensor()
    metagraph = subtensor.metagraph(4)
    axon_info = metagraph.axons[miner_uid]
    client = OpenAI(
        base_url=f"http://{axon_info.ip}:{axon_info.port}/v1",
        api_key="sn4",
        max_retries=0,
        timeout=Timeout(12, connect=5, read=5),
        http_client=DefaultHttpxClient(
            event_hooks={
                "request": [
                    create_header_hook(
                        wallet.hotkey,
                        axon_info.hotkey,
                        "NousResearch/Meta-Llama-3.1-8B-Instruct",
                    )
                ]
            }
        ),
    )
    return client


if __name__ == "__main__":
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the definition of the x y problem ",
        },
    ]
    model = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    MINER_UID = -1
    client = make_client(MINER_UID)
    res = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        logprobs=True,
        max_tokens=200,
        temperature=1.0,
        top_p=0.9,
        stop=[],
        seed=42,
    )
    for chunk in res:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
