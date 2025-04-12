import asyncio
from typing import Dict, List, Tuple
import uuid
import secrets
import bittensor as bt
import json
import aiohttp

from targon.epistula import generate_header


async def broadcast(
    uid,
    models,
    axon_info,
    session: aiohttp.ClientSession,
    hotkey,
) -> Tuple[int, Dict[str, int], str]:
    try:
        req_bytes = json.dumps(
            models, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        async with session.post(
            f"http://{axon_info.ip}:{axon_info.port}/models",
            headers={
                "Content-Type": "application/json",
                **generate_header(hotkey, req_bytes, axon_info.hotkey),
            },
            data=req_bytes,
            timeout=aiohttp.ClientTimeout(total=3),
        ) as res:
            if res.status != 200:
                return uid, {}, f"Models response not 200: {res.status}"
            data = await res.json()
            if not isinstance(data, Dict):
                return uid, {}, "Model Data not list"
            return uid, data, ""

    except Exception as e:
        return uid, {}, f"Unknown error: {e}"


async def cvm_healthcheck(
    metagraph: "bt.metagraph",
    uid: int,
    session: aiohttp.ClientSession,
    hotkey,
) -> Tuple[str, int, List[str]]:
    axon_info = metagraph.axons[uid]
    try:
        url = f"http://{axon_info.ip}:{axon_info.port}/cvm"
        async with session.get(
            url,
            headers={
                **generate_header(hotkey, b"", axon_info.hotkey),
            },
            timeout=aiohttp.ClientTimeout(total=3),
        ) as response:
            if response.status != 200:
                bt.logging.error(
                    f"Failed to get cvm nodes from miner {uid}: HTTP {response.status}"
                )
                return axon_info.hotkey, uid, []

            nodes = await response.json()
            healthy_nodes = []
            node_tasks = []
            for node_url in nodes:
                node_tasks.append(
                    get_node_health(node_url, uid, session, hotkey, axon_info.hotkey)
                )

            if len(node_tasks) != 0:
                responses = await asyncio.gather(*node_tasks)
                healthy_nodes = [i for i in responses if i is not None]

            if len(healthy_nodes):
                return axon_info.hotkey, uid, healthy_nodes

    except Exception as e:
        bt.logging.error(f"Error checking miner {uid} cvm nodes: {str(e)}")
    return axon_info.hotkey, uid, []


async def get_node_health(
    node_url: str, uid: int, session: aiohttp.ClientSession, self_hotkey, miner_hotkey
):
    try:
        health_response = await session.get(
            f"http://{node_url}/health",
            headers={
                **generate_header(self_hotkey, b"", miner_hotkey),
            },
            timeout=aiohttp.ClientTimeout(total=3),
        )
        if health_response.status == 200:
            return node_url
        else:
            bt.logging.error(f"Health check failed for node {node_url} of miner {uid}")
    except Exception as e:
        bt.logging.error(
            f"Error checking health for node {node_url} of miner {uid}: {str(e)}"
        )
    return None


async def cvm_attest(
    node_url: str,
    uid: int,
    session: aiohttp.ClientSession,
    miner_hotkey,
    self_hotkey,
):
    try:
        # TODO: This is a temporary solution to generate a nonce that is 32 bytes long
        nonce = uuid.uuid4().hex + uuid.uuid4().hex 
        req_bytes = json.dumps(
            {"nonce": nonce}, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        attest_response = await session.post(
            f"http://{node_url}/api/v1/attest",
            data=req_bytes,
            headers={
                "Content-Type": "application/json",
                **generate_header(self_hotkey, req_bytes, miner_hotkey),
            },
        )
        if attest_response.status != 200:
            bt.logging.error(
                f"Failed to attest to node {node_url} of miner {uid}: HTTP {attest_response.status}"
            )
            return None
        result = await attest_response.json()
        # Store nonce with result
        result["expected_nonce"] = nonce

        return (uid, node_url, result)
    except Exception as e:
        bt.logging.error(f"Error verifying node {node_url} of miner {uid}: {str(e)}")
    return None
