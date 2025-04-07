import asyncio
from typing import Dict, List, Tuple
import uuid
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
    cvm_nodes: Dict[int, List[str]],
):
    axon_info = metagraph.axons[uid]
    try:
        url = f"http://{axon_info.ip}:{axon_info.port}/cvm"
        async with session.get(url) as response:
            if response.status != 200:
                bt.logging.error(
                    f"Failed to get cvm nodes from miner {uid}: HTTP {response.status}"
                )
                cvm_nodes[uid] = []
                return

            nodes = await response.json()
            healthy_nodes = []
            node_tasks = []
            for node_url in nodes:
                node_tasks.append(get_node_health(node_url, uid, session))

            if len(node_tasks) != 0:
                responses = await asyncio.gather(*node_tasks)
                healthy_nodes = [i for i in responses if i is not None]

            if len(healthy_nodes):
                cvm_nodes[uid] = healthy_nodes
                return

    except Exception as e:
        bt.logging.error(f"Error checking miner {uid} cvm nodes: {str(e)}")
    cvm_nodes[uid] = []
    return


async def get_node_health(node_url: str, uid: int, session: aiohttp.ClientSession):
    try:
        health_response = await session.get(f"{node_url}/health")
        if health_response.status == 200:
            return node_url
        else:
            bt.logging.error(f"Health check failed for node {node_url} of miner {uid}")
    except Exception as e:
        bt.logging.error(
            f"Error checking health for node {node_url} of miner {uid}: {str(e)}"
        )
    return None


async def cvm_attest(node_url: str, uid: int, session: aiohttp.ClientSession):
    try:
        # Generate and store nonce
        nonce = str(uuid.uuid4())
        attest_response = await session.post(
            f"{node_url}/api/v1/attest",
            json={"nonce": nonce},
            headers={"Content-Type": "application/json"},
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
