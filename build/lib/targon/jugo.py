from typing import Dict, List
import json

import aiohttp

from targon.epistula import generate_header
import bittensor as bt

JUGO_URL = "https://jugo.targon.com"


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
