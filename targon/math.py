from math import exp
import base64
import bittensor as bt
import numpy as np
from nv_attestation_sdk import attestation
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union

from targon.utils import fail_with_none
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


# Load policy at module level
def load_policy(policy_path: str) -> Optional[str]:
    try:
        with open(policy_path, "r") as f:
            policy = json.load(f)
        bt.logging.info(f"Loaded appraisal policy from {policy_path}")
        return json.dumps(policy)
    except Exception as e:
        bt.logging.error(f"Failed to load appraisal policy: {e}")
        return None


# Load policy once at module level
POLICY_PATH = os.environ.get("APPRAISAL_POLICY", "targon/remote_policy.json")
ATTESTATION_POLICY = load_policy(POLICY_PATH)


def validate_attestation(
    token: str, expected_nonce: str, policy: Optional[str] = ATTESTATION_POLICY
) -> bool:
    try:
        if not policy:
            bt.logging.error("No valid policy loaded for attestation validation")
            return False

        NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"
        client = attestation.Attestation("Verifier")
        client.add_verifier(
            attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, ""
        )
        client.set_token("Verifier", token)
        client.set_nonce(expected_nonce)
        valid = client.validate_token(policy)
        bt.logging.info(
            "Attestation token validated successfully."
        ) if valid else bt.logging.error("Attestation token validation failed.")
        return valid
    except Exception as e:
        bt.logging.error(f"Exception during token validation: {e}")
        return False


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


def normalize(arr: Union[List[int], List[float]]):
    arr_sum = np.sum(arr)
    if arr_sum == 0:
        return arr
    return arr / arr_sum


def sigmoid(num):
    return 1 / (1 + exp(-((num - 0.5) / 0.1)))


def calculate_attestation_score(
    attestations: Optional[Dict[int, Dict[str, List[Dict[str, Any]]]]]
) -> Dict[int, float]:
    if not attestations:
        return {}
    scores = {}

    for uid, nodes in attestations.items():
        verified_gpus_count = 0

        for _, attestations_list in nodes.items():
            for attestation in attestations_list:
                # Verify top-level success indicators
                if not attestation.get("success") or not attestation.get("validated"):
                    continue

                # Verify nonce matches what we sent
                expected_nonce = attestation.get("expected_nonce")
                received_nonce = attestation.get("nonce")
                if (
                    not expected_nonce
                    or not received_nonce
                    or expected_nonce != received_nonce
                ):
                    continue

                # Validate with NVIDIA NRAS
                token = attestation.get("token")
                if not token:
                    bt.logging.error(f"No attestation token provided")
                    continue

                if not validate_attestation(token, expected_nonce):
                    bt.logging.error(f"NVIDIA attestation validation failed")
                    continue

                # Count each valid GPU
                gpus = attestation.get("gpus", [])

                for gpu in gpus:
                    claims = gpu.get("claims", {})
                    if (
                        claims.get("attestation_success")
                        and claims.get("measres") == "success"
                    ):
                        # Score based on GPU model
                        gpu_model = gpu.get("model", "unknown").upper()
                        match gpu_model:
                            case s if "H200" in s:
                                gpu_score = 2.0
                            case s if "H100" in s:
                                gpu_score = 1.0
                            # TODO support other gpus, also nuke non h100 and h200 gpus. Pretty sure you can't do this but just in case.
                            case _:
                                gpu_score = 0.1

                        verified_gpus_count += gpu_score
                        bt.logging.info(
                            f"GPU {gpu.get('id', 'unknown')} model {gpu_model} scored {gpu_score}"
                        )

        # calculate final score
        scores[uid] = verified_gpus_count

    return scores


@fail_with_none("Failed getting Weights")
def get_weights(
    attestations: Dict[int, Dict[str, List[Dict[str, Any]]]],
) -> Tuple[List[int], List[float], List[Dict]]:
    attestation_scores = calculate_attestation_score(attestations)

    data_for_jugo = {}
    attestation_scores_list = list(attestation_scores.values())
    if sum(attestation_scores_list) == 0:
        bt.logging.warning("Not setting weights, no responses from miners")
        # Burn alpha
        return [28], [1], []

    # This gets post-processed again later on for final weights
    uids = list(attestation_scores.keys())
    rewards = normalize([attestation_scores.get(uid, 0) for uid in uids])
    rewards = [float(x * 0.15) for x in rewards]
    # burn 85% for now
    rewards.append(0.85)
    uids.append(28)

    bt.logging.info(f"All scores: {attestation_scores}")
    final_weights = []

    for uid in uids:
        data_for_jugo[uid] = {
            "data": {
                "final_weight_after_expo_before_normal": 0
            }
        }

    for uid, w in zip(uids, rewards):
        data_for_jugo[uid]["data"]["final_weight_after_expo_before_normal"] = w
        final_weights.append(w)

    bt.logging.info(f"Raw Weights: {final_weights}")
    return uids, final_weights, list(data_for_jugo.values())
