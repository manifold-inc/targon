from math import exp
import base64
import bittensor as bt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from targon.utils import fail_with_none
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


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


def normalize(arr: List[float]):
    arr_sum = np.sum(arr)
    if arr_sum == 0:
        return arr
    return arr / arr_sum


def sigmoid(num):
    return 1 / (1 + exp(-((num - 0.5) / 0.1)))


# Returns mean treating none as zero, and the % of nones
def safe_mean_score(data) -> Tuple[float, float]:
    clean_data = [x for x in data if x is not None]
    if len(clean_data) == 0:
        return 0.0, 1
    mean_value = np.mean(clean_data)
    if np.isnan(mean_value) or np.isinf(mean_value):
        return 0.0, 1
    return float(mean_value) * sigmoid(len(clean_data) / len(data)), 1 - len(
        clean_data
    ) / len(data)


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
                # TODO @dhruv verify each jwt w/ nvidia
                if not attestation.get("success") or not attestation.get("validated"):
                    continue

                # Count each valid GPU
                gpus = attestation.get("gpus", [])

                for gpu in gpus:
                    claims = gpu.get("claims", {})
                    # TODO @ahmed gpu score should be different based on gpu type;
                    # Lets say 1pt for h100 and 2pt for h200. Make this a switch case so its easy to expand
                    if (
                        claims.get("attestation_success")
                        and claims.get("measres") == "success"
                    ):
                        verified_gpus_count += 1

        # calculate final score
        scores[uid] = verified_gpus_count

    return scores


@fail_with_none("Failed getting Weights")
def get_weights(
    miner_models: Dict[int, Dict[str, int]],
    organics: Dict[str, Dict[str, list[int]]],
    metadata: Dict,
    attestations: Dict[int, Dict[str, List[Dict[str, Any]]]],
) -> Tuple[List[int], List[float], List[Dict]]:
    # Mean and sigmoid of tps scores from each model. Since all miners are queried with
    # All models, more models served = higher score. *then* it becomes a speed game.

    scores = {}
    total_organics = metadata["total_attempted"]
    if total_organics == 0:
        raise Exception("No organics to score")

    attestation_scores = calculate_attestation_score(attestations)

    data_for_jugo = {}
    for uid in miner_models.keys():
        miner_success_rate = (
            metadata.get("miners", {}).get(str(uid), {}).get("success_rate", 0)
        )
        miner_completed = (
            metadata.get("miners", {}).get(str(uid), {}).get("completed", 0)
        )
        data_for_jugo[uid] = {
            "uid": uid,
            "data": {
                "tested_organics": organics.get(str(uid)),
                "miner_success_rate": miner_success_rate,
                "miner_completed": miner_completed,
                "overall_organics": total_organics,
                "attestation_score": attestation_scores.get(uid, 0),
                "final_weight_before_expo": 0,
                "final_weight_after_expo_before_normal": 0,
            },
        }
        data = data_for_jugo[uid]
        scores[uid] = 0
        if (organic := organics.get(str(uid))) is None:
            continue
        safe_mean_scores = {}
        fail_rate_modifier = 1
        for model, orgs in organic.items():
            # Only score when there are actually enough scored requests

            if not len(orgs):
                continue

            # put back when throughput higher
            score, fail_rate = safe_mean_score(orgs)

            # Exploiting a model; less points overall
            if len(orgs) > 10 and fail_rate >= 0.15:
                fail_rate_modifier = max(fail_rate_modifier - fail_rate, 0)
                continue

            # More models you do, more sum you get.
            # Baseline is avg of context your serving * gpu count of that model
            safe_mean_scores[model] = score
            scores[uid] += score

        data["data"]["safe_mean_scores"] = safe_mean_scores
        data["data"]["fail_rate_modifier"] = fail_rate_modifier**2

        if not scores[uid]:
            continue

        if miner_success_rate < 0.5:
            scores[uid] = 0
            continue
        if miner_completed < 25:
            scores[uid] = 0
            continue

        # Boost for high success rates
        if miner_success_rate >= 0.95:
            miner_success_rate = 1.05
        elif miner_success_rate >= 0.85:
            miner_success_rate = 1

        # Shift values so we have more room to play with success rate calc and completed calc
        scores[uid] = scores[uid] * 100
        pre_formula = scores[uid]
        scores[uid] = (
            scores[uid]
            * (miner_completed / total_organics)
            * miner_success_rate
            * fail_rate_modifier
        )
        data["data"][
            "formula"
        ] = f"sum_safe_mean_score[uid]={pre_formula} * ({miner_completed=}/{total_organics=}) * {miner_success_rate=} * {fail_rate_modifier=} = {scores[uid]}"
        data["data"]["final_weight_before_expo"] = scores[uid]

    tps_list = list(scores.values())
    attestation_scores_list = list(attestation_scores.values())
    if len(tps_list) == 0 and sum(attestation_scores_list) == 0:
        bt.logging.warning("Not setting weights, no responses from miners")
        return [], [], []

    # This gets post-processed again later on for final weights
    uids: List[int] = sorted(scores.keys())
    v5_scores = normalize([scores.get(uid, 0) for uid in uids])
    v5_scores = [x * 0.3 for x in v5_scores]
    v6_scores = normalize([attestation_scores.get(uid, 0) for uid in uids])
    v6_scores = [x * 0.7 for x in v5_scores]
    rewards = [v5_scores[uid] + v6_scores[uid] for uid in uids]

    bt.logging.info(f"All scores: {json.dumps(scores)}")
    if sum(rewards) < 1 / 1e9:
        bt.logging.warning("No one gave responses worth scoring")
        return [], [], []

    # We can leave expo on final set for now. Will need to tune as needed
    raw_weights = [max(r - (max(rewards) / 2), 0) for r in rewards]
    raw_weights = [(r**4) for r in rewards]

    final_weights = []
    for i, (uid, w) in enumerate(zip(uids, raw_weights)):
        data_for_jugo[uid]["data"]["final_weight_after_expo_before_normal"] = float(w)
        if rewards[i] == 0:
            data_for_jugo[uid]["data"]["final_weight_after_expo_before_normal"] = 0
            final_weights.append(0)
            continue
        final_weights.append(float(w))
    bt.logging.info(f"Raw Weights: {final_weights}")
    return uids, final_weights, list(data_for_jugo.values())
