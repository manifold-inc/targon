from math import exp
import base64
import bittensor as bt
import numpy as np
from typing import Dict, List, Tuple

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


def normalize(arr: List[float], t_min=0, t_max=1) -> List[float]:
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def normalize_ignore_sub_zero(arr: List[float], t_min=0, t_max=1) -> List[float]:
    norm_arr = []
    diff = t_max - t_min
    min_non_zero = min([a for a in arr if a > 0]) * 0.9
    diff_arr = max(arr) - min_non_zero
    for i in arr:
        if i == 0:
            i = min_non_zero * 1.05
        if i == -1:
            norm_arr.append(0)
            continue
        temp = (((i - min_non_zero) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


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


@fail_with_none("Failed getting Weights")
def get_weights(
    miner_models: Dict[int, Dict[str, int]],
    organics: Dict[str, Dict[str, list[int]]],
    metadata: Dict,
) -> Tuple[List[int], List[float], List[Dict]]:
    # Mean and sigmoid of tps scores from each model. Since all miners are queried with
    # All models, more models served = higher score. *then* it becomes a speed game.

    scores = {}
    total_organics = metadata["total_attempted"]
    if total_organics == 0:
        raise Exception("No organics to score")

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
                "final_weight_before_expo": 0,
                "final_weight_after_expo_before_normal": 0,
            },
        }
        data = data_for_jugo[uid]
        scores[uid] = 0
        if (organic := organics.get(str(uid))) is None:
            continue
        safe_mean_scores = {}
        exploited = False
        for model, orgs in organic.items():
            # Only score when there are actually enough scored requests

            if not len(orgs):
                continue

            # put back when throughput higher
            score, fail_rate = safe_mean_score(orgs)

            # Exploiting a model; no points for that model
            if len(orgs) > 10 and fail_rate >= 0.15:
                exploited = True
                continue

            # More models you do, more sum you get.
            # Baseline is avg of context your serving * gpu count of that model
            safe_mean_scores[model] = score
            scores[uid] += score

        data["data"]["safe_mean_scores"] = safe_mean_scores
        data["data"]["is_exploiting"] = exploited

        if exploited or not scores[uid]:
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
            scores[uid] * (miner_completed / total_organics) * miner_success_rate
        )
        data["data"][
            "formula"
        ] = f"sum_safe_mean_score[uid]={pre_formula} * ({miner_completed=}/{total_organics=}) * {miner_success_rate=} = {scores[uid]}"
        data["data"]["final_weight_before_expo"] = scores[uid]

    tps_list = list(scores.values())
    if len(tps_list) == 0:
        bt.logging.warning("Not setting weights, no responses from miners")
        return [], [], []
    uids: List[int] = sorted(scores.keys())
    rewards = [scores[uid] for uid in uids]

    bt.logging.info(f"All scores: {json.dumps(scores)}")
    if sum(rewards) < 1 / 1e9:
        bt.logging.warning("No one gave responses worth scoring")
        return [], [], []

    raw_weights = [(r**4) for r in rewards]
    raw_weights = [max(x - (max(raw_weights) / 2), 0) for x in raw_weights]

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
