from math import exp
import bittensor as bt
import numpy as np
from typing import Dict, List, Optional, Tuple

from targon.config import SLIDING_WINDOW
from targon.utils import fail_with_none


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


def safe_mean_score(data):
    clean_data = [x for x in data if x is not None]
    if len(clean_data) == 0:
        return 0.0
    mean_value = np.mean(clean_data)
    if np.isnan(mean_value) or np.isinf(mean_value):
        return 0.0
    return float(mean_value) * sigmoid(len(clean_data) / len(data))


@fail_with_none("Failed getting Weights")
def get_weights(
    miner_models: Dict[int, List[str]],
    miner_scores: Dict[int, Dict[str, List[Optional[float]]]],
    organics: Dict[str, Dict[str, list[int]]],
    models: List[str],  # Validator Models
) -> Tuple[List[int], List[float]]:
    # Mean and sigmoid of tps scores from each model. Since all miners are queried with
    # All models, more models served = higher score. *then* it becomes a speed game.
    tps = {}
    total_organics = 0
    for uid in miner_scores:
        if (organic := organics.get(str(uid))) is not None:
            total_organics += sum([len(o) for o in organic.values()])

    for uid in miner_scores:
        synth_scores = 0
        for model in miner_models.get(uid, []):
            if model not in models:
                continue
            if miner_scores.get(uid) is None:
                continue
            if miner_scores[uid].get(model) is None:
                continue

            synth_scores += safe_mean_score(miner_scores[uid][model][-SLIDING_WINDOW:])

        tps[uid] = 0
        if synth_scores == 0:
            # passed some syntehtics
            tps[uid] = -1
            continue

        if (organic := organics.get(str(uid))) is not None:
            # Boost miners for doing more organics
            self_total = 0
            for orgs in organic.values():
                self_total += len(orgs)
                tps[uid] += safe_mean_score(orgs)
            tps[uid] = (tps[uid] * ((self_total / total_organics) + 1)) * 2

    tps_list = list(tps.values())
    if len(tps_list) == 0:
        bt.logging.warning("Not setting weights, no responses from miners")
        return [], []
    uids: List[int] = sorted(tps.keys())
    rewards = [tps[uid] for uid in uids]

    bt.logging.info(f"All wps: {tps}")
    if sum(rewards) < 1 / 1e9:
        bt.logging.warning("No one gave responses worth scoring")
        return [], []
    raw_weights = normalize_ignore_sub_zero(rewards)
    raw_weights = (np.e ** (np.log(max(raw_weights)) / max(raw_weights))) ** raw_weights
    bt.logging.info(f"Raw Weights: {raw_weights}")
    return uids, raw_weights
