from math import exp
import bittensor as bt
import numpy as np
from typing import Dict, List, Optional, Tuple

from targon.utils import fail_with_none


def normalize(arr: List[float], t_min=0, t_max=1) -> List[float]:
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
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
    models: List[str],
) -> Tuple[List[int], List[float]]:
    # Mean and sigmoid of scores from each model. Since all miners are queried with
    # All models, more models served = higher score. *then* it becomes a speed game.
    scores = {}
    for uid in miner_scores:
        scores[uid] = 0
        for model in miner_models.get(uid, []):
            if model not in models:
                continue
            if miner_scores.get(uid) is None:
                continue
            if miner_scores[uid].get(model) is None:
                continue
            scores[uid] += safe_mean_score(miner_scores[uid][model][-15:])

    scores_list = list(scores.values())
    if len(scores_list) == 0:
        bt.logging.warning("Not setting weights, no responses from miners")
        return [], []
    uids: List[int] = sorted(scores_list.keys())
    rewards = [scores[uid] for uid in uids]

    bt.logging.info(f"All scores: {scores}")
    if sum(rewards) < 1 / 1e9:
        bt.logging.warning("No one gave responses worth scoring")
        return [], []
    raw_weights = normalize(rewards)
    bt.logging.info(f"Raw Weights: {raw_weights}")
    return uids, raw_weights
