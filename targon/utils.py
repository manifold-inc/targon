from math import exp, floor
from asyncpg.connection import traceback
import bittensor as bt
import numpy as np
from typing import List, Tuple
from typing import List
from pydantic import BaseModel
import asyncpg


def print_info(metagraph, hotkey, block, isMiner=True):
    uid = metagraph.hotkeys.index(hotkey)
    log = f"UID:{uid} | Block:{block} | Consensus:{metagraph.C[uid]} | "
    if isMiner:
        bt.logging.info(
            log
            + f"Stake:{metagraph.S[uid]} | Trust:{metagraph.T[uid]} | Incentive:{metagraph.I[uid]} | Emission:{metagraph.E[uid]}"
        )
        return
    bt.logging.info(log + f"VTrust:{metagraph.Tv[uid]} | ")


def jaro_distance(s1, s2):
    if s1 == s2:
        return 1.0

    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0
    max_dist = floor(max(len(s1), len(s2)) * 0.75) - 1
    match = 0
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)

    for i in range(len1):
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
            if s1[i] == s2[j] and hash_s2[j] == 0:
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break
    if match == 0:
        return 0.0
    t = 0
    point = 0
    for i in range(len1):
        if hash_s1[i]:
            while hash_s2[point] == 0:
                point += 1
            if s1[i] != s2[point]:
                point += 1
                t += 1
            else:
                point += 1
        t /= 2
    return (match / len1 + match / len2 + (match - t) / match) / 3.0


# https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/
# This is inversed, so 1 == very similar and 0 is non-similar
def jaro_winkler(s1, s2):
    jaro_dist = jaro_distance(s1, s2)
    if jaro_dist > 0.7:
        prefix = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        prefix = min(4, prefix)
        jaro_dist += 0.25 * prefix * (1 - jaro_dist)
    return jaro_dist


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


class InferenceStats(BaseModel):
    time_to_first_token: float
    time_for_all_tokens: float
    total_time: float
    wps: float
    tokens: List[str]
    response: str
    verified: bool
    jaro_score: float


def check_tokens(miner_output, ground_truth_output) -> Tuple[float, bool]:
    # Calculate the score from 0 to 1
    half_of_ground = floor(len(ground_truth_output) / 3)
    score_1 = jaro_winkler(
        ground_truth_output[:half_of_ground], miner_output[:half_of_ground]
    )
    score_2 = jaro_winkler(
        ground_truth_output[half_of_ground:], miner_output[half_of_ground:]
    )
    score = (score_1 + score_2) / 2

    if len(miner_output) < (len(ground_truth_output) * 0.7):
        return score, False
    return score, score > 0.9 and score_2 > 0.55
