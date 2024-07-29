from math import floor
import numpy as np
from typing import List
from typing import List
from pydantic import BaseModel
import asyncpg

def print_info(metagraph, hotkey, block, isMiner=True):
    uid = metagraph.hotkeys.index(hotkey)
    log = f"UID:{uid} | Block:{block} | Consensus:{metagraph.C[uid]} | "
    if isMiner:
        return (
            log
            + f"Stake:{metagraph.S[uid]} | Trust:{metagraph.T[uid]} | Incentive:{metagraph.I[uid]} | Emission:{metagraph.E[uid]}"
        )
    return log + f"VTrust:{metagraph.Tv[uid]} | "

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
        jaro_dist += .25 * prefix * (1 - jaro_dist)
    return jaro_dist

def normalize(arr: List[float], t_min=0, t_max=1) -> List[float]:
    """
    Normalizes a list of floats to a specified range [t_min, t_max].

    This function scales the input list of floats such that the minimum value in the list
    is mapped to t_min and the maximum value in the list is mapped to t_max. The values
    in between are scaled proportionally.

    Args:
    arr (List[float]): The list of floats to be normalized.
    t_min (float): The minimum value of the target range. Default is 0.
    t_max (float): The maximum value of the target range. Default is 1.

    Returns:
    List[float]: A new list containing the normalized values.
    """
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def safe_mean(data):
    """
    Computes the mean of a list of numbers, returning 0.0 if the list is empty or if the
    computed mean is NaN or infinite.

    This function ensures that the mean calculation is safe by handling edge cases where
    the input list is empty or the mean is not a finite number.

    Args:
    data (list): A list of numbers to compute the mean of.

    Returns:
    float: The mean of the list if it's a valid number, otherwise 0.0.
    """
    if len(data) == 0:
        return 0.0
    mean_value = np.mean(data)
    if np.isnan(mean_value) or np.isinf(mean_value):
        return 0.0
    return float(mean_value)

class InferenceStats(BaseModel):
    time_to_first_token: float
    time_for_all_tokens: float
    total_time: float
    tokens_per_second: float
    tokens: List[str]
    response: str
    verified: bool

def check_tokens(miner_output, ground_truth_output):
    if len(miner_output) < (len(ground_truth_output) * 0.8):
        return False

    # Calculate the score from 0 to 1
    score = jaro_winkler(ground_truth_output, miner_output)

    if score < 0.97:
        return False

    return True

async def setup_db(database_url):
    conn = None
    try:
        conn = await asyncpg.connect(database_url)
        await create_table(conn)
    except Exception as e:
        print(f"Error creating Supabase client: {e}")
    finally:
        if conn:
            await conn.close()

async def create_table(conn):
    query = """
    CREATE TABLE IF NOT EXISTS miners_responses (
        id SERIAL PRIMARY KEY,
        uid INTEGER,
        stats JSONB,
        version VARCHAR(10)
    );
    """
    try:
        await conn.execute(query)
        print("Table created successfully.")
    except Exception as e:
        print(f"Error executing table creation query: {e}")

async def add_records(records, database_url):
    conn = None
    try:
        conn = await asyncpg.connect(database_url)
        await conn.executemany('''
            INSERT INTO miners_responses (uid, stats, version) VALUES ($1, $2, $3)
        ''', records)
        print("Records inserted successfully.")
    except Exception as e:
        print(f"Error inserting records: {e}")
    finally:
        if conn:
            await conn.close()
