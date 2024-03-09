# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 philanthrope <-- Main Author
# Copyright © 2024 Manifold Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import bittensor as bt

from redis import asyncio as aioredis


REQUEST_LIMIT_CHALLENGER = 100_000 # 100k every 360 blocks
REQUEST_LIMIT_GRANDMASTER = 25_000 # 10k every 360 blocks
REQUEST_LIMIT_GOLD = 10_000 # 10k every 360 blocks
REQUEST_LIMIT_GOLD = 10_000 # 10k every 360 blocks
REQUEST_LIMIT_SILVER = 5_000 # 5k every 360 blocks
REQUEST_LIMIT_BRONZE = 500 # 1k every 360 blocks


COSINE_SIMILARITY_THRESHOLD_CHALLENGER = 0.98
COSINE_SIMILARITY_THRESHOLD_GRANDMASTER = 0.95
COSINE_SIMILARITY_THRESHOLD_GOLD = 0.90
COSINE_SIMILARITY_THRESHOLD_SILVER = 0.85
COSINE_SIMILARITY_THRESHOLD_BRONZE = 0.80

# Requirements for each tier. These must be maintained for a prover to remain in that tier.
CHALLENGER_INFERENCE_SUCCESS_RATE = 0.999  # 1/1000 chance of failure
CHALLENGER_CHALLENGE_SUCCESS_RATE = 0.999  # 1/1000 chance of failure
GRANDMASTER_INFERENCE_SUCCESS_RATE = 0.989  # 1/100 chance of failure
GRANDMASTER_CHALLENGE_SUCCESS_RATE = 0.989  # 1/100 chance of failure
GOLD_INFERENCE_SUCCESS_RATE = 0.949  # 1/50 chance of failure
GOLD_CHALLENGE_SUCCESS_RATE = 0.949  # 1/50 chance of failure
SILVER_INFERENCE_SUCCESS_RATE = 0.949  # 1/20 chance of failure
SILVER_CHALLENGE_SUCCESS_RATE = 0.949  # 1/20 chance of failure

CHALLENGER_TIER_REWARD_FACTOR = 1.0  # Get 100% rewards
GRANDMASTER_TIER_REWARD_FACTOR = 0.888  # Get 88.8% rewards
GOLD_TIER_REWARD_FACTOR = 0.777  # Get 77.7% rewards
SILVER_TIER_REWARD_FACTOR = 0.555  # Get 55.5% rewards
BRONZE_TIER_REWARD_FACTOR = 0.444  # Get 44.4% rewards

CHALLENGER_TIER_TOTAL_SUCCESSES = 4_000
GRANDMASTER_TIER_TOTAL_SUCCESSES = 2_000
GOLD_TIER_TOTAL_SUCCESSES = 500  # 50
SILVER_TIER_TOTAL_SUCCESSES = 250

TIER_CONFIG = {
    "Bronze": {
        "success_rate": 0.70, 
        "request_limit": 500, 
        "reward_factor": 0.444,
        "similarity_threshold": 0.80
    },
    "Silver": {
        "success_rate": 0.72, 
        "request_limit": 1000, 
        "reward_factor": 0.555,
        "similarity_threshold": 0.84
    },
    "Gold": {
        "success_rate": 0.74, 
        "request_limit": 5000, 
        "reward_factor": 0.666,
        "similarity_threshold": 0.88
    },
    "Platinum": {
        "success_rate": 0.76, 
        "request_limit": 7500, 
        "reward_factor": 0.777,
        "similarity_threshold": 0.92
    },
    "Diamond": {
        "success_rate": 0.78, 
        "request_limit": 10000, 
        "reward_factor": 0.888,
        "similarity_threshold": 0.93
    },
    "Emerald": {
        "success_rate": 0.80, 
        "request_limit": 12500, 
        "reward_factor": 0.900,
        "similarity_threshold": 0.94
    },
    "Ruby": {
        "success_rate": 0.82, 
        "request_limit": 15000, 
        "reward_factor": 0.920,
        "similarity_threshold": 0.95
    },
    "Jade": {
        "success_rate": 0.84, 
        "request_limit": 17500, 
        "reward_factor": 0.940,
        "similarity_threshold": 0.96
    },
    "Master": {
        "success_rate": 0.88, 
        "request_limit": 20000, 
        "reward_factor": 0.960,
        "similarity_threshold": 0.97
    },
    "Grandmaster": 
    {
        "success_rate": 0.92, 
        "request_limit": 22500, 
        "reward_factor": 0.980,
        "similarity_threshold": 0.98
    },
    "Challenger": {
        "success_rate": 0.98, 
        "request_limit": 25000, 
        "reward_factor": 1.0,
        "similarity_threshold": 0.99
    },
}



async def reset_request_stats(stats_key: str, database: aioredis.Redis):
    await database.hmset(
        stats_key,
        {
            "inference_attempts": 0,
            "inference_successes": 0,
            "challenge_successes": 0,
            "challenge_attempts": 0,
            "total_interval_successes": 0,
        },
    )


async def rollover_request_stats(database: aioredis.Redis):
    """
    Asynchronously resets the request statistics for all provers.
    This function should be called periodically to reset the statistics for all provers.
    Args:
        database (redis.Redis): The Redis client instance for database operations.
    """
    prover_stats_keys = [stats_key async for stats_key in database.scan_iter("stats:*")]
    tasks = [reset_request_stats(stats_key, database) for stats_key in prover_stats_keys]
    await asyncio.gather(*tasks)


async def prover_is_registered(ss58_address: str, database: aioredis.Redis):
    """
    Checks if a prover is registered in the database.

    Parameters:
        ss58_address (str): The key representing the hotkey.
        database (redis.Redis): The Redis client instance.

    Returns:
        True if the prover is registered, False otherwise.
    """
    return await database.exists(f"stats:{ss58_address}")


async def register_prover(ss58_address: str, database: aioredis.Redis):
    """
    Registers a new prover in the decentralized storage system, initializing their statistics.
    This function creates a new entry in the database for a prover with default values,
    setting them initially to the Bronze tier and assigning the corresponding Request Limit.
    Args:
        ss58_address (str): The unique address (hotkey) of the prover to be registered.
        database (redis.Redis): The Redis client instance for database operations.
    """
    # Initialize statistics for a new prover in a separate hash
    await database.hmset(
        f"stats:{ss58_address}",
        {
            "inference_attempts": 0,
            "inference_successes": 0,
            "challenge_successes": 0,
            "challenge_attempts": 0,
            "total_successes": 0,
            "tier": "Bronze",
            "request_limit": TIER_CONFIG["Bronze"]["request_limit"],
        },
    )


async def update_statistics(
    ss58_address: str, success: bool, task_type: str, database: aioredis.Redis
):
    """
    Updates the statistics of a prover in the decentralized storage system.
    If the prover is not already registered, they are registered first. This function updates
    the prover's statistics based on the task performed (store, challenge, retrieve) and whether
    it was successful.
    Args:
        ss58_address (str): The unique address (hotkey) of the prover.
        success (bool): Indicates whether the task was successful or not.
        task_type (str): The type of task performed ('store', 'challenge', 'retrieve').
        database (redis.Redis): The Redis client instance for database operations.
    """
    # Check and see if this prover is registered.
    if not await prover_is_registered(ss58_address, database):
        bt.logging.debug(f"Registering new prover {ss58_address}...")
        await register_prover(ss58_address, database)

    # Update statistics in the stats hash
    stats_key = f"stats:{ss58_address}"

    if task_type in ["inference", "challenge"]:
        await database.hincrby(stats_key, f"{task_type}_attempts", 1)
        if success:
            await database.hincrby(stats_key, f"{task_type}_successes", 1)

            # --- add to total_interval_successes
            await database.hincrby(stats_key, "total_interval_successes", 1)


    # Update the total successes that we rollover every epoch
    if await database.hget(stats_key, "total_successes") == None:
        inference_successes = int(await database.hget(stats_key, "inference_successes"))
        challenge_successes = int(await database.hget(stats_key, "challenge_successes"))
        total_successes = inference_successes + challenge_successes
        await database.hset(stats_key, "total_successes", total_successes)
    if success:
        await database.hincrby(stats_key, "total_successes", 1)



async def get_similarity_threshold(ss58_address: str, database: aioredis.Redis):
    """
    Retrieves the similarity threshold based on the tier of a given prover.
    This function returns the similarity threshold that a prover must maintain to remain in their tier.
    Args:
        ss58_address (str): The unique address (hotkey) of the prover.
        database (redis.Redis): The Redis client instance for database operations.
    Returns:
        float: The similarity threshold corresponding to the prover's tier.
    """
    # Default similarity threshold if the prover's tier cannot be determined
    default_similarity_threshold = 0.70

    # Make sure to await the existence check
    if not await database.exists(f"stats:{ss58_address}"):
        bt.logging.warning(f"Prover key {ss58_address} is not registered!")
        return default_similarity_threshold

    # Properly await the async call before decoding
    tier_bytes = await database.hget(f"stats:{ss58_address}", "tier")
    if tier_bytes is None:
        return default_similarity_threshold

    tier = tier_bytes.decode()

    # Assuming we add a 'similarity_threshold' to each tier's configuration in TIER_CONFIG
    similarity_threshold = TIER_CONFIG.get(tier, {}).get("similarity_threshold", default_similarity_threshold)

    return similarity_threshold

async def compute_tier(stats_key: str, database: aioredis.Redis):
    """
    Asynchronously computes and updates the tier for a prover in the decentralized storage system.
    This function should be called periodically to ensure a prover's tier is up-to-date based on
    their performance. It computes the tier based on the prover's success rate in challenges.
    Args:
        stats_key (str): The key representing the prover's statistics in the database.
        database (redis.Redis): The Redis client instance for database operations.
    """

    if not await database.exists(stats_key):
        bt.logging.warning(f"Prover key {stats_key} is not registered!")
        return

    challenge_successes = int(await database.hget(stats_key, "challenge_successes") or 0)
    challenge_attempts = int(await database.hget(stats_key, "challenge_attempts") or 0)
    challenge_success_rate = challenge_successes / challenge_attempts if challenge_attempts > 0 else 0

    current_tier_bytes = await database.hget(stats_key, "tier")
    if current_tier_bytes is None:
        bt.logging.error(f"No tier found for {stats_key}, setting to default 'Bronze'.")
        current_tier = "Bronze"
    else:
        current_tier = current_tier_bytes.decode()

    current_tier_index = list(TIER_CONFIG.keys()).index(current_tier)
    new_tier_index = current_tier_index  # Start with the assumption of no change

    # Check for promotion
    for tier_name, tier_info in list(TIER_CONFIG.items())[current_tier_index+1:]:
        if challenge_success_rate > tier_info["success_rate"]:
            new_tier_index = list(TIER_CONFIG.keys()).index(tier_name)
            break

    # Check for demotion, if no promotion is possible
    if new_tier_index == current_tier_index:
        for tier_name, tier_info in reversed(list(TIER_CONFIG.items())[:current_tier_index]):
            if challenge_success_rate <= tier_info["success_rate"]:
                new_tier_index = list(TIER_CONFIG.keys()).index(tier_name)
                break

    # Apply tier change if there is any
    if new_tier_index != current_tier_index:
        new_tier_name = list(TIER_CONFIG.keys())[new_tier_index]
        await database.hset(stats_key, "tier", new_tier_name)
        await database.hset(stats_key, "request_limit", TIER_CONFIG[new_tier_name]["request_limit"])

async def compute_all_tiers(database: aioredis.Redis):
    """
    Asynchronously computes and updates the tiers for all provers in the decentralized storage system.
    This function should be called periodically to ensure provers' tiers are up-to-date based on
    their performance. It iterates over all provers and calls `compute_tier` for each one.
    """
    provers = [prover async for prover in database.scan_iter("stats:*")]
    tasks = [compute_tier(prover, database) for prover in provers]
    await asyncio.gather(*tasks)

    bt.logging.info(f"Resetting statistics for all hotkeys...")
    await rollover_request_stats(database)
async def get_uid_tier_mapping(database: aioredis.Redis):
    """
    Retrieves a mapping of UIDs to their respective tiers.

    Args:
        database (aioredis.Redis): The Redis client instance for database operations.

    Returns:
        dict: A dictionary mapping UIDs to their tiers.
    """
    uid_tier_mapping = {}
    stats_keys = [key async for key in database.scan_iter("stats:*")]
    for key in stats_keys:
        ss58_address = key.decode().split(":")[1]
        tier = await database.hget(key, "tier")
        if tier is not None:
            uid_tier_mapping[ss58_address] = tier.decode()
    return uid_tier_mapping


async def get_remaining_requests(ss58_address: str, database: aioredis.Redis):
    """
    Calculates the remaining number of requests a prover can make in the current epoch.

    Args:
        ss58_address (str): The unique address (hotkey) of the prover.
        database (aioredis.Redis): The Redis client instance for database operations.

    Returns:
        int: The remaining number of requests for the prover.
    """
    request_limit = int(await database.hget(f"stats:{ss58_address}", "request_limit"))
    total_interval_successes = int(
        await database.hget(f"stats:{ss58_address}", "total_interval_successes")
    )
    return request_limit - total_interval_successes

async def get_tier_factor(ss58_address: str, database: aioredis.Redis):
    """
    Retrieves the reward factor based on the tier of a given prover.
    This function returns a factor that represents the proportion of rewards a prover
    is eligible to receive based on their tier.
    Args:
        ss58_address (str): The unique address (hotkey) of the prover.
        database (redis.Redis): The Redis client instance for database operations.
    Returns:
        float: The reward factor corresponding to the prover's tier.
    """
    tier = await database.hget(f"stats:{ss58_address}", "tier")
    if tier == b"Challenger":
        return CHALLENGER_TIER_REWARD_FACTOR
    elif tier == b"Grandmaster":
        return GRANDMASTER_TIER_REWARD_FACTOR
    elif tier == b"Gold":
        return GOLD_TIER_REWARD_FACTOR
    elif tier == b"Silver":
        return SILVER_TIER_REWARD_FACTOR
    else:
        return BRONZE_TIER_REWARD_FACTOR