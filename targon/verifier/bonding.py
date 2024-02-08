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
import aioredis
import bittensor as bt


REQUEST_LIMIT_CHALLENGER = 100_000 # 100k every 360 blocks
REQUEST_LIMIT_GRANDMASTER = 25_000 # 10k every 360 blocks
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

CHALLENGER_TIER_TOTAL_SUCCESSES = 100_000  # 100,000
GRANDMASTER_TIER_TOTAL_SUCCESSES = 50_000  # 50,000
GOLD_TIER_TOTAL_SUCCESSES = 5_000  # 5,000
SILVER_TIER_TOTAL_SUCCESSES = 1_000  # 1,000


async def reset_request_stats(stats_key: str, database: aioredis.Redis):
    """
    Asynchronously resets the request statistics for a prover.

    This function should be called periodically to reset the statistics for a prover while keeping the tier and total_successes.

    Args:
        ss58_address (str): The unique address (hotkey) of the prover.
        database (redis.Redis): The Redis client instance for database operations.
    """
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
            "tier": "Bronze",  # Init to bronze status
            "request_limit": REQUEST_LIMIT_BRONZE,
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



async def get_similarity_threshhold(ss58_address: str, database: aioredis.Redis):
    """
    Retrieves the similarity threshold based on the tier of a given prover.
    This function returns the similarity threshold that a prover must maintain to remain in their tier.
    Args:
        ss58_address (str): The unique address (hotkey) of the prover.
        database (redis.Redis): The Redis client instance for database operations.
    Returns:
        float: The similarity threshold corresponding to the prover's tier.
    """
    if not await database.exists(ss58_address):
        bt.logging.warning(f"Prover key {ss58_address} is not registered!")
        return COSINE_SIMILARITY_THRESHOLD_BRONZE
    
    tier = await database.hget(f"stats:{ss58_address}", "tier")
    if tier == b"Challenger":
        return COSINE_SIMILARITY_THRESHOLD_CHALLENGER
    elif tier == b"Grandmaster":
        return COSINE_SIMILARITY_THRESHOLD_GRANDMASTER
    elif tier == b"Gold":
        return COSINE_SIMILARITY_THRESHOLD_GOLD
    elif tier == b"Silver":
        return COSINE_SIMILARITY_THRESHOLD_SILVER
    else:
        return COSINE_SIMILARITY_THRESHOLD_BRONZE

async def compute_tier(stats_key: str, database: aioredis.Redis):
    """
    Asynchronously computes the tier of a prover based on their performance statistics.
    The function calculates the success rate for each type of task and total successes,
    then updates the prover's tier if necessary. This could potentially change their Request Limit.
    Args:
        stats_key (str): The key in the database where the prover's statistics are stored.
        database (redis.Redis): The Redis client instance for database operations.
    """
    if not await database.exists(stats_key):
        bt.logging.warning(f"Prover key {stats_key} is not registered!")
        return

    # Get the number of successful challenges
    challenge_successes = int(await database.hget(stats_key, "challenge_successes"))
    # # Get the number of successful stores
    inference_successes = int(await database.hget(stats_key, "inference_successes"))
    # Get the number of total challenges
    challenge_attempts = int(await database.hget(stats_key, "challenge_attempts"))
    # Get the number of total stores
    inference_attempts = int(await database.hget(stats_key, "inference_attempts"))

    # Compute the success rate for each task type
    challenge_success_rate = (
        challenge_successes / challenge_attempts if challenge_attempts > 0 else 0
    )

    inference_success_rate = inference_successes / inference_attempts if inference_attempts > 0 else 0

    total_successes = await database.hget(stats_key, "total_successes")
    if total_successes is None:
        # This value wasn't stored. Legacy provers will have this issue.
        total_successes = inference_successes + challenge_successes
    total_successes = int(total_successes)

    if (
        challenge_success_rate >= CHALLENGER_CHALLENGE_SUCCESS_RATE
        # and inference_success_rate >= CHALLENGER_INFERENCE_SUCCESS_RATE
        and total_successes >= CHALLENGER_TIER_TOTAL_SUCCESSES
    ):
        tier = b"Challenger"
    elif (
        challenge_success_rate >= GRANDMASTER_CHALLENGE_SUCCESS_RATE
        # and inference_success_rate >= GRANDMASTER_INFERENCE_SUCCESS_RATE
        and total_successes >= GRANDMASTER_TIER_TOTAL_SUCCESSES
    ):
        tier = b"Grandmaster"
    elif (
        challenge_success_rate >= GOLD_CHALLENGE_SUCCESS_RATE
        # and inference_success_rate >= GOLD_INFERENCE_SUCCESS_RATE
        and total_successes >= GOLD_TIER_TOTAL_SUCCESSES
    ):
        tier = b"Gold"
    elif (
        challenge_success_rate >= SILVER_CHALLENGE_SUCCESS_RATE
        # and inference_success_rate >= SILVER_INFERENCE_SUCCESS_RATE
        and total_successes >= SILVER_TIER_TOTAL_SUCCESSES
    ):
        tier = b"Silver"
    else:
        tier = b"Bronze"

    # (Potentially) set the new tier in the stats hash
    current_tier = await database.hget(stats_key, "tier")
    if tier != current_tier:
        await database.hset(stats_key, "tier", tier)

        # Update the Request Limit
        if tier == b"Challenger":
            REQUEST_LIMIT = REQUEST_LIMIT_CHALLENGER
        elif tier == b"Grandmaster":
            REQUEST_LIMIT = REQUEST_LIMIT_GRANDMASTER
        elif tier == b"Gold":
            REQUEST_LIMIT = REQUEST_LIMIT_GOLD
        elif tier == b"Silver":
            REQUEST_LIMIT = REQUEST_LIMIT_SILVER
        else:
            REQUEST_LIMIT = REQUEST_LIMIT_BRONZE

        current_limit = await database.hget(stats_key, "request_limit")
        await database.hset(stats_key, "request_limit", REQUEST_LIMIT)
        bt.logging.trace(
            f"Request Limit for {stats_key} set from {current_limit} -> {REQUEST_LIMIT} bytes."
        )


async def compute_all_tiers(database: aioredis.Redis):
    # Iterate over all provers
    """
    Asynchronously computes and updates the tiers for all provers in the decentralized storage system.
    This function should be called periodically to ensure provers' tiers are up-to-date based on
    their performance. It iterates over all provers and calls `compute_tier` for each one.
    Args:
        database (redis.Redis): The Redis client instance for database operations.
    """
    provers = [prover async for prover in database.scan_iter("stats:*")]
    tasks = [compute_tier(prover, database) for prover in provers]
    await asyncio.gather(*tasks)

    # Reset the statistics for the next epoch
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