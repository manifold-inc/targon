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

import json
import aioredis
import asyncio
import bittensor as bt
from typing import Dict, List, Any, Union, Optional, Tuple


# Function to add metadata to a hash in Redis
async def add_metadata_to_hotkey(
    ss58_address: str, data_hash: str, metadata: Dict, database: aioredis.Redis
):
    """
    Associates a data hash and its metadata with a hotkey in Redis.

    Parameters:
        ss58_address (str): The primary key representing the hotkey.
        data_hash (str): The subkey representing the data hash.
        metadata (dict): The metadata to associate with the data hash. Includes the size of the data, the seed,
            and the encryption payload. E.g. {'size': 123, 'seed': 456, 'encryption_payload': 'abc'}.
        database (aioredis.Redis): The Redis client instance.
    """
    # Serialize the metadata as a JSON string
    metadata_json = json.dumps(metadata)
    # Use HSET to associate the data hash with the hotkey
    key = f"hotkey:{ss58_address}"
    await database.hset(key, data_hash, metadata_json)
    bt.logging.trace(f"Associated data hash {data_hash} with hotkey {ss58_address}.")


async def remove_metadata_from_hotkey(
    ss58_address: str, data_hash: str, database: aioredis.Redis
):
    """
    Removes a data hash and its metadata from a hotkey in Redis.

    Parameters:
        ss58_address (str): The primary key representing the hotkey.
        data_hash (str): The subkey representing the data hash.
        database (aioredis.Redis): The Redis client instance.
    """
    # Use HDEL to remove the data hash from the hotkey
    key = f"hotkey:{ss58_address}"
    await database.hdel(key, data_hash)
    bt.logging.trace(f"Removed data hash {data_hash} from hotkey {ss58_address}.")


async def get_metadata_for_hotkey(
    ss58_address: str, database: aioredis.Redis
) -> Dict[str, dict]:
    """
    Retrieves all data hashes and their metadata for a given hotkey.

    Parameters:
        ss58_address (str): The key representing the hotkey.
        database (aioredis.Redis): The Redis client instance.

    Returns:
        A dictionary where keys are data hashes and values are the associated metadata.
    """
    # Fetch all fields (data hashes) and values (metadata) for the hotkey
    all_data_hashes = await database.hgetall(f"hotkey:{ss58_address}")
    bt.logging.trace(
        f"get_metadata_for_hotkey() # hashes found for hotkey {ss58_address}: {len(all_data_hashes)}"
    )

    # Deserialize the metadata for each data hash
    return {
        data_hash.decode("utf-8"): json.loads(metadata.decode("utf-8"))
        for data_hash, metadata in all_data_hashes.items()
    }


async def get_hashes_for_hotkey(
    ss58_address: str, database: aioredis.Redis
) -> List[str]:
    """
    Retrieves all data hashes and their metadata for a given hotkey.

    Parameters:
        ss58_address (str): The key representing the hotkey.
        database (aioredis.Redis): The Redis client instance.

    Returns:
        A dictionary where keys are data hashes and values are the associated metadata.
    """
    # Fetch all fields (data hashes) and values (metadata) for the hotkey
    all_data_hashes = await database.hgetall(f"hotkey:{ss58_address}")

    # Deserialize the metadata for each data hash
    return [
        data_hash.decode("utf-8") for data_hash, metadata in all_data_hashes.items()
    ]


async def remove_hashes_for_hotkey(
    ss58_address: str, hashes: list, database: aioredis.Redis
) -> List[str]:
    """
    Retrieves all data hashes and their metadata for a given hotkey.

    Parameters:
        ss58_address (str): The key representing the hotkey.
        database (aioredis.Redis): The Redis client instance.

    Returns:
        A dictionary where keys are data hashes and values are the associated metadata.
    """
    bt.logging.trace(
        f"remove_hashes_for_hotkey() removing {len(hashes)} hashes from hotkey {ss58_address}"
    )
    for _hash in hashes:
        await remove_metadata_from_hotkey(ss58_address, _hash, database)


async def update_metadata_for_data_hash(
    ss58_address: str, data_hash: str, new_metadata: dict, database: aioredis.Redis
):
    """
    Updates the metadata for a specific data hash associated with a hotkey.

    Parameters:
        ss58_address (str): The key representing the hotkey.
        data_hash (str): The subkey representing the data hash to update.
        new_metadata (dict): The new metadata to associate with the data hash.
        database (aioredis.Redis): The Redis client instance.
    """
    # Serialize the new metadata as a JSON string
    new_metadata_json = json.dumps(new_metadata)
    # Update the field in the hash with the new metadata
    await database.hset(f"hotkey:{ss58_address}", data_hash, new_metadata_json)
    bt.logging.trace(
        f"Updated metadata for data hash {data_hash} under hotkey {ss58_address}."
    )


async def get_metadata_for_hotkey_and_hash(
    ss58_address: str, data_hash: str, database: aioredis.Redis, verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Retrieves specific metadata from a hash in Redis for the given field_key.

    Parameters:
        ss58_address (str): The hotkey assoicated.
        data_hash (str): The data hash associated.
        databse (aioredis.Redis): The Redis client instance.

    Returns:
        The deserialized metadata as a dictionary, or None if not found.
    """
    # Get the JSON string from Redis
    metadata_json = await database.hget(f"hotkey:{ss58_address}", data_hash)
    if verbose:
        bt.logging.trace(
            f"hotkey {ss58_address[:16]} | data_hash {data_hash[:16]} | metadata_json {metadata_json}"
        )
    if metadata_json:
        # Deserialize the JSON string to a Python dictionary
        metadata = json.loads(metadata_json)
        return metadata
    else:
        bt.logging.trace(f"No metadata found for {data_hash} in hash {ss58_address}.")
        return None


async def get_all_chunk_hashes(database: aioredis.Redis) -> Dict[str, List[str]]:
    """
    Retrieves all chunk hashes and associated metadata from the Redis instance.

    Parameters:
        database (aioredis.Redis): The Redis client instance.

    Returns:
        A dictionary where keys are chunk hashes and values are lists of hotkeys associated with each chunk hash.
    """
    # Initialize an empty dictionary to store the inverse map
    chunk_hash_hotkeys = {}

    # Retrieve all hotkeys (assuming keys are named with a 'hotkey:' prefix)
    async for hotkey in database.scan_iter("*"):
        if not hotkey.startswith(b"hotkey:"):
            continue
        # Fetch all fields (data hashes) for the current hotkey
        data_hashes = await database.hkeys(hotkey)
        # Iterate over each data hash and append the hotkey to the corresponding list
        for data_hash in data_hashes:
            data_hash = data_hash.decode("utf-8")
            if data_hash not in chunk_hash_hotkeys:
                chunk_hash_hotkeys[data_hash] = []
            chunk_hash_hotkeys[data_hash].append(hotkey.decode("utf-8").split(":")[1])

    return chunk_hash_hotkeys


async def total_hotkey_requests(
    hotkey: str, database: aioredis.Redis, verbose: bool = False
) -> int:
    """
    Calculates the total number of requests made for a hotkey in the database.

    Parameters:
        database (aioredis.Redis): The Redis client instance.
        hotkey (str): The key representing the hotkey.

    Returns:
        The total number of requests made for the hotkey.
    """
    total_requests = 0
    keys = await database.hkeys(f"hotkey:{hotkey}")
    for data_hash in keys:
        # Get the metadata for the current data hash
        metadata = await get_metadata_for_hotkey_and_hash(
            hotkey, data_hash, database, verbose
        )
        if metadata:
            # Add the number of requests to the total
            total_requests += (metadata["inference_attempts"] + metadata["challenge_attempts"])
    return total_requests


async def hotkey_at_capacity(
    hotkey: str, database: aioredis.Redis, verbose: bool = False
) -> bool:
    """
    Checks if the hotkey is at capacity.

    Parameters:
        database (aioredis.Redis): The Redis client instance.
        hotkey (str): The key representing the hotkey.

    Returns:
        True if the hotkey is at capacity, False otherwise.
    """
    # Get the total storage used by the hotkey
    total_requests = await total_hotkey_requests(hotkey, database, verbose)
    # Check if the hotkey is at capacity
    request_limit = await database.hget(f"stats:{hotkey}", "request_limit")
    if request_limit is None:
        if verbose:
            bt.logging.trace(f"Could not find request limit for {hotkey}.")
        return False
    try:
        limit = int(request_limit)
    except Exception as e:
        if verbose:
            bt.logging.trace(f"Could not parse storage limit for {hotkey} | {e}.")
        return False
    if total_requests >= limit:
        if verbose:
            bt.logging.trace(
                f"Hotkey {hotkey} is at max capacity {limit} Req."
            )
        return True
    else:
        if verbose:
            bt.logging.trace(
                f"Hotkey {hotkey} has {(limit - total_requests)} free."
            )
        return False


async def cache_hotkeys_capacity(
    hotkeys: List[str], database: aioredis.Redis, verbose: bool = False
):
    """
    Caches the capacity information for a list of hotkeys.

    Parameters:
        hotkeys (list): List of hotkey strings to check.
        database (aioredis.Redis): The Redis client instance.

    Returns:
        dict: A dictionary with hotkeys as keys and a tuple of (total_storage, limit) as values.
    """
    hotkeys_capacity = {}

    for hotkey in hotkeys:
        # Get the total storage used by the hotkey
        total_requests = await total_hotkey_requests(hotkey, database, verbose)
        # Get the byte limit for the hotkey
        request_limit = await database.hget(f"stats:{hotkey}", "request_limit")

        if request_limit is None:
            bt.logging.warning(f"Could not find request limit for {hotkey}.")
            limit = None
        else:
            try:
                limit = int(request_limit)
            except Exception as e:
                bt.logging.warning(f"Could not parse storage limit for {hotkey} | {e}.")
                limit = None

        hotkeys_capacity[hotkey] = (total_requests, limit)

    return hotkeys_capacity


async def check_hotkeys_capacity(hotkeys_capacity, hotkey: str, verbose: bool = False):
    """
    Checks if a hotkey is at capacity using the cached information.

    Parameters:
        hotkeys_capacity (dict): Dictionary with cached capacity information.
        hotkey (str): The key representing the hotkey.

    Returns:
        True if the hotkey is at capacity, False otherwise.
    """
    total_requests, limit = hotkeys_capacity.get(hotkey, (0, None))

    if limit is None:
        # Limit information not available or couldn't be parsed
        return False

    if total_requests >= limit:
        if verbose:
            bt.logging.trace(
                f"Hotkey {hotkey} is at max capacity {limit} Req."
            )
        return True
    else:
        if verbose:
            bt.logging.trace(
                f"Hotkey {hotkey} has {(limit - total_requests)} free."
            )
        return False


async def total_verifier_requests(database: aioredis.Redis) -> int:
    """
    Calculates the total request used by all hotkeys in the database.

    Parameters:
        database (aioredis.Redis): The Redis client instance.

    Returns:
        The total request used by all hotkeys in the database in bytes.
    """
    total_requests = 0
    async for key in database.scan_iter("*"):
        if not key.startswith(b"stats:"):
            continue
        # Get the total storage used by the hotkey
        total_requests += await total_hotkey_requests(key.decode("utf-8").split(":")[-1], database)
    return total_requests

async def get_prover_statistics(database: aioredis.Redis) -> Dict[str, Dict[str, str]]:
    """
    Retrieves statistics for all provers in the database.
    Parameters:
        database (aioredis.Redis): The Redis client instance.
    Returns:
        A dictionary where keys are hotkeys and values are dictionaries containing the statistics for each hotkey.
    """
    stats = {}
    async for key in database.scan_iter(b"stats:*"):
        # Await the hgetall call and then process its result
        key_stats = await database.hgetall(key)
        # Process the key_stats as required
        processed_stats = {
            k.decode("utf-8"): v.decode("utf-8") for k, v in key_stats.items()
        }
        stats[key.decode("utf-8").split(":")[-1]] = processed_stats

    return stats


async def get_single_prover_statistics(
    ss58_address: str, database: aioredis.Redis
) -> Dict[str, Dict[str, str]]:
    """
    Retrieves statistics for all provers in the database.
    Parameters:
        database (aioredis.Redis): The Redis client instance.
    Returns:
        A dictionary where keys are hotkeys and values are dictionaries containing the statistics for each hotkey.
    """
    stats = await database.hgetall(f"stats:{ss58_address}")
    return {k.decode("utf-8"): v.decode("utf-8") for k, v in stats.items()}


async def get_redis_db_size(database: aioredis.Redis) -> int:
    """
    Calculates the total approximate size of all keys in a Redis database.
    Parameters:
        database (int): Redis database
    Returns:
        int: Total size of all keys in bytes
    """
    total_size = 0
    async for key in await database.scan_iter("*"):
        size = await database.execute_command("MEMORY USAGE", key)
        if size:
            total_size += size
    return total_size

