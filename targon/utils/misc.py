# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
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
import time
import httpx
import typing
from math import floor
from targon import protocol
from typing import Callable, Any
from functools import lru_cache, update_wrapper

def return_json_params(sampling_params: protocol.ChallengeRequest) -> dict:
    return {
        "inputs": sampling_params.inputs,
        "parameters": {
            "best_of": sampling_params.parameters.best_of,
            "max_new_tokens": sampling_params.parameters.max_new_tokens,
            "seed": sampling_params.parameters.seed,
            "do_sample": sampling_params.parameters.do_sample,
            "repetition_penalty": sampling_params.parameters.repetition_penalty,
            "temperature": sampling_params.parameters.temperature,
            "top_k": sampling_params.parameters.top_k,
            "top_p": sampling_params.parameters.top_p,
            "truncate": sampling_params.parameters.truncate,
            "typical_p": sampling_params.parameters.typical_p,
            "watermark": sampling_params.parameters.watermark,
            "return_full_text": False
        },
        "stream": False
    }

async def get_generated_text(url: str, sampling_params: protocol.ChallengeRequest) -> typing.Optional[str]:
    json_params = {
            "inputs": sampling_params.inputs,
            "parameters": {
                "best_of": sampling_params.parameters.best_of,
                "max_new_tokens": sampling_params.parameters.max_new_tokens,
                "seed": sampling_params.parameters.seed,
                "do_sample": sampling_params.parameters.do_sample,
                "repetition_penalty": sampling_params.parameters.repetition_penalty,
                "temperature": sampling_params.parameters.temperature,
                "top_k": sampling_params.parameters.top_k,
                "top_p": sampling_params.parameters.top_p,
                "truncate": sampling_params.parameters.truncate,
                "typical_p": sampling_params.parameters.typical_p,
                "watermark": sampling_params.parameters.watermark,
                "return_full_text": False
            },
            "stream": False

    }
    async with httpx.AsyncClient() as client:
        try:
            # bt.logging.trace(f"Sending request to {url} with sampling params {sampling_params.dict()}")
            response = await client.post(format_url(url), json=json_params)
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses
            data = response.json()
            
            return data[0].get("generated_text", None)
        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}.")
        except httpx.HTTPStatusError as exc:
            print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
            print(f"Response text: {exc.response.text}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None

def format_url(url: str) -> str:
    """
    Formats the given URL by ensuring that it starts with http:// and ends with a trailing slash.

    Args:
        url (str): The URL to be formatted.

    Returns:
        str: The formatted URL.

    This function is useful for ensuring that URLs are consistently formatted with a trailing slash,
    which can be important for some web servers and APIs.

    Example:
        url = format_url("example.com/api")
    """
    if not url.startswith("http"):
        url = f"http://{url}"
    if not url.endswith("/"):
        url += "/"
    return url

# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    """
    Decorator that creates a cache of the most recently used function calls with a time-to-live (TTL) feature.
    The cache evicts the least recently used entries if the cache exceeds the `maxsize` or if an entry has
    been in the cache longer than the `ttl` period.

    Args:
        maxsize (int): Maximum size of the cache. Once the cache grows to this size, subsequent entries
                       replace the least recently used ones. Defaults to 128.
        typed (bool): If set to True, arguments of different types will be cached separately. For example,
                      f(3) and f(3.0) will be treated as distinct calls with distinct results. Defaults to False.
        ttl (int): The time-to-live for each cache entry, measured in seconds. If set to a non-positive value,
                   the TTL is set to a very large number, effectively making the cache entries permanent. Defaults to -1.

    Returns:
        Callable: A decorator that can be applied to functions to cache their return values.

    The decorator is useful for caching results of functions that are expensive to compute and are called
    with the same arguments frequently within short periods of time. The TTL feature helps in ensuring
    that the cached values are not stale.

    Example:
        @ttl_cache(ttl=10)
        def get_data(param):
            # Expensive data retrieval operation
            return data
    """
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


def _ttl_hash_gen(seconds: int):
    """
    Internal generator function used by the `ttl_cache` decorator to generate a new hash value at regular
    time intervals specified by `seconds`.

    Args:
        seconds (int): The number of seconds after which a new hash value will be generated.

    Yields:
        int: A hash value that represents the current time interval.

    This generator is used to create time-based hash values that enable the `ttl_cache` to determine
    whether cached entries are still valid or if they have expired and should be recalculated.
    """
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    """
    Retrieves the current block number from the blockchain. This method is cached with a time-to-live (TTL)
    of 12 seconds, meaning that it will only refresh the block number from the blockchain at most every 12 seconds,
    reducing the number of calls to the underlying blockchain interface.

    Returns:
        int: The current block number on the blockchain.

    This method is useful for applications that need to access the current block number frequently and can
    tolerate a delay of up to 12 seconds for the latest information. By using a cache with TTL, the method
    efficiently reduces the workload on the blockchain interface.

    Example:
        current_block = ttl_get_block(self)

    Note: self here is the prover or verifier instance
    """
    return self.subtensor.get_current_block()