import math
from os import urandom
import time
import traceback
from typing import Dict, List, Optional, Tuple

from httpx import Timeout
import openai
import requests
from targon.dataset import create_query_prompt, create_search_prompt
from targon.epistula import create_header_hook
from targon.types import Endpoints, InferenceStats
from targon.utils import fail_with_none
import random
import bittensor as bt


@fail_with_none("Error generating dataset")
def generate_request(dataset, model_name, endpoint: Endpoints, port: int):
    # Generate a random seed for reproducibility in sampling and text generation
    random.seed(urandom(100))
    seed = random.randint(10000, 10000000)
    temperature = random.random()
    max_tokens = random.randint(512, 1920)

    total_rows = len(dataset["train"])
    random_row_text = dataset["train"][random.randint(0, total_rows - 1)][
        "conversations"
    ][0]["value"]
    # Generate a query from the sampled text and perform text generation
    messages = create_query_prompt(random_row_text)
    res: Optional[str] = None
    response = None
    for _ in range(3):
        try:
            response = requests.post(
                f"http://localhost:{port}/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "messages": messages,
                    "sampling_params": {
                        "temperature": 0.5,
                        "seed": seed,
                        "max_tokens": random.randint(16, 64),
                    },
                },
            )
            if response.status_code != 200:
                bt.logging.error(f"Failed to generate request for {model_name}")
                return None
            res = response.json().get("text")
        except Exception:
            bt.logging.error(f"Failed to generate request for {model_name}")
        break
    if res is None:
        bt.logging.error(
            f"Failed to generate prompt for {model_name}: {endpoint}: {response}"
        )
        return None

    # Create sampling parameters using the generated seed and token limit
    return {
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model": model_name,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": True,
        **create_search_prompt(res, endpoint),
    }


async def handle_inference(
    metagraph: "bt.metagraph",
    wallet: "bt.wallet",
    request,
    uid: int,
    endpoint: Endpoints,
) -> Tuple[int, InferenceStats]:
    stats = InferenceStats(
        time_to_first_token=0,
        time_for_all_tokens=0,
        tps=0,
        total_time=0,
        tokens=[],
        verified=False,
    )
    try:
        axon_info = metagraph.axons[uid]
        miner = openai.AsyncOpenAI(
            base_url=f"http://{axon_info.ip}:{axon_info.port}/v1",
            api_key="sn4",
            max_retries=0,
            timeout=Timeout(12, connect=5, read=5),
            http_client=openai.DefaultAsyncHttpxClient(
                event_hooks={
                    "request": [
                        create_header_hook(
                            wallet.hotkey, axon_info.hotkey, request["model"]
                        )
                    ]
                }
            ),
        )
        start_token_time = 0
        start_send_message_time = time.time()
        token_times = []
        try:
            match endpoint:
                case Endpoints.CHAT:
                    chat = await miner.chat.completions.create(**request)
                    async for chunk in chat:
                        # Store raw chunk
                        stats.tokens.append(chunk.model_dump())
                            
                        # Track timing
                        if start_token_time == 0:
                            start_token_time = time.time()
                        token_times.append(time.time())

                case Endpoints.COMPLETION:
                    comp = await miner.completions.create(**request)
                    async for chunk in comp:
                        # Store raw chunk
                        stats.tokens.append(chunk.model_dump())
                            
                        # Track timing
                        if start_token_time == 0:
                            start_token_time = time.time()
                        token_times.append(time.time())

        except openai.APIConnectionError as e:
            bt.logging.trace(f"Miner {uid} failed request: {e}")
            stats.error = str(e)
            stats.cause = "BAD_STREAM"
        except Exception as e:
            bt.logging.trace(f"Unknown Error when sending to miner {uid}: {e}")
            stats.error = str(e)
            stats.cause = "BAD_STREAM"

        if start_token_time == 0:
            start_token_time = time.time()
        end_token_time = time.time()
        
        stats.time_to_first_token = start_token_time - start_send_message_time
        stats.time_for_all_tokens = end_token_time - start_token_time
        stats.total_time = end_token_time - start_send_message_time
        stats.tps = min(len(stats.tokens), request["max_tokens"]) / stats.total_time

        # Check for non-streaming behavior
        if len(stats.tokens) > 60:
            time_to_5th_percent = (
                token_times[math.ceil(len(stats.tokens) * 0.05)] - start_send_message_time
            )
            if time_to_5th_percent / stats.total_time >= 0.85:
                stats.verified = False
                stats.error = "Likely non-streamed response"
                stats.cause = "BAD_STREAM"

        return uid, stats

    except Exception as e:
        bt.logging.error(f"{uid}: Error in forward for: {e}")
        bt.logging.error(traceback.format_exc())
        return uid, stats


@fail_with_none("Failed to check tokens")
async def check_tokens(
    request,
    raw_chunks: List[Dict],
    uid,
    endpoint: Endpoints,
    port: int,
    url="http://localhost",
) -> Optional[Dict]:
    try:
        result = requests.post(
            f"{url}:{port}/verify",
            headers={"Content-Type": "application/json"},
            json={
                "model": request.get("model"),
                "request_type": endpoint.value,
                "request_params": request,
                "output_sequence": raw_chunks,
            },
        ).json()
        if result.get("verified") is None:
            bt.logging.error(str(result))
            return None
        return result
    except Exception as e:
        bt.logging.error(f"{uid}: " + str(e))
        return None
