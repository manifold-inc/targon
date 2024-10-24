import math
from os import urandom
import time
import traceback
from typing import Dict, List, Optional, Tuple

from httpx import Timeout
import openai
from requests import post
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

    random_row_text = dataset.sample(n=1)["conversations"].iloc[0][0]["value"]
    # Generate a query from the sampled text and perform text generation
    messages = create_query_prompt(random_row_text)
    res: Optional[str] = None
    response = None
    for _ in range(3):
        try:
            response = post(
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
                        if chunk.choices[0].delta is None:
                            continue
                        if (
                            chunk.choices[0].delta.content == ""
                            or chunk.choices[0].delta.content is None
                        ) and len(stats.tokens) == 0:
                            continue
                        if start_token_time == 0:
                            start_token_time = time.time()
                        choice = chunk.choices[0]
                        logprob = -100
                        token_id = -1
                        choiceprobs = choice.logprobs
                        if choiceprobs is not None:
                            if choiceprobs.content:
                                logprob = choiceprobs.content[0].logprob
                                token_parts = choiceprobs.content[0].token.split(":")
                                if len(token_parts) > 1:
                                    token_id = int(token_parts[1])
                        stats.tokens.append(
                            {
                                "text": choice.delta.content or "",
                                "token_id": token_id,
                                "logprob": logprob,
                            }
                        )
                        token_times.append(time.time())
                case Endpoints.COMPLETION:
                    comp = await miner.completions.create(**request)
                    async for chunk in comp:
                        if (
                            chunk.choices[0].text == "" or chunk.choices[0].text is None
                        ) and len(stats.tokens) == 0:
                            continue
                        if start_token_time == 0:
                            start_token_time = time.time()
                        choice = chunk.choices[0]
                        if choice.logprobs is None:
                            continue
                        token_id = -1
                        logprob = -100
                        if choice.logprobs.token_logprobs:
                            logprob = choice.logprobs.token_logprobs[0]
                        if len(choice.logprobs.tokens) > 0:
                            token_parts = choice.logprobs.tokens[0].split(":")
                            if token_parts > 1:
                                token_id = token_parts[1]
                        stats.tokens.append(
                            {
                                "text": choice.text or "",
                                "token_id": token_id,
                                "logprob": logprob,
                            }
                        )
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
        time_to_first_token = start_token_time - start_send_message_time
        time_for_all_tokens = end_token_time - start_token_time
        if stats.error:
            return uid, stats
        stats.time_to_first_token = time_to_first_token
        stats.time_for_all_tokens = time_for_all_tokens
        stats.total_time = end_token_time - start_send_message_time
        stats.tps = min(len(stats.tokens), request["max_tokens"]) / stats.total_time

        # Detect when response was fully generated, then streamed, which leads to
        # poor user experience (slow time to N tokens vs total time).
        token_count = len(stats.tokens)
        if token_count > 60:
            time_to_5th_percent = (
                token_times[math.ceil(token_count * 0.05)] - start_send_message_time
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
    responses: List[Dict],
    uid,
    endpoint: Endpoints,
    port: int,
    url="http://localhost",
) -> Optional[Dict]:
    try:
        result = post(
            f"{url}:{port}/verify",
            headers={"Content-Type": "application/json"},
            json={
                "model": request.get("model"),
                "request_type": endpoint.value,
                "request_params": request,
                "output_sequence": responses,
            },
        ).json()
        if result.get("verified") is None:
            return None
        return result
    except Exception as e:
        bt.logging.error(f"{uid}: " + str(e))
        return None
