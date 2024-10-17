import numpy as np
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
        stream_quality=0,
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
                        if choice.model_extra is None:
                            continue
                        token_ids = choice.model_extra.get("token_ids") or []
                        token_id = token_ids[0] if len(token_ids) > 0 else -1
                        logprob = -100
                        choiceprobs = choice.logprobs
                        if choiceprobs is not None:
                            if choiceprobs.content:
                                logprob = choiceprobs.content[0].logprob
                        powv = choice.model_extra.get("powv", -1)
                        if powv is None:
                            powv = -1
                        stats.tokens.append(
                            {
                                "text": choice.delta.content or "",
                                "token_id": token_id or 0,
                                "powv": powv,
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
                        if choice.model_extra is None:
                            continue
                        if choice.logprobs is None:
                            continue
                        token_ids = choice.model_extra.get("token_ids") or []
                        token_id = token_ids[0] if len(token_ids) > 0 else -1
                        powv = choice.model_extra.get("powv", -1)
                        if powv is None:
                            powv = -1
                        logprob = -100
                        if choice.logprobs.token_logprobs:
                            logprob = choice.logprobs.token_logprobs[0]
                        stats.tokens.append(
                            {
                                "text": choice.text or "",
                                "token_id": token_id or 0,
                                "powv": powv,
                                "logprob": logprob,
                            }
                        )
                        token_times.append(time.time())
        except openai.APIConnectionError as e:
            bt.logging.trace(f"Miner {uid} failed request: {e}")
            stats.error = str(e)
        except Exception as e:
            bt.logging.trace(f"Unknown Error when sending to miner {uid}: {e}")
            stats.error = str(e)

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
        stats_token_count = min(len(stats.tokens), request["max_tokens"])
        stats.tps = stats_token_count / stats.total_time

        # Calculate a "streaming quality" score - a good user experience in LLM apps
        # typically consists of a good time to first token, fairly consistent times
        # between each token, and of course overall TPS.  Stream quality here is for
        # the former 2 aspects of good UX.

        # Variation in actual TPS vs TPS calculated only after having received the first.
        tps_from_first = min(len(stats.tokens), request["max_tokens"]) / time_for_all_tokens
        tps_divergence = max(1.0, stats.tps / tps_from_first)

        # Calculate a smoothness factor, not including first token.
        token_deltas = [
            token_times[idx+1] - token_times[idx]
            for idx in range(len(token_times) - 1)
        ]
        std = np.std(token_deltas[1:])
        mean = np.mean(token_deltas[1:])
        smoothness = sum([
            1 if mean - std <= value <= mean + std
            else 1.0 - max(1.0, abs(value - mean) / std / 10.0)
        ]) / stats_token_count

        # Time to half of tokens.
        half_time = np.median(token_times) - start_send_message_time
        half_delay_ratio = half_time / stats.total_time

        # Give zero score here if it's unlikely to have been streamed.
        if stats_token_count >= 10 and (tps_divergence <= 0.5 or half_delay_ratio >= 0.90):
            stats.stream_quality = 0.0
        else:
            stats.stream_quality = np.mean([tps_divergence, smoothness, half_delay_ratio])

        return uid, stats
    except Exception as e:
        bt.logging.error(f"{uid}: Error in forward for: {e}")
        bt.logging.error(traceback.format_exc())
        return uid, stats


@fail_with_none("Failed to check tokens")
async def check_tokens(
        request, responses: List[Dict], uid, endpoint: Endpoints, port: int, url='http://localhost'
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
        if err := result.get("error") is not None:
            bt.logging.error(str(err))
            return None
        return result
    except Exception as e:
        bt.logging.error(f"{uid}: " + str(e))
        return None
