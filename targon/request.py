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


def get_tool_parser_for_model(model_name: str) -> Optional[str]:
    """Determine if a model supports tool calling and return its parser type.
    Based on vLLM's supported models documentation."""

    model_lower = model_name.lower()

    # For combined models like "hermes-3-llama-3.1", prefer llama3_json parser
    if "llama-3.1" in model_lower:
        return "llama3_json"

    # Hermes models (hermes)
    # All Nous Research Hermes-series models newer than Hermes 2 Pro
    models = ["hermes-3", "hermes-2-pro"]
    if model_lower in models:
        return "hermes"

    return None


@fail_with_none("Error generating dataset")
def generate_request(
    dataset,
    tool_dataset,
    model_name,
    endpoint: Endpoints,
    metadata,
):
    # Generate a random seed for reproducibility in sampling and text generation
    if metadata is None:
        bt.logging.error(f"No generator / verifier found for {model_name}")
        return None
    random.seed(urandom(100))
    seed = random.randint(10000, 10000000)
    temperature = random.random()
    max_tokens = random.randint(512, 2048)

    # Sample a random row from the prompt dataset
    total_rows = len(dataset["train"])
    random_row_text = dataset["train"][random.randint(0, total_rows - 1)][
        "conversations"
    ][0]["value"]

    # Generate a query from the sampled text and perform text generation
    messages = create_query_prompt(random_row_text)
    res: Optional[str] = None
    response = None
    bt.logging.info("Starting synthetic generation")
    for _ in range(2):
        try:
            response = requests.post(
                f"{metadata['url']}:{metadata['port']}/generate",
                headers={"Content-Type": "application/json"},
                timeout=30,
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
            break
        except Exception:
            bt.logging.error(f"Failed to generate request for {model_name}")
    if res is None:
        bt.logging.error(
            f"Failed to generate prompt for {model_name}: {endpoint}: {response}"
        )
        return None

    # Create base request parameters
    request_params = {
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model": model_name,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": True,
        **create_search_prompt(res, endpoint),
    }

    # Add tools with 25% chance if dataset exists
    tool_parser = get_tool_parser_for_model(model_name)
    if (
        tool_dataset
        and tool_parser
        and len(tool_dataset["train"]) > 0
        and random.random() < 0.25
        and endpoint == Endpoints.CHAT
    ):  # 25% chance to include tools
        # Sample 2-5 random scenarios, each containing one or more related tools
        num_tools = random.randint(2, 5)
        dataset_length = len(tool_dataset["train"])
        available_indices = list(range(dataset_length))
        selected_indices = random.sample(
            available_indices, min(num_tools, dataset_length)
        )

        # Collect all tools from selected rows
        tools = []
        for idx in selected_indices:
            row = tool_dataset["train"][idx]
            tools.extend(row["tools"])

        request_params.update(
            {"tools": tools, "tool_choice": "auto"}  # Always use auto mode
        )

    return request_params


async def handle_inference(
    metagraph: "bt.metagraph",
    wallet: "bt.wallet",
    request,
    uid: int,
    endpoint: Endpoints,
) -> Tuple[int, InferenceStats]:
    stats = InferenceStats(
        gpus=1,
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
            timeout=Timeout(60, connect=5, read=5),
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
            bt.logging.error(f"Unknown Error when sending to miner {uid}: {e}")
            stats.error = str(e)
            stats.cause = "BAD_STREAM"

        if stats.error:
            return uid, stats

        if start_token_time == 0:
            start_token_time = time.time()
        end_token_time = time.time()

        stats.time_to_first_token = start_token_time - start_send_message_time
        stats.time_for_all_tokens = end_token_time - start_token_time
        stats.total_time = end_token_time - start_send_message_time
        stats.tps = min(len(stats.tokens), request["max_tokens"]) / stats.total_time

        return uid, stats

    except Exception as e:
        bt.logging.error(f"{uid}: Error in forward for: {e}")
        bt.logging.error(traceback.format_exc())
        return uid, stats


@fail_with_none("Failed to check tokens")
async def check_tokens(
    request,
    raw_chunks: List[Dict],
    endpoint: Endpoints,
    port: int,
    url,
) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        request_data = {
            "model": request.get("model"),
            "request_type": endpoint.value,
            "request_params": request,
            "raw_chunks": raw_chunks,
        }
        response = requests.post(
            f"{url}:{port}/verify",
            headers={"Content-Type": "application/json"},
            timeout=30,
            json=request_data,
        )

        if response.status_code != 200:
            return None, response.text
        result = response.json()
        if result.get("verified") is None:
            return None, str(result)
        return result, None
    except Exception as e:
        return None, str(e)
