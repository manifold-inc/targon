from collections import defaultdict
import random
import math
import os
import asyncio
import traceback
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams

# Load the model.
MODEL_NAME = os.getenv("MODEL")
if MODEL_NAME is None:
    exit()

GPU_MEMORY_UTIL = float(os.getenv("GPU_MEMORY_UTIL"))
if GPU_MEMORY_UTIL == 0:
    exit()
# Constants.
LOGPROB_LOG_THRESHOLD = 0.65
LOGPROB_FAILURE_THRESHOLD = 0.75
TOP_LOGPROBS = 10
TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL", 1))
MODEL_WRAPPER = LLM(
    model=MODEL_NAME,
    enforce_eager=True,
    gpu_memory_utilization=GPU_MEMORY_UTIL,
    tensor_parallel_size=TENSOR_PARALLEL,
)
TOKENIZER = MODEL_WRAPPER.get_tokenizer()
MODEL = MODEL_WRAPPER.llm_engine.model_executor.driver_worker.model_runner.model  # type: ignore
MODEL_NUM_PARAMS = sum(1 for _ in MODEL.parameters())

# Lock to ensure atomicity.
LOCK = asyncio.Lock()
LOCK_GENERATE = asyncio.Lock()

ENDPOINTS = ["completion"]
if TOKENIZER.chat_template is not None:
    ENDPOINTS.append("chat")


class RequestParams(BaseModel):
    messages: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int


class OutputItem(BaseModel):
    text: str
    logprob: float
    powv: int
    token_id: int


class RequestType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


class VerificationRequest(BaseModel):
    request_type: str
    model: str = MODEL_NAME
    request_params: RequestParams
    output_sequence: List[OutputItem]


class RequestSamplingParams(BaseModel):
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int


class GenerateRequest(BaseModel):
    messages: List[Dict[str, str]]
    sampling_params: RequestSamplingParams


app = FastAPI()


@app.post("/generate")
async def generate_question(req: GenerateRequest):
    async with LOCK_GENERATE:
        try:
            if "chat" in ENDPOINTS:
                output = (
                    MODEL_WRAPPER.chat(
                        messages=req.messages, sampling_params=SamplingParams(**req.sampling_params.model_dump()), use_tqdm=False  # type: ignore
                    )[0]
                    .outputs[0]
                    .text
                )
            else:
                prompt = ""
                for message in req.messages:
                    prompt += (
                        message.get("role", "")
                        + ": "
                        + message.get("content", "")
                        + "\n"
                    )
                prompt += "\nResponse: "
                output = (
                    MODEL_WRAPPER.generate(
                        prompts=prompt,
                        sampling_params=SamplingParams(
                            **req.sampling_params.model_dump()
                        ),
                        use_tqdm=False,
                    )[0]
                    .outputs[0]
                    .text
                )
            return {"text": output}
        except Exception as e:
            print("Failed generate request", str(e), traceback.format_exc())
        return {"text": None}


def verify_logprobs_random(
    request: VerificationRequest, input_text: str
) -> Tuple[bool, str]:
    """
    Generate a handful of random outputs to ensure the logprobs weren't generated after the fact.
    """
    indices = list(range(1, len(request.output_sequence) - 1))
    indices_to_check = list(
        sorted(
            [
                0,  # always check first token
                len(request.output_sequence) - 1,  # always check last token
            ] + random.sample(indices, min(len(indices), 3))
        )
    )

    # Generate a single token at each index, comparing logprobs.
    top_logprobs = int(request.request_params.temperature * 10) + 3
    sampling_params = SamplingParams(
        temperature=request.request_params.temperature,
        seed=request.request_params.seed,
        max_tokens=1,
        logprobs=top_logprobs,
    )
    for idx in indices_to_check:
        full_text = input_text + "".join(
            [item.text for item in request.output_sequence[0:idx]]
        )
        output = MODEL_WRAPPER.generate([full_text], sampling_params, use_tqdm=False)[
            0
        ].outputs[0]

        # The miner's output token should be in the logprobs...
        top_tokens = []
        if output.logprobs is None:
            print("No log probs to check")
            continue
        for lp in output.logprobs:
            top_tokens += list(lp.keys())
        if request.output_sequence[idx].token_id not in top_tokens:
            message = f"Token output at index {idx} [{TOKENIZER.decode([request.output_sequence[idx]])}] not found in top {top_logprobs} logprobs: {[TOKENIZER.decode([token]) for token in top_tokens]}"
            return False, message
    return (
        True,
        f"Successfully verified {len(indices_to_check)} random logprobs: {indices_to_check}",
    )

def find_repeated_subsequences(token_list):
    n = len(token_list)

    # Squeeze token into int list in range [0, (unique token count)]
    token_to_int = {}
    int_token_list = []
    current_int = 0
    for token in token_list:
        if token not in token_to_int:
            token_to_int[token] = current_int
            current_int += 1
        int_token_list.append(token_to_int[token])

    # Generate suffix array.
    suffixes = [(int_token_list[i:], i) for i in range(n)]
    suffixes.sort()  # O(n log n)
    SA = [suffix[1] for suffix in suffixes]

    # Discover longest common prefixes.
    rank = [0]*n
    for i in range(n):
        rank[SA[i]] = i
    LCP = [0]*(n-1)
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = SA[rank[i]-1]
            while i + h < n and j + h < n and int_token_list[i + h] == int_token_list[j + h]:
                h += 1
            LCP[rank[i]-1] = h
            if h > 0:
                h -= 1

    # Find repeats.
    repeats = defaultdict(list)
    stack = []
    for i in range(len(LCP)):
        lcp_length = LCP[i]
        if lcp_length >= 15:
            positions = [SA[i], SA[i+1]]
            j = i+1
            min_lcp = lcp_length
            while j < len(LCP) and LCP[j] >= 15:
                min_lcp = min(min_lcp, LCP[j])
                positions.append(SA[j+1])
                j += 1
            substr = tuple(token_list[SA[i]:SA[i]+min_lcp])
            repeats[substr].extend(positions)
            i = j

    # De-dupe.
    final_subsequences = {}
    for substr, positions in repeats.items():
        unique_positions = sorted(set(positions))
        substr_length = len(substr)
        valid_positions = []
        for pos in unique_positions:
            if pos + substr_length <= n:
                valid_positions.append(pos)
        if len(valid_positions) > 1:
            final_subsequences[substr] = len(valid_positions)
    return final_subsequences

def verify_logprobs(
    request: VerificationRequest, input_text: str, input_tokens: List[int]
) -> Optional[Tuple[bool, str]]:
    """
    Compare the produced logprob values against the ground truth, or at least
    the ground truth according to this particular GPU/software pairing.
    """

    # Set up sampling parameters for the "fast" check, which just compares input logprobs against output logprobs.
    sampling_params = SamplingParams(
        temperature=request.request_params.temperature,
        seed=request.request_params.seed,
        max_tokens=1,
        logprobs=1,
        prompt_logprobs=5,
    )

    # Generate output for a single token, which will return input logprobs based on prompt_logprobs=1
    output = None
    for _ in range(5):
        full_text = input_text + "".join(
            [item.text for item in request.output_sequence]
        )
        output = MODEL_WRAPPER.generate([full_text], sampling_params, use_tqdm=False)[0]
        if output.prompt_logprobs is not None:
            break

    if not output or output.prompt_logprobs is None:
        return None

    # The actual logprobs should be *very* close, but typically not 100% because of GPU/driver/etc. differences.
    total_score = 0.0
    idxs = min(
        len(output.prompt_logprobs) - len(input_tokens) - 3,
        len(request.output_sequence) - 1,
    )
    perfect_tokens = 0
    eos_token_id = getattr(TOKENIZER, "eos_token_id", -1)
    eot_token_id = TOKENIZER.get_vocab().get("<|eot_id|>", -1)
    output_tokens = [item.token_id for item in request.output_sequence]
    really_low_prob = 0
    not_first = 0
    for idx in range(idxs):
        item = request.output_sequence[idx]
        expected_logprob = output.prompt_logprobs[idx + len(input_tokens)]
        assert expected_logprob is not None
        eos_logprob = expected_logprob.get(eos_token_id)
        eot_logprob = expected_logprob.get(eot_token_id)
        if not eos_logprob and eot_logprob or (eos_logprob and eot_logprob and eot_logprob.rank < eos_logprob.rank):
            eos_logprob = eot_logprob
        expected_logprob = expected_logprob.get(item.token_id)
        if eos_logprob and (not expected_logprob or eos_logprob.rank < expected_logprob.rank):
            return False, f"Expected EOS/EOT token at index {idx}"
        if expected_logprob is None:
            continue
        rank = expected_logprob.rank
        if rank >= 75:
            return False, f"Found extraordinarily improbable token '{TOKENIZER.decode([item.token_id])}' at index {idx}: {rank=}"
        elif rank >= 25:
            really_low_prob += 1
        if rank != 1:
            not_first += 1
        expected_logprob = expected_logprob.logprob
        produced_logprob = item.logprob
        score = 1.0 - min(
            1.0, abs(math.exp(expected_logprob) - math.exp(produced_logprob))
        )

        # Prevents over fitting smaller models
        if produced_logprob == 0:
            perfect_tokens += 1

        # To accomodate architectural difference and such, we'll give a perfect score if >= 0.9
        if score >= 0.9:
            score = 1.0

        # Logprobs rarely match well for high temps so we can use rank instead.
        if rank == 1 and request.request_params.temperature >= 0.9 and produced_logprob != 0:
            score = 1.0

        total_score += score

    # Check if miner produced non-top ranking tokens more than top-ranking tokens.
    ratio = not_first / len(output_tokens)
    if ratio >= 0.5:
        return False, f"{not_first} of {len(output_tokens)} [{ratio=}] tokens were not rank 1."

    # Calculate average score.
    average_score = round(total_score / idxs, 5)
    passes = average_score >= LOGPROB_FAILURE_THRESHOLD
    perfect_avg = round(perfect_tokens / idxs, 5)
    message = (
        f"Successfully verified logprob for {len(request.output_sequence)} outputs with {average_score} and {perfect_avg}% perfect tokens"
        if passes
        else f"Low average logprob score: {average_score}"
    )
    if passes and perfect_avg >= (1 - min(request.request_params.temperature, 0.6)):
        message = f"Overfitted response tokens. {perfect_avg}% perfect"
        passes = False
    if really_low_prob >= 5:
        message = f"Found {really_low_prob} highly improbable tokens."
        passes = False

    return (
        passes,
        message,
    )

def verify_repetition(output_tokens):
    for subseq, count in find_repeated_subsequences(output_tokens).items():
        repeated_text = TOKENIZER.decode(subseq)

        # How many times was this subsequence repeated?
        if count >= 10:
            return False, f"Found phrase repeated {count} times: {repeated_text}"

        # How much of the total output does the repeated text account for?
        ratio = (len(subseq) * count) / len(output_tokens)
        if ratio >= 0.75:
            return False, f"Found phrase repeated {count} times accounting for {ratio} of overall output: {repeated_text}"
    return True, "Reasonable token repetition."

@app.post("/verify")
async def verify(request: VerificationRequest) -> Dict:
    """Verify a miner's output."""

    # If the miner didn't return any outputs, fail.
    if len(request.output_sequence) < 3:
        return {
            "verified": False,
            "reason": "Output sequence too short!",
        }
    if (
        request.request_params.max_tokens
        and len(request.output_sequence) > request.request_params.max_tokens
    ):
        return {
            "verified": False,
            "reason": "Too many tokens produced!",
        }
    if request.model != MODEL_NAME:
        return {
            "verified": False,
            "reason": "Unable to verify model={request.model}, since we are using {MODEL_NAME}",
        }

    # Tokenize the input sequence.
    input_text = (
        request.request_params.prompt
        if request.request_type == RequestType.COMPLETION.value
        else TOKENIZER.apply_chat_template(
            request.request_params.messages,  # type: ignore
            tokenize=False,
            add_special_tokens=False,
        )
    )
    assert isinstance(input_text, str)
    if hasattr(TOKENIZER, "bos_token"):
        if input_text.startswith(TOKENIZER.bos_token):  # type: ignore
            input_text = input_text[len(TOKENIZER.bos_token) :]  # type: ignore
    input_tokens = TOKENIZER(input_text).input_ids

    # Verify!
    async with LOCK:
        return_value = {
            "verified": False,
            "logprob_fast_pass": False,
            "logprob_fast_message": None,
        }

        # Logprob checks.
        res = verify_logprobs(request, str(input_text), input_tokens)
        if res is None:
            return {"error": "Failed to check log probs"}
        result, message = res
        return_value.update(
            {
                "logprob_fast_pass": result,
                "logprob_fast_message": message,
            }
        )
        if not result:
            return return_value

        # Token repetition check.
        output_tokens = [item.token_id for item in request.output_sequence]
        result, message = verify_repetition(output_tokens)
        return_value.update(
            {
                "repetition_pass": result,
                "repetition_message": message,
            }
        )
        if not result:
            return return_value

        # Random logprob check.
        if request.request_params.temperature > 0.75:
            return_value.update({"verified": True})
            return return_value

        res = verify_logprobs_random(request, str(input_text))
        if res is None:
            return {"error": "Failed to check log probs"}
        result, message = res
        return_value.update(
            {
                "logprob_pass": result,
                "logprob_message": message,
            }
        )
        if not result:
            return return_value

        return_value.update({"verified": True})
        return return_value


@app.get("/endpoints")
def endpoints():
    return ENDPOINTS


@app.get("/")
def ping():
    return "", 200
