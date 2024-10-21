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
MODEL_NAME = os.getenv("MODEL", None)
if MODEL_NAME is None:
    exit()

GPU_MEMORY_UTIL = float(os.getenv("GPU_MEMORY_UTIL", 0))
if GPU_MEMORY_UTIL == 0:
    exit()
# Constants.
LOGPROB_LOG_THRESHOLD = 0.65
LOGPROB_FAILURE_THRESHOLD = 0.75
TOP_LOGPROBS = 10
TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL", 1))
print(MODEL_NAME, GPU_MEMORY_UTIL)
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


def verify_powv(
    request: VerificationRequest, input_tokens: List[int]
) -> Tuple[bool, str]:
    """
    Check the returned `powv` values against the ground truth.
    """
    input_sum = sum(input_tokens)
    if TENSOR_PARALLEL > 1:
        return (True, "")

    # Iterate through output sequence, checking powv values.
    output_sum = 0
    for idx in range(len(request.output_sequence) - 1):
        item = request.output_sequence[idx]
        powv = 0
        token_sum = input_sum + output_sum
        param_index = token_sum % MODEL_NUM_PARAMS
        for k, param in enumerate(MODEL.parameters()):
            if k != param_index:
                continue
            if param.dim() == 1:
                weights = param.tolist()
            else:
                tensor_index = output_sum % param.size()[0]
                weights = param[tensor_index].tolist()
            if len(weights) == 0:
                param_index += 1
                continue
            weight_index = input_sum % len(weights)
            powv = math.floor(weights[weight_index] * token_sum)
            if powv != item.powv:
                error_msg = (
                    f"Failed powv check at output index {idx}: {powv} vs {item.powv}"
                )
                return False, error_msg
            output_sum += item.token_id

    return (
        True,
        f"Successfully verified powv for {len(request.output_sequence)} outputs",
    )


def verify_logprobs_random(
    request: VerificationRequest, input_text: str
) -> Tuple[bool, str]:
    """
    Generate a handful of random outputs to ensure the logprobs weren't generated after the fact.
    """
    indices_to_check = list(
        sorted(
            [
                0,  # always check first token
                len(request.output_sequence) - 1,  # always check last token
                random.choice(
                    list(range(1, len(request.output_sequence) - 1))
                ),  # random offset
            ]
        )
    )

    # Generate a single token at each index, comparing logprobs.
    sampling_params = SamplingParams(
        temperature=request.request_params.temperature,
        seed=request.request_params.seed,
        max_tokens=1,
        logprobs=TOP_LOGPROBS,
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
            message = f"Token output at index {idx} [{request.output_sequence[idx]}] not found in top {TOP_LOGPROBS} top logprobs: {top_tokens}"
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
    highest_logprobs = []
    eos_expected = []
    eos_token_id = getattr(TOKENIZER, "eos_token_id", "-1")
    output_tokens = [item.token_id for item in request.output_sequence]
    for idx in range(idxs):
        item = request.output_sequence[idx]
        expected_logprob = output.prompt_logprobs[idx + len(input_tokens)]
        assert expected_logprob is not None
        highest_logprobs.append(max([lp.logprob for lp in expected_logprob.values()]))
        eos_logprob = expected_logprob.get(eos_token_id)
        expected_logprob = expected_logprob.get(item.token_id)
        if eos_logprob is not None and eos_logprob.logprob == highest_logprobs[-1]:
            eos_expected.append(eos_logprob.logprob)
        if expected_logprob is None:
            continue
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

        total_score += score

    # Check for skipped EOS tokens.
    mean_top_logprob = np.mean(highest_logprobs)
    top_logprob_std = np.std(highest_logprobs)
    if top_logprob_std:
        for eos_logprob in eos_expected:
            if eos_logprob < mean_top_logprob:
                continue
            zscore = (eos_logprob - mean_top_logprob) / top_logprob_std
            if zscore >= 3:
                return False, f"EOS token skipped [{eos_logprob=} {zscore=}]"
    if len(eos_expected) >= 7:
        return False, f"EOS token expected {len(eos_expected)} times before end of stream"

    # Check for token repetition.
    repeats = find_repeated_subsequences(output_tokens)
    for subseq, count in repeats.items():
        repeated_text = TOKENIZER.decode(subseq)

        # How many times was this subsequence repeated?
        if count >= 10:
            return False, f"Found phrase repeated {count} times: {repeated_text}"

        # How much of the total output does the repeated text account for?
        ratio = (len(subseq) * count) / len(output_tokens)
        if ratio >= 0.75:
            return False, f"Found phrase repeated {count} times accounting for {ratio} of overall output: {repeated_text}"

    average_score = total_score / idxs
    passes = average_score >= LOGPROB_FAILURE_THRESHOLD
    perfect_avg = perfect_tokens / idxs
    message = (
        f"Successfully verified logprob for {len(request.output_sequence)} outputs with {average_score} and {perfect_avg}% perfect tokens"
        if passes
        else f"Low average logprob score: {average_score}"
    )
    if passes and perfect_avg >= (1 - min(request.request_params.temperature, 0.6)):
        message = f"Overfitted response tokens. {perfect_avg}% perfect"
        passes = False

    return (
        passes,
        message,
    )

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
        # Check the weight values via powv.
        result, message = verify_powv(request, input_tokens)
        return_value = {
            "verified": False,
            "powv_pass": result,
            "powv_message": message,
            "logprob_fast_pass": False,
            "logprob_fast_message": None,
        }
        if not result:
            return_value.update({"verified": False})
            return return_value

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
