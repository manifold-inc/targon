from collections import defaultdict
import random
import math
import os
import asyncio
import traceback
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
            ]
            + random.sample(indices, min(len(indices), 3))
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
            message = f"Token output at index {idx} [{TOKENIZER.decode([request.output_sequence[idx].token_id])}] not found in top {top_logprobs} logprobs: {[TOKENIZER.decode([token]) for token in top_tokens]}"
            return False, message
    return (
        True,
        f"Successfully verified {len(indices_to_check)} random logprobs: {indices_to_check}",
    )


def verify_logprobs(
    request: VerificationRequest, input_text: str, input_tokens: List[int]
) -> Optional[Tuple[bool, str, str]]:
    """
    Compare the produced logprob values against the ground truth, or at least
    the ground truth according to this particular GPU/software pairing.
    """

    # Set up sampling parameters for the "fast" check, which just compares input logprobs against output logprobs.
    top_logprobs = int(request.request_params.temperature * 10) + 3
    sampling_params = SamplingParams(
        temperature=request.request_params.temperature,
        seed=request.request_params.seed,
        max_tokens=1,
        logprobs=top_logprobs,
        prompt_logprobs=top_logprobs,
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
    eot_token_id = TOKENIZER.get_vocab().get("<|eot_id|>", -1)  # type: ignore
    output_tokens = [item.token_id for item in request.output_sequence]
    really_low_prob = 0
    not_first = 0
    for idx in range(idxs):
        item = request.output_sequence[idx]
        expected_logprob = output.prompt_logprobs[idx + len(input_tokens)]
        assert expected_logprob is not None
        eos_logprob = expected_logprob.get(eos_token_id)
        eot_logprob = expected_logprob.get(eot_token_id)
        if (
            not eos_logprob
            and eot_logprob
            or (
                eos_logprob
                and eot_logprob
                and eot_logprob.rank != None
                and eos_logprob.rank != None
                and eot_logprob.rank < eos_logprob.rank
            )
        ):
            eos_logprob = eot_logprob
        expected_logprob = expected_logprob.get(item.token_id)
        if eos_logprob and (
            not expected_logprob
            or (
                eos_logprob
                and expected_logprob.rank != None
                and eos_logprob.rank != None
                and eos_logprob.rank < expected_logprob.rank
                and expected_logprob.rank > 10
            )
        ):
            return False, f"Expected EOS/EOT token at index {idx}", "SKIPPED_EOS_EOT"
        if expected_logprob is None:
            continue
        rank = expected_logprob.rank
        assert rank != None
        if rank >= 75:
            return (
                False,
                f"Found extraordinarily improbable token '{TOKENIZER.decode([item.token_id])}' at index {idx}: {rank=}",
                "UNLIKELY_TOKEN",
            )
        elif rank >= 25:
            really_low_prob += 1
        elif rank > top_logprobs:
            continue
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
        if (
            rank == 1
            and request.request_params.temperature >= 0.9
            and produced_logprob != 0
        ):
            score = 1.0

        total_score += score

    # Check if miner produced non-top ranking tokens more than top-ranking tokens.
    ratio = not_first / len(output_tokens)
    if ratio >= 0.5:
        return (
            False,
            f"{not_first} of {len(output_tokens)} [{ratio=}] tokens were not rank 1.",
            "UNLIKELY_TOKENS",
        )

    # Check if miner prematurely stopped generating, meaning the single output token generated
    # from the "throwaway" above was NOT an EOS/EOT token.
    if eos_token_id > 0 or eot_token_id > 0:
        if len(output_tokens) < request.request_params.max_tokens:
            last_token_probs = []
            if output:
                last_token_probs = output.outputs[0]
                last_token_probs = (
                    last_token_probs.logprobs[0]
                    if last_token_probs and last_token_probs.logprobs
                    else []
                )
            if (
                eos_token_id not in last_token_probs
                and eot_token_id not in last_token_probs
                and len(last_token_probs) != 0
            ):
                return (
                    False,
                    "Premature end of generation, EOS/EOT unlikely after last token.",
                    "EARLY_END",
                )

    # Calculate average score.
    average_score = round(total_score / idxs, 5)
    passes = average_score >= LOGPROB_FAILURE_THRESHOLD
    perfect_avg = round(perfect_tokens / idxs, 5)
    if passes and perfect_avg >= (
        1 - min(request.request_params.temperature * 0.5, 0.6)
    ):
        return False, f"Overfitted response tokens. {perfect_avg}% perfect", "OVERFIT"
    if really_low_prob >= 5:
        return (
            False,
            f"Found {really_low_prob} highly improbable tokens.",
            "UNLIKELY_TOKEN",
        )

    return True, "", ""


@app.post("/verify")
async def verify(request: VerificationRequest) -> Dict:
    """Verify a miner's output."""

    # If the miner didn't return any outputs, fail.
    if len(request.output_sequence) < 3:
        return {
            "verified": False,
            "error": "Output sequence too short!",
            "cause": "TOO_SHORT",
        }
    if (
        request.request_params.max_tokens
        and len(request.output_sequence) > request.request_params.max_tokens
    ):
        return {
            "verified": False,
            "error": "Too many tokens produced!",
            "cause": "TOO_LONG",
        }
    if request.model != MODEL_NAME:
        return {
            "verified": False,
            "error": "Unable to verify model={request.model}, since we are using {MODEL_NAME}",
            "cause": "INTERNAL_ERROR",
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
            "error": None,
        }

        # Logprob checks.
        res = verify_logprobs(request, str(input_text), input_tokens)
        if res is None:
            return {"error": "Failed to check log probs", "cause": "INTERNAL_ERROR"}
        result, message, cause = res
        return_value.update(
            {
                "verified": result,
                "cause": cause,
                "error": message,
            }
        )
        if not result:
            return return_value

        # Random logprob check.
        if request.request_params.temperature > 0.75:
            return {"verified": True}

        res = verify_logprobs_random(request, str(input_text))
        if res is None:
            return {
                "error": "Failed to check log probs",
                "cause": "INTERNAL_ERROR",
            }
        result, message = res
        return_value.update(
            {
                "verified": result,
                "cause": "LOGPROB_RANDOM",
                "error": message,
            }
        )
        if not result:
            return return_value

        return {"verified": True}


@app.get("/endpoints")
def endpoints():
    return ENDPOINTS


@app.get("/")
def ping():
    return "", 200
