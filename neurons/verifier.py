import random
import math
import os
import asyncio
from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams
from fastapi import FastAPI

app = FastAPI()

# Load the model.
MODEL_NAME = os.getenv("MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")
MODEL_WRAPPER = LLM(model=MODEL_NAME, enforce_eager=True)
TOKENIZER = MODEL_WRAPPER.get_tokenizer()
MODEL = MODEL_WRAPPER.llm_engine.model_executor.driver_worker.model_runner.model
MODEL_NUM_PARAMS = sum(1 for _ in MODEL.parameters())

# Constants.
LOGPROB_LOG_THRESHOLD = 0.65
LOGPROB_FAILURE_THRESHOLD = 0.85
TOP_LOGPROBS = 7

# Lock to ensure atomicity.
LOCK = asyncio.Lock()


class RequestParams(BaseModel):
    messages: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = 0.0
    seed: Optional[int] = 42
    max_tokens: Optional[int] = None


class OutputItem(BaseModel):
    text: str
    logprob: float
    powv: int
    token_id: int


class RequestType(Enum):
    chat = "chat"
    completion = "completion"


class VerificationRequest(BaseModel):
    request_type: str
    model: str = MODEL_NAME
    request_params: RequestParams
    output_sequence: List[OutputItem]


def verify_powv(
    request: VerificationRequest, input_tokens: List[int]
) -> Tuple[bool, str]:
    """
    Check the returned `powv` values against the ground truth.
    """
    input_sum = sum(input_tokens)

    # Iterate through output sequence, checking powv values.
    output_sum = 0
    for idx in range(len(request.output_sequence)):
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

    # All clear.
    print("Successfully verified powv!")
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
                random.choice(list(range(1, len(request.output_sequence) - 1))), # random offset
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
        output = MODEL_WRAPPER.generate([full_text], sampling_params)[0].outputs[0]

        # The miner's output token should be in the logprobs...
        top_tokens = []
        for lp in output.logprobs:
            top_tokens += list(lp.keys())
        if request.output_sequence[idx].token_id not in top_tokens:
            message = f"Token output at index {idx} [{request.output_sequence[idx]}] not found in top {TOP_LOGPROBS} top logprobs: {top_tokens}"
            print(message)
            return False, message
    return (
        True,
        f"Successfully verified {len(indices_to_check)} random logprobs: {indices_to_check}",
    )


def verify_logprobs_fast(
    request: VerificationRequest, input_text: str, input_tokens: List[int]
) -> Tuple[bool, str]:
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
        prompt_logprobs=1,
    )

    # Generate output for a single token, which will return input logprobs based on prompt_logprobs=1
    full_text = input_text + "".join([item.text for item in request.output_sequence])
    output = MODEL_WRAPPER.generate([full_text], sampling_params)[0]

    # The actual logprobs should be *very* close, but typically not 100% because of GPU/driver/etc. differences.
    total_score = 0.0
    for idx in range(len(request.output_sequence)):
        item = request.output_sequence[idx]
        expected_logprob = output.prompt_logprobs[idx + len(input_tokens)][
            item.token_id
        ].logprob
        # print(F"EXPECTED: {expected_logprob}")
        produced_logprob = item.logprob
        delta = abs(produced_logprob - expected_logprob)
        score = (1.0 - delta) ** 2

        # To accomodate architectural difference and such, we'll give a perfect score if >= 0.9
        if score >= 0.9:
            score = 1.0

        # Log low scores.
        elif score <= LOGPROB_LOG_THRESHOLD:
            print(
                f"Low logprob score for output sequence {idx} {expected_logprob=} {produced_logprob=}"
            )

        total_score += score

    average_score = total_score / len(request.output_sequence)
    if average_score < LOGPROB_FAILURE_THRESHOLD:
        message = f"Low average logprob score: {average_score}"
        print(message)
        return False, message
    return (
        True,
        f"Successfully verified logprob for {len(request.output_sequence)} outputs with {average_score=}",
    )


@app.post("/verify")
async def verify(request: VerificationRequest) -> Dict:
    """Verify a miner's output."""

    # If the miner didn't return any outputs, fail.
    if len(request.output_sequence) < 3:
        return {
            "status": "fail",
            "reason": "Output sequence too short!",
        }
    if (
        request.request_params.max_tokens
        and len(request.output_sequence) > request.request_params.max_tokens
    ):
        return {
            "status": "fail",
            "reason": "Too many tokens produced!",
        }
    if request.model != MODEL_NAME:
        return {
            "status": "fail",
            "reason": "Unable to verify model={request.model}, since we are using {MODEL_NAME}",
        }

    # Tokenize the input sequence.
    input_text = (
        request.request_params.prompt
        if request.request_type == RequestType.completion.value
        else TOKENIZER.apply_chat_template(
            request.request_params.messages, tokenize=False
        )
    )
    input_tokens = TOKENIZER(input_text).input_ids

    # Verify!
    async with LOCK:

        # Check the weight values via powv.
        result, message = verify_powv(request, input_tokens)
        return_value = {
            "verified": True,
            "powv_pass": result,
            "powv_message": message,
            "logprob_fast_pass": True,
            "logprob_fast_message": None,
            "logprob_random_pass": True,
            "logprob_random_message": None,
        }
        if not result:
            return_value.update({"verified": False})
            return return_value

        # Fast(ish) logprob check, based on input sequence processing.
        result, message = verify_logprobs_fast(request, str(input_text), input_tokens)
        return_value.update(
            {
                "logprob_fast_pass": result,
                "logprob_fast_message": message,
            }
        )
        if not result:
            return_value.update({"verified": False})
            return return_value

        # Slow(ish) random logprob spotchecks.
        result, message = verify_logprobs_random(request, str(input_text))
        return_value.update(
            {
                "logprob_random_pass": result,
                "logprob_random_message": message,
            }
        )
        if not result:
            return_value.update({"verified": False})

        return return_value
