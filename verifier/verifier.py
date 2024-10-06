import random
import math
import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams

# Load the model.
MODEL_NAME = os.getenv("MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")
GPU_MEMORY_UTIL = float(os.getenv("GPU_MEMORY_UTIL", 1))
# Constants.
LOGPROB_LOG_THRESHOLD = 0.65
LOGPROB_FAILURE_THRESHOLD = 0.85
TOP_LOGPROBS = 7
MODEL_WRAPPER = LLM(
    model=MODEL_NAME,
    enforce_eager=True,
    gpu_memory_utilization=0.4,
    max_model_len=4096,
)
TOKENIZER = MODEL_WRAPPER.get_tokenizer()
MODEL = MODEL_WRAPPER.llm_engine.model_executor.driver_worker.model_runner.model  # type: ignore
MODEL_NUM_PARAMS = sum(1 for _ in MODEL.parameters())

# Lock to ensure atomicity.
LOCK = asyncio.Lock()


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
def generate_question(req: GenerateRequest):
    output = (
        MODEL_WRAPPER.chat(
            messages=req.messages, sampling_params=SamplingParams(**req.sampling_params.model_dump()), use_tqdm=False  # type: ignore
        )[0]
        .outputs[0]
        .text
    )
    return {"text": output}


def verify_powv(
    request: VerificationRequest, input_tokens: List[int]
) -> Tuple[bool, str]:
    """
    Check the returned `powv` values against the ground truth.
    """
    input_sum = sum(input_tokens)

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
        prompt_logprobs=5,
    )

    # Generate output for a single token, which will return input logprobs based on prompt_logprobs=1
    full_text = input_text + "".join([item.text for item in request.output_sequence])
    output = MODEL_WRAPPER.generate([full_text], sampling_params, use_tqdm=False)[0]
    assert output.prompt_logprobs is not None

    # The actual logprobs should be *very* close, but typically not 100% because of GPU/driver/etc. differences.
    total_score = 0.0
    idxs = min(
        len(output.prompt_logprobs) - len(input_tokens) - 3,
        len(request.output_sequence) - 1,
    )
    for idx in range(idxs):
        item = request.output_sequence[idx]
        expected_logprob = output.prompt_logprobs[idx + len(input_tokens)]
        assert expected_logprob is not None
        expected_logprob = expected_logprob.get(item.token_id)
        if expected_logprob is not None:
            expected_logprob = expected_logprob.logprob
        else:
            expected_logprob = 0
        produced_logprob = item.logprob
        delta = abs(produced_logprob - expected_logprob)
        score = (1.0 - delta) ** 2

        # To accomodate architectural difference and such, we'll give a perfect score if >= 0.9
        if score >= 0.9:
            score = 1.0

        total_score += score

    average_score = total_score / idxs
    if average_score < LOGPROB_FAILURE_THRESHOLD:
        message = f"Low average logprob score: {average_score}"
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
            "logprob_random_pass": False,
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
            return return_value

        # Slow(ish) random logprob spotchecks.
        if request.request_params.temperature < 0.5:
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
        else:
            return_value.update(
                {"logprob_random_message": "Temperature too high to check"}
            )

        return_value.update({"verified": True})
        return return_value

@app.get('/')
def ping():
    return "", 200
