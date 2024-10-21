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
    for idx in range(idxs):
        item = request.output_sequence[idx]
        expected_logprob = output.prompt_logprobs[idx + len(input_tokens)]
        assert expected_logprob is not None
        expected_logprob = expected_logprob.get(item.token_id)
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

def count_repeating_sequences(token_ids: List[int], min_repeat_length: int=5) -> int:
    """
    Counts the number of distinct repeating sequences in a list of token IDs.

    This function analyzes a list of token IDs to identify and count unique subsequences that repeat, ensuring that only distinct sequences are considered. It filters out any subsequences that are extensions of shorter repeating sequences.

    Args:
        token_ids (List[int]): A list of integer token IDs to analyze.
        min_repeat_length (int, optional): The minimum length of subsequences to consider for repetition. Defaults to 5.

    Returns:
        int: The count of distinct repeating sequences found in the input list.
    """
    repeating_sequences = {}
    sequence_length = len(token_ids)

    # Iterate through the sequence to find all subsequences of length >= 5
    for length in range(min_repeat_length, sequence_length + 1):
        i = 0
        while i <= sequence_length - length:
            subsequence = tuple(token_ids[i : i + length])
            found = False
            for j in range(i + length, sequence_length - length + 1):
                if tuple(token_ids[j : j + length]) == subsequence:
                    if subsequence not in repeating_sequences:
                        repeating_sequences[subsequence] = (
                            2  # Initial occurrence + 1 repeat
                        )
                    else:
                        repeating_sequences[subsequence] += 1
                    found = True
                    break  # Stop after finding the first repeat
            if found:
                # Skip the entire subsequence after a match is found
                i += length
            else:
                i += 1

    # Filter out any subsequences that are extended versions of shorter ones
    filtered_sequences = {}
    for seq, count in repeating_sequences.items():
        if not any(seq[:i] in repeating_sequences for i in range(1, len(seq))):
            filtered_sequences[seq] = count
            
    # Return the count of distinct repeating sequences
    return len(filtered_sequences)  


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
            "powv_pass": False,
            "powv_message": "Powv not checked",
            "logprob_fast_pass": False,
            "logprob_fast_message": None,
            "repeated_tokens": False,
        }
        # Check for repeating sequences of token ids.
        max_repeats = 0
        repeating_token_sequences = count_repeating_sequences(input_tokens)
        if repeating_token_sequences > max_repeats:
            return_value["repeated_tokens"] = True
            return_value.update({"verified": False})
            return return_value
        
        # Check the weight values via powv.
        result, message = verify_powv(request, input_tokens)
        return_value.update( { "powv_pass": result, "powv_message": message })
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
