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
    print("No model name provided, exiting.")
    exit()
# Constants.
LOGPROB_LOG_THRESHOLD = 0.65
LOGPROB_FAILURE_THRESHOLD = 0.75
TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL", 1))
MODEL_WRAPPER = LLM(
    model=MODEL_NAME,
    enforce_eager=True,
    gpu_memory_utilization=0.9,
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
    # Optional parameters that depend on endpoint
    messages: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None

    # Core parameters
    temperature: Optional[float] = None  # range 0.0-2.0
    top_p: Optional[float] = None  # Default None, range 0.0-1.0
    max_tokens: Optional[int] = None  # Optional, must be 1+
    stop: Optional[List[str]] = None  # Optional, defaults to None
    seed: Optional[int] = None  # Optional

    # Additional optional parameters
    top_k: Optional[int] = None  # Default None, range 0+
    frequency_penalty: Optional[float] = None  # Default None, range -2.0-2.0
    presence_penalty: Optional[float] = None  # Default None, range -2.0-2.0
    repetition_penalty: Optional[float] = None  # Default None, range 0.0-2.0
    min_p: Optional[float] = None  # Default None, range 0.0-1.0
    top_a: Optional[float] = None  # Default None, range 0.0-1.0


class OutputItem(BaseModel):
    text: str
    logprob: float
    token_id: int


class RequestType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class VerificationRequest(BaseModel):
    request_type: str
    model: str = MODEL_NAME
    request_params: RequestParams
    raw_chunks: List[Dict]


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
                        messages=req.messages,
                        sampling_params=SamplingParams(
                            **req.sampling_params.model_dump()
                        ),
                        use_tqdm=False,  # type: ignore
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
    temperature: float,
    seed: int,
    input_text: str,
    output_sequence: List[OutputItem]
) -> Optional[Tuple[bool, str]]:
    """
    Generate a handful of random outputs to ensure the logprobs weren't generated after the fact.
    """
    indices = list(range(1, len(output_sequence) - 1))
    indices_to_check = list(
        sorted(
            [
                0,  # always check first token
                len(output_sequence) - 1,  # always check last token
            ]
            + random.sample(indices, min(len(indices), 3))
        )
    )

    # Generate a single token at each index, comparing logprobs.
    top_logprobs = int(temperature * 10) + 3
    sampling_params = SamplingParams(
        temperature=temperature,
        seed=seed,
        max_tokens=1,
        logprobs=top_logprobs,
    )
    for idx in indices_to_check:
        full_text = input_text + "".join(
            [item.text for item in output_sequence[0:idx]]
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
        if output_sequence[idx].token_id not in top_tokens:
            message = f"Token output at index {idx} [{TOKENIZER.decode([output_sequence[idx].token_id])}] not found in top {top_logprobs} logprobs: {[TOKENIZER.decode([token]) for token in top_tokens]}"
            return False, message
    return (
        True,
        f"Successfully verified {len(indices_to_check)} random logprobs: {indices_to_check}",
    )


def verify_logprobs(
    temperature: float,
    seed: int,
    max_tokens: Optional[int],
    input_text: str, 
    input_tokens: List[int],
    output_sequence: List[OutputItem]
) -> Optional[Tuple[bool, str, str]]:
    """
    Compare the produced logprob values against the ground truth, or at least
    the ground truth according to this particular GPU/software pairing.
    """

    # Set up sampling parameters for the "fast" check, which just compares input logprobs against output logprobs.
    top_logprobs = int(temperature * 10) + 6
    sampling_params = SamplingParams(
        temperature=temperature,
        seed=seed,
        max_tokens=1,
        logprobs=top_logprobs,
        prompt_logprobs=top_logprobs,
    )

    # Generate output for a single token, which will return input logprobs based on prompt_logprobs=1
    output = None
    for _ in range(5):
        full_text = input_text + "".join(
            [item.text for item in output_sequence]
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
        len(output_sequence) - 1,
    )
    perfect_tokens = 0
    eos_token_id = getattr(TOKENIZER, "eos_token_id", -1)
    eot_token_id = TOKENIZER.get_vocab().get("<|eot_id|>", -1)  # type: ignore
    output_tokens = [item.token_id for item in output_sequence]
    really_low_prob = 0
    not_first = 0
    for idx in range(idxs):
        item = output_sequence[idx]
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
            return (
                False,
                f"Expected EOS/EOT token at index {idx}",
                "SKIPPED_EOS_EOT",
            )
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
            and temperature >= 0.9
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
        if max_tokens and len(output_tokens) < max_tokens:
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
        1 - min(temperature * 0.5, 0.6)
    ):
        return False, f"Overfitted response tokens. {perfect_avg}% perfect", "OVERFIT"
    if really_low_prob >= 5:
        return (
            False,
            f"Found {really_low_prob} highly improbable tokens.",
            "UNLIKELY_TOKEN",
        )

    return True, "", ""


def verify_usage(
    input_tokens_length: int,
    usage: Usage,
    output_sequence_length: int
) -> Optional[Tuple[bool, str, str]]:
    """Verify the usage information in the response."""
    # Get actual token counts
    actual_completion_tokens = output_sequence_length
    actual_total_tokens = input_tokens_length + actual_completion_tokens

    # Verify token counts
    if usage.completion_tokens != actual_completion_tokens:
        return (
            False,
            f"Reported completion tokens ({usage.completion_tokens}) does not match actual count ({actual_completion_tokens})",
            "INCORRECT_USAGE_DATA",
        )

    if usage.prompt_tokens != input_tokens_length:
        return (
            False,
            f"Reported prompt tokens ({usage.prompt_tokens}) does not match actual count ({input_tokens_length})",
            "INCORRECT_USAGE_DATA",
        )

    if usage.total_tokens != actual_total_tokens:
        return (
            False,
            f"Reported total tokens ({usage.total_tokens}) does not match actual count ({actual_total_tokens})",
            "INCORRECT_USAGE_DATA",
        )

    return True, "", ""


def parse_chunk(chunk: Dict, request_type: str) -> Optional[OutputItem]:
    """Parse a raw chunk into an OutputItem with token info"""
    try:
        choice = chunk.get('choices', [])[0]
        
        # Initialize defaults
        token_id = -1
        logprob = -100
        
        if request_type == "CHAT":
            if choice.get('delta') is None:
                return None
                
            # Check for empty content
            content = choice.get('delta', {}).get('content')
            if content == "" or content is None:
                return None
                
            choiceprobs = choice.get('logprobs')
            if choiceprobs is not None:
                if choiceprobs.get('content'):
                    logprob = choiceprobs['content'][0]['logprob']
                    token = choiceprobs['content'][0]['token']
                    if token is None:
                        return None
                    if not token.startswith("token_id:"):
                        return None
                    token_parts = token.split(":")
                    if len(token_parts) > 1:
                        token_id = int(token_parts[1])
            
            return OutputItem(
                text=content or "",
                token_id=token_id,
                logprob=logprob
            )
                        
        elif request_type == "COMPLETION":
            text = choice.get('text')
            if text is None:
                return None
                
            # Check logprobs exist
            if choice.get('logprobs') is None:
                return None
                
            if choice['logprobs'].get('token_logprobs'):
                logprob = choice['logprobs']['token_logprobs'][0]
                
            if (choice['logprobs'].get('tokens') is not None
                and len(choice['logprobs']['tokens']) > 0):
                token = choice['logprobs']['tokens'][0]
                if token is None:
                    return None
                if not token.startswith("token_id:"):
                    return None
                token_parts = token.split(":")
                if len(token_parts) > 1:
                    token_id = int(token_parts[1])
            
            return OutputItem(
                text=text or "",
                token_id=token_id,
                logprob=logprob
            )
        
        return None
        
    except Exception as e:
        print(f"Failed to parse chunk: {e}")
        return None

@app.post("/verify")
async def verify(request: VerificationRequest) -> Dict:
    """Verify a miner's output."""
    
    # Parse raw chunks into OutputItems
    output_sequence = []
    for chunk in request.raw_chunks:
        if parsed := parse_chunk(chunk, request.request_type):
            output_sequence.append(parsed)

    # If we couldn't parse enough tokens, fail
    if len(output_sequence) < 3:
        return {
            "verified": False,
            "error": "Output sequence too short!",
            "cause": "TOO_SHORT",
        }
    
    # Check max tokens
    if (
        request.request_params.max_tokens
        and len(output_sequence) > request.request_params.max_tokens
    ):
        return {
            "verified": False,
            "error": f"Too many tokens produced: {request.request_params.max_tokens} < {len(output_sequence)}",
            "cause": "TOO_LONG",
        }
        
    if request.model != MODEL_NAME:
        return {
            "verified": False,
            "error": f"Unable to verify model={request.model}, since we are using {MODEL_NAME}",
            "cause": "INTERNAL_ERROR",
        }

    final_chunk = request.raw_chunks[-1]
    usage_data = final_chunk.get('usage')
    if not usage_data:
        return {
            "verified": False,
            "error": "No usage information in final chunk",
            "cause": "NO_USAGE"
        }
    
    try:
        usage = Usage(**usage_data)
    except Exception as e:
        return {
            "verified": False,
            "error": f"Invalid usage data: {str(e)}",
            "cause": "INVALID_USAGE"
        }

    # Tokenize the input sequence
    input_text = (
        request.request_params.prompt
        if request.request_type == RequestType.COMPLETION.value
        else TOKENIZER.apply_chat_template(
            request.request_params.messages,  # type: ignore
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
    )
    assert isinstance(input_text, str)
    if hasattr(TOKENIZER, "bos_token"):
        if input_text.startswith(TOKENIZER.bos_token):  # type: ignore
            input_text = input_text[len(TOKENIZER.bos_token):]  # type: ignore
    input_tokens = TOKENIZER(input_text).input_ids

    # Verify!
    async with LOCK:
        return_value = {
            "verified": False,
            "error": None,
        }

        # Verify usage information
        res = verify_usage(len(input_tokens), usage, len(output_sequence))
        if res is None:
            return {"error": "Failed to check usage", "cause": "INTERNAL_ERROR"}
        result, message, cause = res
        return_value.update({
            "verified": result,
            "cause": cause,
            "error": message,
        })
        if not result:
            return return_value

        # Logprob checks
        res = verify_logprobs(request.request_params.temperature, request.request_params.seed, request.request_params.max_tokens, str(input_text), input_tokens, output_sequence)
        if res is None:
            return {"error": "Failed to check log probs", "cause": "INTERNAL_ERROR"}
        result, message, cause = res
        return_value.update({
            "verified": result,
            "cause": cause,
            "error": message,
        })
        if not result:
            return return_value

        # Random logprob check
        if request.request_params.temperature > 0.75:
            return {"verified": True}

        res = verify_logprobs_random(request.request_params.temperature, request.request_params.seed, str(input_text), output_sequence)
        if res is None:
            return {
                "error": "Failed to check log probs",
                "cause": "INTERNAL_ERROR",
            }
        result, message = res
        return_value.update({
            "verified": result,
            "cause": "LOGPROB_RANDOM",
            "error": message,
        })
        if not result:
            return return_value

        return {"verified": True}


@app.get("/endpoints")
def endpoints():
    return ENDPOINTS


@app.get("/")
def ping():
    return "", 200
