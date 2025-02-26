import random
import traceback
import os
import asyncio
import json
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import sglang

# Load the model.
MODEL_NAME = os.getenv("MODEL", None)
if MODEL_NAME is None:
    print("No model name provided, exiting.")
    exit()
# Constants.
LOGPROB_LOG_THRESHOLD = 0.65
LOGPROB_FAILURE_THRESHOLD = 0.75
TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL", 1))
PIPELINE_PARALLEL = int(os.getenv("PIPELINE_PARALLEL", 1))
CONTEXT_LENGTH = os.getenv("CONTEXT_LENGTH", None)
if CONTEXT_LENGTH != None:
    CONTEXT_LENGTH = int(CONTEXT_LENGTH)
MODEL_WRAPPER = sglang.Engine(
    model_path=MODEL_NAME,
    tp_size=TENSOR_PARALLEL,
    trust_remote_code=True,
    context_length=CONTEXT_LENGTH,
)
# MODEL_WRAPPER = AsyncLLMEngine.from_engine_args(
#    AsyncEngineArgs(
#        model=MODEL_NAME,
#        gpu_memory_utilization=0.95,
#        tensor_parallel_size=TENSOR_PARALLEL,
#        trust_remote_code=True,
#        max_num_batched_tokens=2048,
#        max_num_seqs=32,
#    )
# )

# Lock to ensure atomicity.
LOCK = asyncio.Lock()


class RequestParams(BaseModel):
    # Optional parameters that depend on endpoint
    messages: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    max_tokens: int
    temperature: float
    seed: int

    # Core parameters
    top_p: Optional[float] = None  # Default None, range 0.0-1.0
    stop: Optional[List[str]] = None  # Optional, defaults to None

    # Additional optional parameters
    top_k: Optional[int] = None  # Default None, range 0+
    frequency_penalty: Optional[float] = None  # Default None, range -2.0-2.0
    presence_penalty: Optional[float] = None  # Default None, range -2.0-2.0
    repetition_penalty: Optional[float] = None  # Default None, range 0.0-2.0
    min_p: Optional[float] = None  # Default None, range 0.0-1.0
    top_a: Optional[float] = None  # Default None, range 0.0-1.0

    # Stream parameters
    stream: Optional[bool] = None
    stream_options: Optional[Dict] = None
    logprobs: Optional[bool] = None

    # Tool parameters
    tools: Optional[List[Dict[str, Any]]] = None  # Optional list of tool definitions
    tool_choice: Optional[
        Union[str, Dict[str, Any]]
    ] = None  # Optional tool choice specification


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
    async with LOCK:
        try:
            prompt = ""
            for message in req.messages:
                prompt += (
                    message.get("role", "") + ": " + message.get("content", "") + "\n"
                )
            prompt += "\nResponse: "
            output = await MODEL_WRAPPER.async_generate(
                prompt=prompt,
                sampling_params=req.sampling_params.model_dump(),
            )
            assert type(output) is Dict
            return {"text": output["text"]}
        except Exception:
            return {"text": None}


async def verify_logprobs_random(
    temperature: float,
    input_text: str,
    output_sequence: List[OutputItem],
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> Optional[Tuple[bool, str]]:
    """
    Generate a handful of random outputs to ensure the logprobs weren't generated after the fact.
    """
    TOKENIZER = MODEL_WRAPPER.tokenizer_manager.tokenizer
    assert TOKENIZER is not None
    if len(output_sequence) < 4:
        return True, "Skipped random verification due to insufficient tokens"

    indices = list(range(1, len(output_sequence) - 1))
    indices_to_check = list(
        sorted(
            [
                1,  # always check first token
                len(output_sequence) - 1,  # always check last token
            ]
            + random.sample(indices, min(len(indices), 3))
        )
    )

    # Generate a single token at each index, comparing logprobs
    top_logprobs = int(temperature * 10) + 3
    sampling_params_dict = {
        "temperature": temperature,
        "max_new_tokens": 1,
    }
    if tools:
        sampling_params_dict["tools"] = tools
        sampling_params_dict["tool_choice"] = tool_choice

    for idx in indices_to_check:
        full_text = input_text + "".join([item.text for item in output_sequence[0:idx]])

        output = await MODEL_WRAPPER.async_generate(
            full_text,
            sampling_params_dict,
            logprob_start_len=0,
            top_logprobs_num=15,
            return_logprob=True,
            stream=False,
        )
        assert isinstance(output, dict)

        top_tokens = []
        if output["meta_info"]["output_top_logprobs"] is None:
            continue

        for lp in output["meta_info"]["output_top_logprobs"][0]:
            top_tokens.append(lp[1])

        if output_sequence[idx].token_id not in top_tokens:
            message = f"Token output at index {idx} [{TOKENIZER.decode([output_sequence[idx].token_id])}] not found in top {top_logprobs} logprobs: {[TOKENIZER.decode([token]) for token in top_tokens]}"
            return False, message

    success_msg = f"Successfully verified {len(indices_to_check)} random logprobs: {indices_to_check}"
    return True, success_msg


async def verify_logprobs(
    temperature: float,
    max_tokens: int,
    input_text: str,
    input_tokens: List[int],
    output_sequence: List[OutputItem],
) -> Optional[Tuple[bool, str, str]]:
    """
    Compare the produced logprob values against the ground truth, or at least
    the ground truth according to this particular GPU/software pairing.
    """
    TOKENIZER = MODEL_WRAPPER.tokenizer_manager.tokenizer
    assert TOKENIZER is not None
    # Generate output for a single token
    output = None
    full_text = input_text + "".join([item.text for item in output_sequence])
    output = await MODEL_WRAPPER.async_generate(
        prompt=full_text,
        sampling_params={
            "temperature": temperature,
            "max_new_tokens": 1,
        },
        logprob_start_len=0,
        top_logprobs_num=15,
        return_logprob=True,
        stream=False,
    )
    assert isinstance(output, dict)

    # The actual logprobs should be very close but not 100% due to GPU/driver differences
    idxs = min(
        output["meta_info"]["prompt_tokens"] - len(input_tokens) - 3,
        len(output_sequence) - 1,
    )
    perfect_tokens = 0
    eos_token_id = getattr(TOKENIZER, "eos_token_id", -1)
    eot_token_id = TOKENIZER.get_vocab().get("<|eot_id|>", -1)  # type: ignore
    really_low_prob = 0
    meta = output["meta_info"]
    for idx in range(idxs):
        # Get top logprobs
        expected_logprob_list = meta["input_top_logprobs"][idx + len(input_tokens)]

        # Get true logprob for token
        real_logprob, token_id, _ = meta["input_token_logprobs"][
            idx + len(input_tokens)
        ]

        # Arrage top log probs in vllm style
        expected_logprob_set = {
            token: logprob for logprob, token, _ in expected_logprob_list
        }

        # Check if eos / eot tokens exist in top probs
        eos_logprob = expected_logprob_set.get(eos_token_id)
        eot_logprob = expected_logprob_set.get(eot_token_id)

        # Just check one variable as eot / eos
        if (eos_logprob is None and eot_logprob is not None) or (
            eot_logprob is not None
            and eos_logprob is not None
            and eot_logprob < eos_logprob
        ):
            eos_logprob = eot_logprob

        # Get text of current token
        token_text = TOKENIZER.decode([token_id])

        # Skipped eos/eot check
        if eos_logprob is not None and (
            real_logprob is None
            or (
                eos_logprob is not None
                and real_logprob is not None
                and eos_logprob > real_logprob
                and real_logprob < eos_logprob * 2
            )
        ):
            error_msg = f"Expected EOS/EOT token at index {idx}, {token_text=}, {expected_logprob_set=}, {eos_logprob=}"
            return False, error_msg, "SKIPPED_EOS_EOT"

        # Shouldnt happen
        if real_logprob is None:
            continue

        # Extreemly unlikely token
        if real_logprob <= -13:
            error_msg = f"Found extraordinarily improbable token '{token_text}' at index {idx}: {real_logprob=}"
            return False, error_msg, "UNLIKELY_TOKEN"

        # Unlikely token
        elif real_logprob <= -8:
            really_low_prob += 1

        # Exact token
        if real_logprob > 0.0001:
            perfect_tokens += 1

    # Check if miner prematurely stopped generating
    if eos_token_id > 0 or eot_token_id > 0:
        if len(output_sequence) < max_tokens:
            expected_logprob_list = meta["output_top_logprobs"][0]
            last_token_probs = [token for _, token, _ in expected_logprob_list]
            if (
                eos_token_id not in last_token_probs
                and eot_token_id not in last_token_probs
                and len(last_token_probs) != 0
            ):
                error_msg = (
                    "Premature end of generation, EOS/EOT unlikely after last token"
                )
                return False, error_msg, "EARLY_END"

    # general check for distilled models (will be almost all perfect)
    perfect_avg = round(perfect_tokens / idxs, 5)
    if perfect_avg >= 1:
        error_msg = f"Overfitted response tokens. {perfect_avg}% perfect"
        return False, error_msg, "OVERFIT"

    # Check for too many unlikely tokens
    if really_low_prob >= 5:
        error_msg = f"Found {really_low_prob} highly improbable tokens"
        return False, error_msg, "UNLIKELY_TOKEN"

    return True, "", ""


def verify_usage(
    input_tokens: int,
    output_sequence: int,
    usage: Usage,
) -> Optional[Tuple[bool, str, str]]:
    """Verify the usage information in the response."""
    # Count tokens including special tokens
    actual_total_tokens = input_tokens + output_sequence

    completion_diff = usage.completion_tokens / output_sequence
    if completion_diff < 0.5 or completion_diff > 2:
        error_msg = f"Reported completion tokens ({usage.completion_tokens}) does not match actual count ({output_sequence})"
        return False, error_msg, "INCORRECT_USAGE_DATA"

    prompt_diff = usage.prompt_tokens / input_tokens
    if (prompt_diff < 0.5) or (prompt_diff > 2):
        error_msg = f"Reported prompt tokens ({usage.prompt_tokens}) does not match actual count ({input_tokens})"
        return False, error_msg, "INCORRECT_USAGE_DATA"

    total_diff = usage.total_tokens / (input_tokens + output_sequence)
    if total_diff < 0.5 or total_diff > 2:
        error_msg = f"Reported total tokens ({usage.total_tokens}) does not match actual count ({actual_total_tokens})"
        return False, error_msg, "INCORRECT_USAGE_DATA"

    return True, "", ""


def parse_token_id(token: Optional[str]) -> Optional[int]:
    """Parse token string into token ID."""
    if token is None:
        return -1

    if token.startswith("token_id:"):
        try:
            return int(token.split(":")[1])
        except (IndexError, ValueError):
            return None

    try:
        return int(token)
    except (ValueError, TypeError):
        return None


def parse_chunk(chunk: Dict, request_type: str) -> Optional[OutputItem]:
    """Parse a raw chunk into an OutputItem with token info."""
    try:
        choices = chunk.get("choices", [])
        if not choices:
            return None

        choice = choices[0]

        if request_type == "CHAT":
            delta = choice.get("delta")
            if delta is None:
                return None

            # Skip assistant role messages without content/tools
            if delta.get("role") == "assistant" and not any(
                [delta.get(k) for k in ["content", "tool_calls", "function_call"]]
            ):
                return None

            choiceprobs = choice.get("logprobs")
            if choiceprobs is None:
                return None

            # Handle tool calls
            if delta.get("tool_calls"):
                tool_call = delta["tool_calls"][0]
                if tool_call.get("function"):
                    content_probs = choiceprobs.get("content", [{}])[0]
                    token = content_probs.get("token", "")
                    token_id = parse_token_id(token)
                    if token_id is None:
                        return None
                    return OutputItem(
                        text=json.dumps(tool_call),
                        token_id=token_id,
                        logprob=content_probs.get("logprob", -100),
                    )

            # Handle regular chat content
            content_probs = choiceprobs.get("content", [{}])[0]
            token = content_probs.get("token", "")
            token_id = parse_token_id(token)
            if token_id is None:
                return None
            text = delta.get("content", "")

            # Allow empty text if we have a valid token_id (for EOS tokens)
            if token_id == -1:
                return None

            return OutputItem(
                text=text,
                token_id=token_id,
                logprob=content_probs.get("logprob", -100),
            )

        elif request_type == "COMPLETION":
            text = choice.get("text")
            if text is None:
                return None

            logprobs = choice.get("logprobs")
            if logprobs is None:
                return None

            token = logprobs.get("tokens", [""])[0]
            token_id = parse_token_id(token)
            if token_id is None:
                return None

            if token_id == -1:
                return None

            token_logprobs = logprobs.get("token_logprobs", [])
            logprob = token_logprobs[0] if token_logprobs else -100

            return OutputItem(text=text, token_id=token_id, logprob=logprob)

    except Exception:
        return None


@app.post("/verify")
async def verify_wrapper(request: VerificationRequest) -> Dict:
    res = {}
    try:
        res = await verify(request)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
    return res


async def verify(request: VerificationRequest) -> Dict:
    """Verify a miner's output."""
    output_sequence: List[OutputItem] = []
    input_text = None

    # Parse raw chunks into OutputItems
    for chunk in request.raw_chunks:
        if parsed := parse_chunk(chunk, request.request_type):
            output_sequence.append(parsed)

    # If we couldn't parse enough tokens, fail
    if len(output_sequence) < 3:
        return {
            "verified": False,
            "error": f"Output sequence too short! Only parsed {len(output_sequence)} tokens",
            "cause": "TOO_SHORT",
        }

    # Check max tokens - allow for model-specific limits
    max_allowed = request.request_params.max_tokens

    if len(output_sequence) > max_allowed:
        return {
            "verified": False,
            "error": f"Too many tokens produced: {max_allowed} < {len(output_sequence)}",
            "cause": "TOO_LONG",
        }

    if request.model != MODEL_NAME:
        return {
            "error": f"Unable to verify model={request.model}, since we are using {MODEL_NAME}",
            "cause": "INTERNAL_ERROR",
        }

    final_chunk = request.raw_chunks[-1]
    usage_data = final_chunk.get("usage")
    if not usage_data:
        return {
            "verified": False,
            "error": "No usage information in final chunk",
            "cause": "NO_USAGE",
        }

    reported_usage = Usage(**usage_data)

    # Tokenize the input sequence
    TOKENIZER = MODEL_WRAPPER.tokenizer_manager.tokenizer
    assert TOKENIZER is not None
    input_text = (
        request.request_params.prompt
        if request.request_type == RequestType.COMPLETION.value
        else TOKENIZER.apply_chat_template(
            request.request_params.messages,  # type: ignore
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True,
            tools=request.request_params.tools,  # type: ignore
            tool_choice=request.request_params.tool_choice,
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

        # Verify usage information
        # Response - 1 for usage chunk
        res = verify_usage(
            len(input_tokens), len(request.raw_chunks) - 2, reported_usage
        )
        if res is None:
            return {"error": "Failed to check usage", "cause": "INTERNAL_ERROR"}
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

        # Pops think character for r1
        if input_text.strip().endswith(output_sequence[1].text):
            output_sequence.pop(1)
        if input_text.strip().endswith(output_sequence[0].text):
            output_sequence.pop(0)
        # Logprob checks
        res = await verify_logprobs(
            request.request_params.temperature,
            request.request_params.max_tokens,
            str(input_text),
            input_tokens,
            output_sequence,
        )
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

        print("Verified Response")
        return {
            "verified": True,
            "input_tokens": len(input_tokens),
            "response_tokens": len([o for o in output_sequence if o.text != ""]),
            "gpus": TENSOR_PARALLEL,
        }


@app.get("/metadata")
async def endpoints():
    ENDPOINTS = ["completion"]
    TOKENIZER = MODEL_WRAPPER.tokenizer_manager.tokenizer
    assert TOKENIZER is not None
    if TOKENIZER.chat_template is not None:
        ENDPOINTS.append("chat")

    return {"endpoints": ENDPOINTS}


@app.get("/")
def ping():
    return "", 200
