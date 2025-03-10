from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json

from pydantic import BaseModel


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


class RequestType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class VerificationRequest(BaseModel):
    request_type: str
    model: str = ""
    request_params: RequestParams
    raw_chunks: List[Dict]


class RequestSamplingParams(BaseModel):
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int


class GenerateRequest(BaseModel):
    messages: List[Dict[str, str]]
    sampling_params: RequestSamplingParams


def parse_chunk(chunk: Dict, request_type: str) -> Optional[str]:
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

            # Handle tool calls
            if delta.get("tool_calls"):
                tool_call = delta["tool_calls"][0]
                if tool_call.get("function"):
                    return json.dumps(tool_call)

            # Handle regular chat content
            text = delta.get("content", "")
            return text

        elif request_type == "COMPLETION":
            text = choice.get("text")
            return text

    except Exception:
        return None
