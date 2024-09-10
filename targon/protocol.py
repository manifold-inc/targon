# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Manifold Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, List, Optional


class InferenceSamplingParams(BaseModel):
    """
    SamplingParams is a pydantic model that represents the sampling parameters for the OpenAI Compabtable API model.
    """

    seed: int = Field(
        title="Seed",
        description="The seed used to generate the output.",
    )

    best_of: Optional[int] = Field(
        default=1,
        title="Best of",
        description="The number of samples to generate.",
    )

    max_tokens: Optional[int] = Field(
        default=32,
        title="Max New Tokens",
        description="The maximum number of tokens to generate in the completion.",
    )

    repetition_penalty: Optional[float] = Field(
        default=1.0,
        title="Repetition Penalty",
        description="The repetition penalty.",
    )

    stop: Optional[List[str]] = Field(
        default=[""],
        title="Stop",
        description="The stop words.",
    )

    temperature: Optional[float] = Field(
        default=0.01,
        title="Temperature",
        description="Sampling temperature to use, between 0 and 2.",
    )

    top_k: Optional[int] = Field(
        default=10,
        title="Top K",
        description="Nucleus sampling parameter, top_p probability mass.",
    )

    top_p: Optional[float] = Field(
        default=0.998,
        title="Top P",
        description="Nucleus sampling parameter, top_p probability mass.",
    )

    stream: Optional[bool] = Field(
        default=True,
        title="Stream",
        description="Whether to stream.",
    )

class InferenceStats(BaseModel):
    time_to_first_token: float
    time_for_all_tokens: float
    total_time: float
    tps: float
    response: str
    tokens: List[Any]
    verified: bool



class Endpoints(Enum):
    CHAT = 'CHAT',
    COMPLETION = 'COMPLETION'
