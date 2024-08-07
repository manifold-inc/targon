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


from bittensor.stream import ClientResponse
import json
import pydantic
import bittensor as bt

from typing import Any, List, Optional


class InferenceSamplingParams(pydantic.BaseModel):
    """
    SamplingParams is a pydantic model that represents the sampling parameters for the OpenAI Compabtable API model.
    """

    seed: int = pydantic.Field(
        title="Seed",
        description="The seed used to generate the output.",
    )

    best_of: Optional[int] = pydantic.Field(
        default=1,
        title="Best of",
        description="The number of samples to generate.",
    )

    decoder_input_details: Optional[bool] = pydantic.Field(
        default=True,
        title="Decoder Input Details",
        description="Whether to return the decoder input details.",
    )

    details: Optional[bool] = pydantic.Field(
        default=False,
        title="Details",
        description="Whether to return the details.",
    )

    do_sample: Optional[bool] = pydantic.Field(
        default=True,
        title="Do Sample",
        description="Whether to sample.",
    )

    max_new_tokens: Optional[int] = pydantic.Field(
        default=32,
        title="Max New Tokens",
        description="The maximum number of tokens to generate in the completion.",
    )

    repetition_penalty: Optional[float] = pydantic.Field(
        default=1.0,
        title="Repetition Penalty",
        description="The repetition penalty.",
    )

    return_full_text: Optional[bool] = pydantic.Field(
        default=False,
        title="Return Full Text",
        description="Whether to return the full text.",
    )

    stop: Optional[List[str]] = pydantic.Field(
        default=[""],
        title="Stop",
        description="The stop words.",
    )

    temperature: Optional[float] = pydantic.Field(
        default=0.01,
        title="Temperature",
        description="Sampling temperature to use, between 0 and 2.",
    )

    top_k: Optional[int] = pydantic.Field(
        default=10,
        title="Top K",
        description="Nucleus sampling parameter, top_p probability mass.",
    )

    top_n_tokens: Optional[int] = pydantic.Field(
        default=5,
        title="Top N Tokens",
        description="The number of tokens to return.",
    )

    top_p: Optional[float] = pydantic.Field(
        default=0.998,
        title="Top P",
        description="Nucleus sampling parameter, top_p probability mass.",
    )

    truncate: Optional[int] = pydantic.Field(
        default=None,
        title="Truncate",
        description="The truncation length.",
    )

    typical_p: Optional[float] = pydantic.Field(
        default=0.9999999,
        title="Typical P",
        description="The typical probability.",
    )

    watermark: Optional[bool] = pydantic.Field(
        default=False,
        title="Watermark",
        description="Whether to watermark.",
    )

    stream: Optional[bool] = pydantic.Field(
        default=False,
        title="Stream",
        description="Whether to stream.",
    )


class Inference(bt.StreamingSynapse):
    """
    Inference is a specialized implementation tailored for prompting functionalities within
    the Bittensor network or similar systems, designed to be compatible with the OpenAI API reference.

    This class is intended to interact with a streaming response that contains a sequence of tokens,
    representing prompts or messages in a chat scenario.

    Attributes:
    - `model` (str): The ID of the model to use.
    - `messages` (List[Dict[str, str]]): A list of messages, where each message is a dictionary with `role` and `content`.
    """

    messages: str = pydantic.Field(
        title="Message",
        description="The messages to be sent to the Bittensor network.",
    )

    sampling_params: Optional[InferenceSamplingParams] = pydantic.Field(
        default=InferenceSamplingParams(seed=333),
        title="Sampling Params",
        description="The sampling parameters for the OpenAI Compatible model.",
    )

    async def process_streaming_response(self, response: ClientResponse) -> Any:
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8")
            yield tokens

    def deserialize(self):
        return json.loads(self.messages)

    def extract_response_json(self, response: ClientResponse) -> dict:
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix):
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "messages": self.messages,
        }
