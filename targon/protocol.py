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


import pydantic
import bittensor as bt

from typing import List
from starlette.responses import StreamingResponse


class InferenceSamplingParams(pydantic.BaseModel):
    '''
    SamplingParams is a pydantic model that represents the sampling parameters for the TGI model.
    '''
    best_of: int = pydantic.Field(
        1,
        title="Best of",
        description="The number of samples to generate.",
    )

    decoder_input_details: bool = pydantic.Field(
        True,
        title="Decoder Input Details",
        description="Whether to return the decoder input details.",
    )

    details: bool = pydantic.Field(
        False,
        title="Details",
        description="Whether to return the details.",
    )

    do_sample: bool = pydantic.Field(
        True,
        title="Do Sample",
        description="Whether to sample.",
    )

    max_new_tokens: int = pydantic.Field(
        32,
        title="Max New Tokens",
        description="The maximum number of tokens to generate in the completion.",
    )

    repetition_penalty: float = pydantic.Field(
        1.0,
        title="Repetition Penalty",
        description="The repetition penalty.",
    )

    return_full_text: bool = pydantic.Field(
        False,
        title="Return Full Text",
        description="Whether to return the full text.",
    )

    seed: int = pydantic.Field(
        None,
        title="Seed",
        description="The seed used to generate the output.",
    )

    stop: List[str] = pydantic.Field(
        [""],
        title="Stop",
        description="The stop words.",
    )

    temperature: float = pydantic.Field(
        0.01,
        title="Temperature",
        description="Sampling temperature to use, between 0 and 2.",
    )

    top_k: int = pydantic.Field(
        10,
        title="Top K",
        description="Nucleus sampling parameter, top_p probability mass.",
    )

    top_n_tokens: int = pydantic.Field(
        5,
        title="Top N Tokens",
        description="The number of tokens to return.",
    )

    top_p: float = pydantic.Field(
        0.998,
        title="Top P",
        description="Nucleus sampling parameter, top_p probability mass.",
    )

    truncate: int = pydantic.Field(
        None,
        title="Truncate",
        description="The truncation length.",
    )

    typical_p: float = pydantic.Field(
        0.9999999,
        title="Typical P",
        description="The typical probability.",
    )

    watermark: bool = pydantic.Field(
        False,
        title="Watermark",
        description="Whether to watermark.",
    )

    stream: bool = pydantic.Field(
        False,
        title="Stream",
        description="Whether to stream.",
    )



import pydantic
from typing import List, Dict, Optional

class Inference(bt.StreamingSynapse):
    """
    Inference is a specialized implementation tailored for prompting functionalities within
    the Bittensor network or similar systems, designed to be compatible with the OpenAI API reference.

    This class is intended to interact with a streaming response that contains a sequence of tokens,
    representing prompts or messages in a chat scenario.

    Attributes:
    - `model` (str): The ID of the model to use.
    - `messages` (List[Dict[str, str]]): A list of messages, where each message is a dictionary with `role` and `content`.
    - `completion` (Optional[str]): Stores the processed result of the streaming tokens.
    """


    sources: List[str] = pydantic.Field(
        ...,
        title="Sources",
        description="A list of sources related to the query.",
    )

    query: str = pydantic.Field(
        ...,
        title="Query",
        description="The query to be sent to the Bittensor network.",
    )

    sampling_params: InferenceSamplingParams = pydantic.Field(
        InferenceSamplingParams(),
        title="Sampling Params",
        description="The sampling parameters for the TGI model.",
    )
    completion: str = pydantic.Field(
        None,
        title="Completion",
        description="The processed result of the streaming tokens.",
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["sources", "query", "seed"],
        title="Required Hash Fields",
        description="A list of fields that are required for the hash.",
    )



    async def process_streaming_response(self, response: StreamingResponse):
        """
        `process_streaming_response` is an asynchronous method designed to process the incoming streaming response from the
        Bittensor network. It's the heart of the Challenge class, ensuring that streaming tokens, which represent
        prompts or messages, are decoded and appropriately managed.

        As the streaming response is consumed, the tokens are decoded from their 'utf-8' encoded format, split based on
        newline characters, and concatenated into the `completion` attribute. This accumulation of decoded tokens in the
        `completion` attribute allows for a continuous and coherent accumulation of the streaming content.

        Args:
            response: The streaming response object containing the content chunks to be processed. Each chunk in this
                      response is expected to be a set of tokens that can be decoded and split into individual messages or prompts.
        """
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8").split("\n")
            for token in tokens:
                if token:
                    self.completion += token
            yield tokens

    def deserialize(self) -> str:
        """
        Deserializes the response by returning the completion attribute.

        Returns:
            str: The completion result.
        """
        return self.completion

    def extract_response_json(self, response: StreamingResponse) -> dict:
        """
        `extract_response_json` is a method that performs the crucial task of extracting pertinent JSON data from the given
        response. The method is especially useful when you need a detailed insight into the streaming response's metadata
        or when debugging response-related issues.

        Beyond just extracting the JSON data, the method also processes and structures the data for easier consumption
        and understanding. For instance, it extracts specific headers related to dendrite and axon, offering insights
        about the Bittensor network's internal processes. The method ultimately returns a dictionary with a structured
        view of the extracted data.

        Args:
            response: The response object from which to extract the JSON data. This object typically includes headers and
                      content which can be used to glean insights about the response.

        Returns:
            dict: A structured dictionary containing:
                - Basic response metadata such as name, timeout, total_size, and header_size.
                - Dendrite and Axon related information extracted from headers.
                - Roles and Messages pertaining to the current Challenge instance.
                - The accumulated completion.
        """
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
            "sources": self.sources,
            "query": self.query,
            "completion": self.completion,
        }



