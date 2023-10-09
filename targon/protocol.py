import time
import torch
import pydantic

import bittensor as bt

from typing import List
from starlette.responses import StreamingResponse



class Targon( bt.Synapse ):

    class Config:
        """
        Pydantic model configuration class for Targon. This class sets validation of attribute assignment as True.
        validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
        """

        validate_assignment = True

    
    def deserialize(self) -> "Targon":
        """
        Returns:
            Targon: The current instance of the Targon class.
        """
        return self


    roles: List[str] = pydantic.Field(
        ...,
        title="Roles",
        description="roles for the LLM in chat scenario. Immuatable.",
        allow_mutation=False,
    )

    messages: List[str] = pydantic.Field(
        ...,
        title="Messages",
        description="List of messages between user and AI. Immutable.",
        allow_mutation=False,
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current Targon object. This attribute is mutable and can be updated.",
    )

    images: list[ bt.Tensor ] = []

    required_hash_fields: List[str] = pydantic.Field(
        ["messages"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )


class TargonStreaming( bt.StreamingSynapse ):


    roles: List[str] = pydantic.Field(
        ...,
        title="Roles",
        description="roles for the LLM in chat scenario. Immuatable.",
        allow_mutation=False,
    )

    messages: List[str] = pydantic.Field(
        ...,
        title="Messages",
        description="List of messages between user and AI. Immutable.",
        allow_mutation=False,
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current Targon object. This attribute is mutable and can be updated.",
    )

    # images: list[ bt.Tensor ] = []

    required_hash_fields: List[str] = pydantic.Field(
        ["messages"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

    async def process_streaming_response(self, response: StreamingResponse):
        """
        `process_streaming_response` is an asynchronous method designed to process the incoming streaming response from the
        Bittensor network. It's the heart of the StreamPrompting class, ensuring that streaming tokens, which represent
        prompts or messages, are decoded and appropriately managed.

        As the streaming response is consumed, the tokens are decoded from their 'utf-8' encoded format, split based on
        newline characters, and concatenated into the `completion` attribute. This accumulation of decoded tokens in the
        `completion` attribute allows for a continuous and coherent accumulation of the streaming content.

        Args:
            response: The streaming response object containing the content chunks to be processed. Each chunk in this
                      response is expected to be a set of tokens that can be decoded and split into individual messages or prompts.

        Usage:
            Generally, this method is called when there's an incoming streaming response to be processed.

            ```python
            stream_prompter = StreamPrompting(roles=["role1", "role2"], messages=["message1", "message2"])
            await stream_prompter.process_streaming_response(response)
            ```

        Note:
            It's important to remember that this method is asynchronous. Ensure it's called within an appropriate
            asynchronous context.
        """
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8").split("\n")
            for token in tokens:
                if token:
                    self.completion += token

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
                - Roles and Messages pertaining to the current StreamPrompting instance.
                - The accumulated completion.

        Usage:
            This method can be used after processing a response to gather detailed metadata:

            ```python
            stream_prompter = StreamPrompting(roles=["role1", "role2"], messages=["message1", "message2"])
            # After processing the response...
            json_info = stream_prompter.extract_response_json(response)
            ```

        Note:
            While the primary output is the structured dictionary, understanding this output can be instrumental in
            troubleshooting or in extracting specific insights about the interaction with the Bittensor network.
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
            "roles": self.roles,
            "messages": self.messages,
            # "images": self.images,
            "completion": self.completion,
        }