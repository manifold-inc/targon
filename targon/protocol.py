import time
import torch
import pydantic

import bittensor as bt

from typing import List


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


