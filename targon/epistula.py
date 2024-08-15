import json
from typing import Annotated, Any, Dict, Generic, List, Optional, TypeVar, Union

import time
from pydantic import Field
from pydantic.generics import GenericModel
from substrateinterface import Keypair


# Define a type variable
T = TypeVar("T")


class EpistulaRequest(GenericModel, Generic[T]):
    data: T
    nonce: int = Field(
        title="Nonce", description="Unix timestamp of when request was sent"
    )
    signed_by: str = Field(title="Signed By", description="Hotkey of sender / signer")
    signed_for: str = Field(
        title="Signed For", description="Hotkey of intended receiver"
    )


def generate_body(
    data: Union[Dict[Any, Any], List[Any]], receiver_hotkey: str, sender_hotkey: str
) -> Dict[str, Any]:
    return {
        "data": data,
        "nonce": time.time_ns(),
        "signed_by": sender_hotkey,
        "signed_for": receiver_hotkey,
    }


def generate_header(
    hotkey: Keypair, body: Union[Dict[Any, Any], List[Any]]
) -> Dict[str, Any]:
    return {"Body-Signature": "0x" + hotkey.sign(json.dumps(body)).hex()}


def verify_signature(
    signature, body: bytes, nonce, sender, now
) -> Optional[Annotated[str, "Error Message"]]:
    if not isinstance(signature, str):
        return "Invalid Signature"
    if not isinstance(nonce, int):
        return "Invalid Nonce"
    if not isinstance(sender, str):
        return "Invalid Sender key"
    if not isinstance(body, bytes):
        return "Body is not of type bytes"
    ALLOWED_DELTA_NS = 5 * 1000000000
    keypair = Keypair(ss58_address=sender)
    if nonce + ALLOWED_DELTA_NS < now:
        return "Request is too stale"
    verified = keypair.verify(body, signature)
    if not verified:
        return "Signature Mismatch"
    return None
