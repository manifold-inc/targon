from typing import Annotated, Any, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, Field
from substrateinterface import Keypair


class EpistulaRequest(BaseModel):
    data: Type[Any]
    nonce: float = Field(
        title="Nonce", description="Unix timestamp of when request was sent"
    )
    signed_by: str = Field(title="Signed By", description="Hotkey of sender / signer")
    signed_for: str = Field(
        title="Signed For", description="Hotkey of intended receiver"
    )

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
    verified = keypair.verify(signature, body)
    if not verified:
        return "Signature Mismatch"
    return None
