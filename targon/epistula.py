import json
from hashlib import sha256
from uuid import uuid4
from math import ceil
from typing import Annotated, Any, Dict, List, Optional, Union

import time
from substrateinterface import Keypair


def generate_header(
    hotkey: Keypair,
    body: Union[Dict[Any, Any], List[Any]],
    signed_for: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = round(time.time() * 1000)
    timestampInterval = ceil(timestamp / 1e4) * 1e4
    uuid = str(uuid4())
    headers = {
        "Epistula-Version": str(2),
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x"
        + hotkey.sign(
            f"{sha256(json.dumps(body).encode('utf-8'))}.{uuid}.{timestamp}.{signed_for}"
        ).hex(),
    }
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        headers["Epistula-Secret-Signature-0"] = (
            "0x" + hotkey.sign(str(timestampInterval - 1) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-1"] = (
            "0x" + hotkey.sign(str(timestampInterval) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-2"] = (
            "0x" + hotkey.sign(str(timestampInterval + 1) + "." + signed_for).hex()
        )
    return headers


def verify_signature(
    signature, body: bytes, timestamp, uuid, signed_for, sender, now
) -> Optional[Annotated[str, "Error Message"]]:
    if not isinstance(signature, str):
        return "Invalid Signature"
    timestamp = int(timestamp)
    if not isinstance(timestamp, int):
        return "Invalid Timestamp"
    if not isinstance(sender, str):
        return "Invalid Sender key"
    if not isinstance(signed_for, str):
        return "Invalid receiver key"
    if not isinstance(uuid, str):
        return "Invalid uuid"
    if not isinstance(body, bytes):
        return "Body is not of type bytes"
    ALLOWED_DELTA_MS = 5000
    keypair = Keypair(ss58_address=sender)
    if timestamp + ALLOWED_DELTA_MS < now:
        return "Request is too stale"
    verified = keypair.verify(
        f"{sha256(body)}.{uuid}.{timestamp}.{signed_for}", signature
    )
    if not verified:
        return "Signature Mismatch"
    return None
