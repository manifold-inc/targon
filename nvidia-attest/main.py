import os
from pydantic import BaseModel
from nv_attestation_sdk import attestation
from fastapi import FastAPI
import json
from typing import Optional
from logconfig import setupLogging

logger = setupLogging()


def load_policy(policy_path: str) -> Optional[str]:
    try:
        with open(policy_path, "r") as f:
            policy = json.load(f)
        return json.dumps(policy)
    except Exception as e:
        return None


POLICY_PATH = os.environ.get("APPRAISAL_POLICY", "targon/remote_policy.json")
ATTESTATION_POLICY = load_policy(POLICY_PATH)

app = FastAPI()


@app.get("/")
def ping():
    return ""


class Request(BaseModel):
    token: str
    expected_nonce: str


@app.post("/attest")
def attest(req: Request) -> bool:
    try:
        NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"
        client = attestation.Attestation("Verifier")
        client.add_verifier(
            attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, ""
        )
        client.set_token("Verifier", req.token)
        client.set_nonce(req.expected_nonce)
        valid: bool = client.validate_token(ATTESTATION_POLICY)  # type: ignore
        return valid
    except Exception as e:
        return False
