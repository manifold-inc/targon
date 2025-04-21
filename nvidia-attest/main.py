from pydantic import BaseModel
from nv_attestation_sdk import attestation
from fastapi import FastAPI
import json
from typing import Dict, List, Optional
from logconfig import setupLogging

logger = setupLogging()


def load_policy() -> Optional[str]:
    try:
        with open("remote_policy.json", "r") as f:
            policy = json.load(f)
        return json.dumps(policy)
    except Exception as e:
        logger.error(f"No policy found: {e}")
        return None

app = FastAPI()


@app.get("/")
def ping():
    return ""


class AttestClaimInfo(BaseModel):
    hwmodel: str
    driver_version: str
    vbios_version: str
    measres: str
    attestation_success: bool


class AttestGPUInfo(BaseModel):
    driver_version: str
    vbios_version: str
    measres: str
    attestation_success: bool


class AttestGPU(BaseModel):
    id: str
    model: str
    claims: AttestGPUInfo


class AttestResponse(BaseModel):
    success: bool
    nonce: str
    token: str
    claims: Dict[str, AttestClaimInfo]
    validated: bool
    gpus: List[AttestGPU]


class Request(BaseModel):
    data: AttestResponse
    expected_nonce: str


@app.post("/attest")
def attest(req: Request) -> bool:
    try:
        NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"
        client = attestation.Attestation("Verifier")
        client.add_verifier(
            attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, ""
        )
        client.set_token("Verifier", req.data.token)
        client.set_nonce(req.expected_nonce)
        valid: bool = client.validate_token(ATTESTATION_POLICY)  # type: ignore
        return valid
    except Exception as e:
        logger.error(f"Error during attestation: {e}")
        return False
