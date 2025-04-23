from pydantic import BaseModel
from nv_attestation_sdk import attestation
from fastapi import FastAPI
import json
from typing import Optional, Dict, Any
from logconfig import setupLogging
import jwt

logger = setupLogging()


def load_policy(filename: str) -> Optional[str]:
    try:
        with open(filename, "r") as f:
            policy = json.load(f)
        return json.dumps(policy)
    except Exception as e:
        logger.error(f"No policy found: {e}")
        return None


app = FastAPI()
GPU_ATTESTATION_POLICY = load_policy("gpu_remote_policy")
SWITCH_ATTESTATION_POLICY = load_policy("switch_remote_policy")

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
    attestation_result: bool
    token: str
    valid: bool

class Request(BaseModel):
    data: AttestResponse
    expected_nonce: str

def extract_gpu_claims_from_token(token_data: str, expected_nonce: str) -> Dict[str, Any]:
    """
    Extract claims from token data using PyJWT.
    
    Args:
        token_data: Token data from get_token()
        
    Returns:
        Dictionary of claims or empty dict if parsing fails
    """
    try:
        gpu_claims = {}
        
        # Handle string token (likely in JSON format)
        if isinstance(token_data, str):
            try:
                # First, try to parse as JSON
                token_json = json.loads(token_data)
                
                # The token has a complex nested structure:
                # [[JWT, token], {REMOTE_GPU_CLAIMS: [[JWT, token], {GPU-0: token, ...}]}]
                
                # Look for LOCAL_GPU_CLAIMS
                if isinstance(token_json, list) and len(token_json) >= 2 and isinstance(token_json[1], dict):
                    for claims_key, claims_val in token_json[1].items():
                        if claims_key == "LOCAL_GPU_CLAIMS" and isinstance(claims_val, list) and len(claims_val) >= 2:
                            gpu_dict = claims_val[1]
                            if isinstance(gpu_dict, dict):
                                # Now we have a dictionary of GPU-ID to JWT token
                                for gpu_id, gpu_token in gpu_dict.items():
                                    if gpu_id.startswith("GPU-"):
                                        # For each GPU, extract the claims by decoding the JWT with PyJWT
                                        try:
                                            # Decode without verification (we're just extracting claims)
                                            gpu_token_data = jwt.decode(
                                                gpu_token, 
                                                options={"verify_signature": False},
                                                algorithms=["ES384", "HS256"]  # Support both NVIDIA's ES384 and test HS256
                                            )
                                            
                                            # Add this GPU's claims to our results
                                            gpu_claims[gpu_id] = {
                                                "gpu_type": gpu_token_data.get("gpu_type", "Unknown"),
                                            }

                                            if gpu_token_data.get("eat_nonce") != expected_nonce:
                                                return False
                                            
                                        except jwt.PyJWTError as e:
                                            logger.debug(f"Failed to decode JWT for {gpu_id}: {str(e)}")
                                            gpu_claims[gpu_id] = {
                                                "error": f"Failed to decode JWT: {str(e)}"
                                            }
            except Exception as e:
                logger.debug(f"Failed to parse token as JSON: {str(e)}")
        
        # If we successfully extracted claims, return them
        if gpu_claims:
            return gpu_claims
            
        # If we get here, we couldn't extract claims
        logger.debug("Unable to extract claims from token")
        return {}
        
    except Exception as e:
        logger.warning(f"Error extracting claims from token: {str(e)}")
        return {}

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
        valid: bool = client.validate_token(GPU_ATTESTATION_POLICY)  # type: ignore
        if not valid:
            return False
        
        claims = extract_gpu_claims_from_token(req.data.token, req.expected_nonce)
        if not claims:
            return False
        


    except Exception as e:
        logger.error(f"Error during attestation: {e}")
        return False

logger.info("Starting nvidia-attest")
