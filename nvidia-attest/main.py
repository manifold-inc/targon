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

class GPUAttestation(BaseModel):
    attestation_result: bool
    token: str
    valid: bool

class SwitchAttestation(BaseModel):
    attestation_result: bool
    token: str
    valid: bool

class GPUClaims(BaseModel):
    gpu_type: str

class AttestationResponse(BaseModel):
    gpu_attestation_success: bool
    switch_attestation_success: bool
    gpu_claims: Optional[Dict[str, GPUClaims]] = None

class Request(BaseModel):
    gpu: GPUAttestation
    switch: SwitchAttestation
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

def extract_switch_claims_from_token(token_data: str, expected_nonce: str) -> bool:
    """
    Extract claims from switch token data using PyJWT.
    
    Args:
        token_data: Token data from get_token()
        expected_nonce: Expected nonce value to verify
        
    Returns:
        True if claims are extracted successfully and nonce matches, False otherwise
    """
    try:
        # Handle string token (likely in JSON format)
        if isinstance(token_data, str):
            try:
                # First, try to parse as JSON
                token_json = json.loads(token_data)
                
                # The token has a complex nested structure:
                # [[JWT, token], {REMOTE_SWITCH_CLAIMS: [[JWT, token], {SWITCH-0: token, ...}]}]
                
                # Look for LOCAL_SWITCH_CLAIMS
                if isinstance(token_json, list) and len(token_json) >= 2 and isinstance(token_json[1], dict):
                    for claims_key, claims_val in token_json[1].items():
                        if claims_key == "LOCAL_SWITCH_CLAIMS" and isinstance(claims_val, list) and len(claims_val) >= 2:
                            switch_dict = claims_val[1]
                            if isinstance(switch_dict, dict):
                                # Now we have a dictionary of SWITCH-ID to JWT token
                                for switch_id, switch_token in switch_dict.items():
                                    if switch_id.startswith("SWITCH-"):
                                        try:
                                            # Decode without verification (we're just extracting claims)
                                            switch_token_data = jwt.decode(
                                                switch_token, 
                                                options={"verify_signature": False},
                                                algorithms=["ES384", "HS256"]  # Support both NVIDIA's ES384 and test HS256
                                            )
                                            
                                            # Verify nonce match
                                            if switch_token_data.get("eat_nonce") != expected_nonce:
                                                return False
                                            
                                            # Return the switch claims
                                            return True
                                        except jwt.PyJWTError as e:
                                            logger.debug(f"Failed to decode JWT for {switch_id}: {str(e)}")
                                            return False
            except Exception as e:
                logger.debug(f"Failed to parse token as JSON: {str(e)}")
        
        # If we get here, we couldn't extract claims
        logger.debug("Unable to extract claims from token")
        return False
        
    except Exception as e:
        logger.warning(f"Error extracting claims from token: {str(e)}")
        return False

@app.post("/attest", response_model=AttestationResponse)
def attest(req: Request) -> AttestationResponse:
    try:        
        # GPU Attestation
        gpu_client = attestation.Attestation("GPUVerifier")
        gpu_client.set_name("HGX-node")
        gpu_client.set_nonce(req.expected_nonce)
        gpu_client.set_claims_version("2.0")
        gpu_client.set_ocsp_nonce_disabled(False)
        
        gpu_client.add_verifier(
            dev=attestation.Devices.GPU,
            env=attestation.Environment.LOCAL,
            url="https://nras.attestation.nvidia.com/v3/attest/gpu",
            evidence="",
            ocsp_url="https://ocsp.ndis.nvidia.com/",
            rim_url="https://rim.attestation.nvidia.com/v1/rim/"
        )

        # Set the token from the request
        gpu_client.set_token("GPUVerifier", req.gpu.token)
        
        # Validate GPU token with policy
        if not gpu_client.validate_token(GPU_ATTESTATION_POLICY):
            return AttestationResponse(
                gpu_attestation_success=False,
                switch_attestation_success=False
            )

        # Verify GPU claims
        gpu_claims = extract_gpu_claims_from_token(req.gpu.token, req.expected_nonce)
        if not gpu_claims:
            return AttestationResponse(
                gpu_attestation_success=False,
                switch_attestation_success=False
            )

        # Switch Attestation
        switch_client = attestation.Attestation("SwitchVerifier")
        switch_client.set_name("HGX-node")
        switch_client.set_nonce(req.expected_nonce)
        switch_client.set_claims_version("2.0")
        switch_client.set_ocsp_nonce_disabled(False)
        
        switch_client.add_verifier(
            dev=attestation.Devices.SWITCH,
            env=attestation.Environment.LOCAL,
            url="https://nras.attestation.nvidia.com/v3/attest/switch",
            evidence="",
            ocsp_url="https://ocsp.ndis.nvidia.com/",
            rim_url="https://rim.attestation.nvidia.com/v1/rim/"
        )

        # Set the token from the request
        switch_client.set_token("SwitchVerifier", req.switch.token)
        
        # Validate switch token with policy
        if not switch_client.validate_token(SWITCH_ATTESTATION_POLICY):
            return AttestationResponse(
                gpu_attestation_success=True,
                switch_attestation_success=False,
                gpu_claims=gpu_claims
            )

        # Verify switch claims
        if not extract_switch_claims_from_token(req.switch.token, req.expected_nonce):
            return AttestationResponse(
                gpu_attestation_success=True,
                switch_attestation_success=False,
                gpu_claims=gpu_claims
            )

        return AttestationResponse(
            gpu_attestation_success=True,
            switch_attestation_success=True,
            gpu_claims=gpu_claims
        )

    except Exception as e:
        logger.error(f"Error during attestation: {e}")
        return AttestationResponse(
            gpu_attestation_success=False,
            switch_attestation_success=False
        )

logger.info("Starting nvidia-attest")
