from typing import Dict, List, Optional, Tuple

import aiohttp
import requests
from targon.types import Endpoints
from targon.utils import fail_with_none


def get_tool_parser_for_model(model_name: str) -> Optional[str]:
    """Determine if a model supports tool calling and return its parser type.
    Based on vLLM's supported models documentation."""

    model_lower = model_name.lower()

    # For combined models like "hermes-3-llama-3.1", prefer llama3_json parser
    if "llama-3.1" in model_lower:
        return "llama3_json"

    # Hermes models (hermes)
    # All Nous Research Hermes-series models newer than Hermes 2 Pro
    models = ["hermes-3", "hermes-2-pro"]
    if model_lower in models:
        return "hermes"

    return None


@fail_with_none("Failed to check tokens")
async def check_tokens(
    request,
    raw_chunks: List[Dict],
    endpoint: Endpoints,
    port: Optional[int],
    url,
    request_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        request_data = {
            "model": request.get("model"),
            "request_type": endpoint.value,
            "request_params": request,
            "raw_chunks": raw_chunks,
            "request_id": request_id,
        }
        _url = f"{url}:{port}/verify"
        if port is None:
            _url = f"{url}/verify"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"bearer {api_key}",
                },
                timeout=aiohttp.ClientTimeout(total=60),
                json=request_data,
            ) as response:
                if response.status != 200:
                    return None, await response.text()
                result = await response.json()
                if result.get("verified") is None:
                    return None, str(result)
                return result, None
    except Exception as e:
        return None, str(e)
