from typing import Dict, Tuple
import json
import aiohttp

from targon.epistula import generate_header


async def broadcast(
    uid,
    models,
    axon_info,
    session: aiohttp.ClientSession,
    hotkey,
) -> Tuple[int, Dict[str, int], str]:
    try:
        req_bytes = json.dumps(
            models, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        async with session.post(
            f"http://{axon_info.ip}:{axon_info.port}/models",
            headers={
                "Content-Type": "application/json",
                **generate_header(hotkey, req_bytes, axon_info.hotkey),
            },
            data=req_bytes,
            timeout=aiohttp.ClientTimeout(total=3),
        ) as res:
            if res.status != 200:
                return uid, {}, f"Models response not 200: {res.status}"
            data = await res.json()
            if not isinstance(data, Dict):
                return uid, {}, "Model Data not list"
            return uid, data, ""

    except Exception as e:
        return uid, {}, f"Unknown error: {e}"
