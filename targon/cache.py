import json
import traceback
from typing import Any, Dict, List
import bittensor as bt


def load_cache(file_name: str, block: int, miners: List[int]):
    miner_tps = {}
    try:
        with open(file_name, "r") as file:
            loaded_data: Dict[str, Any] = json.load(file)
            # Only load cache if fresh
            if loaded_data.get("version", 0) < 400000:
                raise Exception("Cache file from older targon version")
            if loaded_data.get("block_saved", 0) > block - 360:
                miner_cache: Dict[str, Any] = loaded_data.get("miner_tps", {})
                miner_tps = dict([(int(k), v) for k, v in miner_cache.items()])
    except IOError:
        bt.logging.info("No cache file found")
    except EOFError:
        bt.logging.warning("Curropted pickle file")
    except Exception as e:
        bt.logging.error(f"Failed reading cache file: {e}")
        bt.logging.error(traceback.format_exc())

    for miner in miners:
        if miner_tps.get(miner) is None:
            miner_tps[miner] = {}
    bt.logging.info("Loading cached data")
    return miner_tps
