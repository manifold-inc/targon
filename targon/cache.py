import json
import traceback
from typing import Any, Dict, List
import bittensor as bt


def save_organics(organics, filename="organics_cache.json"):
    try:
        with open(filename, "w") as f:
            json.dump(organics, f)
    except Exception as e:
        bt.logging.error(f"Failed writing cache file: {e}")
        bt.logging.error(traceback.format_exc())


def load_organics(filename="organics_cache.json"):
    try:
        with open(filename, "r") as file:
            loaded_data: Dict[str, Any] = json.load(file)
        return loaded_data
    except IOError:
        bt.logging.info("No cache file found")
    except EOFError:
        bt.logging.warning("Curropted pickle file")
    except Exception as e:
        bt.logging.error(f"Failed reading cache file: {e}")
        bt.logging.error(traceback.format_exc())
    return {}
