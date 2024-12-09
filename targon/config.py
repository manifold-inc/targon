# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2024 Manifold Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import bittensor as bt

import requests
import dotenv


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


dotenv.load_dotenv()
AUTO_UPDATE = not str2bool(os.getenv("NO_AUTO_UPDATE", "False"))
IMAGE_TAG = os.getenv("IMAGE_TAG", "latest")
HEARTBEAT = str2bool(os.getenv("HEARTBEAT", "False"))
IS_TESTNET = str2bool(os.getenv("IS_TESTNET", "False"))


def validate_config_and_neuron_path(config):
    r"""Checks/validates the config namespace object."""
    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    bt.logging.info(f"Logging path: {full_path}")
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)
    return config


def add_args(parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=4)

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Neuron Name",
        default="targon",
    )

    parser.add_argument(
        "--epoch-length",
        type=int,
        dest="epoch_length",
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=360,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode",
        default=False,
    )

    parser.add_argument(
        "--autoupdate-off",
        action="store_false",
        dest="autoupdate",
        help="Disable automatic updates to Targon on latest version on Main.",
        default=True,
    )


def add_miner_args(parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--model-endpoint",
        dest="model_endpoint",
        type=str,
        help="The endpoint to use for the OpenAI Compatible client.",
        default="http://127.0.0.1:8000/v1",
    )

    parser.add_argument(
        "--no-force-validator-permit",
        dest="no_force_validator_permit",
        action="store_true",
        help="If set, we will not force incoming requests to have a permit.",
        default=False,
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        type=str,
        help="API key for openai compatable api",
        default="12345",
    )


def add_validator_args(parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--cache-file",
        dest="cache_file",
        type=str,
        help="File to save scores, and other misc data that can persist through validator restarts",
        default="cache.json",
    )

    parser.add_argument(
        "--miner-timeout",
        dest="miner_timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=12,
    )

    parser.add_argument(
        "--vpermit-tao-limit",
        dest="vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=4096,
    )

    parser.add_argument(
        "--database.url",
        dest="database.url",
        type=str,
        help="Database URL to score organic queries",
        default=None,
    )

    parser.add_argument(
        "--models.mode",
        dest="models.mode",
        type=str,
        help="Which method to use when fetching models",
        choices=["endpoint", "config", "default"],
        default="default",
    )
    parser.add_argument(
        "--models.endpoint",
        dest="models.endpoint",
        type=str,
        help="Endpoint to query for models",
        default="https://targon.sybil.com/api/models",
    )


def get_models_from_endpoint(endpoint: str):
    try:
        res = requests.get(endpoint)
        bt.logging.info(res.text)
        res = res.json()
        if not isinstance(res, list):
            raise Exception(
                f"Unexpected type received from endpoint. Must be type list. got {res}"
            )
        return res
    except Exception as e:
        bt.logging.error(f"Failed to get models from {endpoint}: {str(e)}")
    return None


def get_models_from_config():
    filename = "./models.txt"
    try:
        with open(filename, "r") as file:
            models = file.read().strip().split("\n")
            if not len(models):
                bt.logging.error("No models in models file")
            else:
                bt.logging.info(f"Found models {str(models)}")
            return models
    except IOError:
        bt.logging.info("No model file found")
    except EOFError:
        bt.logging.warning("Curropted models file")
    except Exception as e:
        bt.logging.error(f"Failed reading model file: {e}")
    return None
