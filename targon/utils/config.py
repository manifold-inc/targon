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
import torch
import argparse
import bittensor as bt
from loguru import logger
from typing import List

#TODO: enable 4bit and 8bit precision llms via config

def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/provers to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    bt.logging.info(f'Logging path: {full_path}')
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            os.path.join(config.neuron.full_path, "events.log"),
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=4)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=360,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and all network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default="2 GB",
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )
    
    parser.add_argument(
        "--neuron.log_full",
        action="store_true",
        help="If set, logs more information.",
        default=False,
    )

    parser.add_argument(
        "--no_background_thread",
        action="store_true",
        help="If set, we dont run the neuron in a background thread.",
        default=False,
    )

    # add blacklist keys
    parser.add_argument(
        '--blacklist.coldkeys',
        type=List[str],
        help='List of coldkeys to blacklist.',
        default=[],
    )

    parser.add_argument(
        '--blacklist.hotkeys',
        type=List[str],
        help='List of hotkeys to blacklist.',
        default=[],
    )


def add_prover_args(cls, parser):
    """Add prover specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default='prover',
    )

    parser.add_argument(
        "--blacklist.force_verifier_permit",
        action="store_false",
        help="If set, we will force incoming requests to have a permit.",
        default=True,
    )

    parser.add_argument(
        "--disable_auto_update",
        action="store_true",
        help="If true, the validator will disable auto-update of its software from the repository.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_false",
        help="If set, provers will accept queries from non registered entities. (Dangerous!)",
        default=True,
    )

    parser.add_argument(
        "--neuron.tgi_endpoint",
        type=str,
        help="The endpoint to use for the TGI client.",
        default="http://127.0.0.1:8080",
    )

def add_verifier_args(cls, parser):
    """Add verifier specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default='verifier',
    )

    parser.add_argument(
        "--disable_auto_update",
        action="store_true",
        help="If true, the validator will disable auto-update of its software from the repository.",
        default=False,
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=12,
    )

    parser.add_argument(
        "--neuron.max_tokens",
        type=int,
        help="The maximum number of tokens in generated responses.",
        default=256,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.verbose",
        action="store_true",
        help="If set, we will print more information.",
        default=False,
    )

    parser.add_argument(
        "--neuron.reward_mode",
        default="sigmoid",
        type=str,
        choices=["minmax", "sigmoid"],
        help="Reward mode for the validator.",
    )

    parser.add_argument(
        "--neuron.challenge_url",
        type=str,
        help="The url to use for the challenge server.",
        default="https://challenge.sybil.com/",
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of provers to query in a single step.",
        default=48,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.05,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the verifier needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a verifier with a vpermit.",
            default=4096,
        )
    
    parser.add_argument(
        "--neuron.tgi_endpoint",
        type=str,
        help="The endpoint to use for the TGI client.",
        default="http://localhost:8080",
    )
    
    parser.add_argument(
        "--database.host",
        type=str,
        help="The path to write debug logs to.",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--database.port",
        type=int,
        help="The path to write debug logs to.",
        default=6379,
    )

    parser.add_argument(
        "--database.index",
        type=int,
        help="The path to write debug logs to.",
        default=1,
    )

    parser.add_argument(
        "--database.password",
        type=str,
        help="the password to use for the redis database.",
        default=None,
    )

    parser.add_argument(
        "--neuron.compute_stats_interval",
        type=int,
        help="The interval at which to compute statistics.",
        default=360,
    )




def config(cls):
    """
    Returns the configuration object specific to this prover or verifier after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)