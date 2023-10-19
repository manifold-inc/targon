# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2023 Manifold Labs

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
import argparse
import bittensor as bt


def check_config(cls, config: "bt.Config"):
    """
    Validates the given configuration for the Miner by ensuring all necessary settings
    and directories are correctly set up. It checks the config for axon, wallet, logging,
    and subtensor. Additionally, it ensures that the logging directory exists or creates one.

    Args:
        cls: The class reference, typically referring to the Miner class.
        config (bt.Config): The configuration object holding various settings for the miner.

    Raises:
        Various exceptions can be raised by the check_config methods of axon, wallet, logging,
        and subtensor if the configurations are not valid.
    """
    bt.axon.check_config(config)
    bt.logging.check_config(config)
    full_path = os.path.expanduser(
        "{}/{}/{}/{}".format(
            config.logging.logging_dir,
            config.wallet.get("name", bt.defaults.wallet.name),
            config.wallet.get("hotkey", bt.defaults.wallet.hotkey),
            config.miner.name,
        )
    )
    config.miner.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.miner.full_path):
        os.makedirs(config.miner.full_path)


def get_config() -> "bt.Config":
    """
    Initializes and retrieves a configuration object for the Miner. This function sets up
    and reads the command-line arguments to customize various miner settings. The function
    also sets up the logging directory for the miner.

    Returns:
        bt.Config: A configuration object populated with settings from command-line arguments
                   and defaults where necessary.

    Note:
        Running this function with the `--help` argument will print a help message detailing
        all the available command-line arguments for customization.
    """
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--axon.port", type=int, default=8098, help="Port to run the axon on."
    )
    # Subtensor network to connect to
    parser.add_argument(
        "--subtensor.network",
        default="finney",
        help="Bittensor network to connect to.",
    )
    # Chain endpoint to connect to
    parser.add_argument(
        "--subtensor.chain_endpoint",
        default="wss://entrypoint-finney.opentensor.ai:443",
        help="Chain endpoint to connect to.",
    )
    # Adds override arguments for network and netuid. TargonSearchResultStream is 4.
    parser.add_argument("--netuid", type=int, default=4, help="The chain subnet uid.")

    parser.add_argument(
        "--miner.root",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="~/.bittensor/miners/",
    )
    parser.add_argument(
        "--miner.name",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="Bittensor Miner",
    )

    # Run config.
    parser.add_argument(
        "--miner.blocks_per_epoch",
        type=str,
        help="Blocks until the miner sets weights on chain",
        default=100,
    )

    # Blacklist.
    parser.add_argument(
        "--miner.blacklist.blacklist",
        type=str,
        required=False,
        nargs="*",
        help="Blacklist certain hotkeys",
        default=[],
    )
    parser.add_argument(
        "--miner.blacklist.whitelist",
        type=str,
        required=False,
        nargs="*",
        help="Whitelist certain hotkeys",
        default=[],
    )
    parser.add_argument(
        "--miner.blacklist.force_validator_permit",
        action="store_true",
        help="Only allow requests from validators",
        default=False,
    )
    parser.add_argument(
        "--miner.blacklist.allow_non_registered",
        action="store_true",
        help="If True, the miner will allow non-registered hotkeys to mine.",
        default=False,
    )
    parser.add_argument(
        "--miner.blacklist.minimum_stake_requirement",
        type=float,
        help="Minimum stake requirement",
        default=0.0,
    )
    parser.add_argument(
        "--miner.blacklist.prompt_cache_block_span",
        type=int,
        help="Amount of blocks to keep a prompt in cache",
        default=7200,
    )
    parser.add_argument(
        "--miner.blacklist.use_prompt_cache",
        action="store_true",
        help="If True, the miner will use the prompt cache to store recent request prompts.",
        default=False,
    )
    parser.add_argument(
        "--miner.blacklist.min_request_period",
        type=int,
        help="Time period (in minute) to serve a maximum of 50 requests for each hotkey",
        default=5,
    )

    # Priority.
    parser.add_argument(
        "--miner.priority.default",
        type=float,
        help="Default priority of non-registered requests",
        default=0.0,
    )
    parser.add_argument(
        "--miner.priority.time_stake_multiplicate",
        type=int,
        help="Time (in minute) it takes to make the stake twice more important in the priority queue",
        default=10,
    )
    parser.add_argument(
        "--miner.priority.len_request_timestamps",
        type=int,
        help="Number of historic request timestamps to record",
        default=50,
    )
    # Switches.
    parser.add_argument(
        "--miner.no_set_weights",
        action="store_true",
        help="If True, the miner does not set weights.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_serve",
        action="store_true",
        help="If True, the miner doesnt serve the axon.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_start_axon",
        action="store_true",
        help="If True, the miner doesnt start the axon.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_register",
        action="store_true",
        help="If True, the miner doesnt register its wallet.",
        default=False,
    )

    # Mocks.
    parser.add_argument(
        "--miner.mock_subtensor",
        action="store_true",
        help="If True, the miner will allow non-registered hotkeys to mine.",
        default=False,
    )

    # Wandb
    parser.add_argument(
        "--wandb.on", action="store_true", help="Turn on wandb.", default=False
    )
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where youre sending the new run.",
        default=None,
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="An entity is a username or team name where youre sending runs.",
        default=None,
    )

    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)

    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)

    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)

    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)

    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)

    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config
