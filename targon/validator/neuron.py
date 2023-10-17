# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import copy
import torch
import asyncio
import bittensor as bt

from targon.validator.dataset import Dataset, MockDataset
from targon.validator.dendrite import AsyncDendritePool
from targon.validator.gating import GatingModel, SentenceEmbedGatingModel
from targon.validator.mock import MockDendritePool, MockRewardModel, MockGatingModel

# Load local forward function.
from targon.validator.config import add_args, check_config, config
from targon.validator.run import run
from targon.validator.misc import ttl_get_block
from targon.validator.utils import init_wandb

# Load gating models
from targon.validator.reward import (
    Blacklist,
    TaskValidator,
    NSFWRewardModel,
    DirectPreferenceRewardModel,
    OpenAssistantRewardModel,
    ReciprocateRewardModel,
    RelevanceRewardModel,
    MockRewardModel,
    DahoasRewardModel,
    DiversityRewardModel,
    PromptRewardModel,
    RewardModelType,
)


class neuron:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    def run(self):
        run(self)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"

    def __init__(self):
        self.config = neuron.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        print(self.config)
        bt.logging.info("neuron.__init__()")

        # Init device.
        bt.logging.debug("loading", "device")
        self.device = torch.device(self.config.neuron.device)
        bt.logging.debug(str(self.device))

        # Init subtensor
        bt.logging.debug("loading", "subtensor")
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.debug(str(self.subtensor))

        # Init wallet.
        bt.logging.debug("loading", "wallet")
        self.wallet = bt.wallet(config=self.config)
        self.wallet.create_if_non_existent()
        if not self.config.wallet._mock:
            if not self.subtensor.is_hotkey_registered_on_subnet(hotkey_ss58=self.wallet.hotkey.ss58_address, netuid=self.config.netuid):
                raise Exception(f'Wallet not currently registered on netuid {self.config.netuid}, please first register wallet before running')
                
        bt.logging.debug(str(self.wallet))

        # Init metagraph.
        bt.logging.debug("loading", "metagraph")
        self.metagraph = bt.metagraph(netuid=self.config.netuid, network=self.subtensor.network, sync=False) # Make sure not to sync without passing subtensor
        self.metagraph.sync(subtensor=self.subtensor) # Sync metagraph with subtensor.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        bt.logging.debug(str(self.metagraph))

        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(self.device)
        bt.logging.debug(str(self.moving_averaged_scores))

        # Dataset: used to generate the base prompts ( initial randomness. )
        bt.logging.debug("loading", "dataset")
        if self.config.neuron.mock_dataset:
            self.dataset = MockDataset()
        else:
            self.dataset = Dataset()
        bt.logging.debug(str(self.dataset))

        # Init the gating model which learns which miners to select for each query.
        bt.logging.debug("loading", "gating_model")
        if not self.config.gating.num_uids:
            self.config.gating.num_uids = self.subtensor.max_n(self.config.netuid)

        if self.config.neuron.mock_gating_model:
            self.gating_model = MockGatingModel(self.metagraph.n.item())
        elif self.config.neuron.use_custom_gating_model:
            self.gating_model = SentenceEmbedGatingModel(metagraph=self.metagraph, config=self.config).to(self.device)
        else:
            self.gating_model = GatingModel(metagraph=self.metagraph, config=self.config).to(self.device)
        bt.logging.debug(str(self.gating_model))

        if not self.config.neuron.axon_off:
            bt.logging.debug('serving ip to chain...')
            try:
                axon = bt.axon( 
                    wallet=self.wallet, metagraph=self.metagraph, config=self.config 
                )

                try:
                    self.subtensor.serve_axon(
                        netuid=self.config.netuid,
                        axon=axon,
                        use_upnpc=False,
                        wait_for_finalization=True,
                    )
                except Exception as e:
                    bt.logging.error(f'Failed to serve Axon with exception: {e}')
                    pass

                del axon
            except Exception as e:
                bt.logging.error(f'Failed to create Axon initialize with exception: {e}')
                pass

        else:
            bt.logging.debug('axon off, not serving ip to chain.')

        # Dendrite pool for querying the network during  training.
        bt.logging.debug("loading", "dendrite_pool")
        if self.config.neuron.mock_dendrite_pool:
            self.dendrite_pool = MockDendritePool()
        else:
            self.dendrite_pool = AsyncDendritePool(keypair=self.wallet.hotkey, metagraph=self.metagraph)
        bt.logging.debug(str(self.dendrite_pool))

        # Init Reward model
        bt.logging.debug("loading", "reward_functions")
        if self.config.neuron.mock_reward_models:
            self.reward_functions = []
            self.reward_weights = []
            self.masking_functions = [
                MockRewardModel(RewardModelType.blacklist.value),
                MockRewardModel(RewardModelType.nsfw.value),
            ]
            bt.logging.debug(str(self.reward_functions))
        else:
            self.reward_weights = torch.tensor(
                [
                    self.config.reward.dpo_weight,
                    self.config.reward.rlhf_weight,
                    self.config.reward.reciprocate_weight,
                    self.config.reward.dahoas_weight,
                    self.config.reward.prompt_based_weight,
                ],
                dtype=torch.float32,
            ).to(self.device)

            # Ensure reward function weights sum to 1.
            if self.reward_weights.sum() != 1:
                message = (
                    f"Reward function weights do not sum to 1 (Current sum: {self.reward_weights.sum()}.)"
                    f"Check your reward config file at `reward/config.py` or ensure that all your cli reward flags sum to 1."
                )
                bt.logging.error(message)
                raise Exception(message)

            self.reward_functions = [
                DirectPreferenceRewardModel(device=self.device)
                if self.config.reward.dpo_weight > 0
                else MockRewardModel(RewardModelType.dpo.value),
                OpenAssistantRewardModel(device=self.device)
                if self.config.reward.rlhf_weight > 0
                else MockRewardModel(RewardModelType.rlhf.value),
                ReciprocateRewardModel(device=self.device)
                if self.config.reward.reciprocate_weight > 0
                else MockRewardModel(RewardModelType.reciprocate.value),
                DahoasRewardModel(path=self.config.neuron.full_path, device=self.device)
                if self.config.reward.dahoas_weight > 0
                else MockRewardModel(RewardModelType.dahoas.value),
                PromptRewardModel(device=self.device)
                if self.config.reward.prompt_based_weight > 0
                else MockRewardModel(RewardModelType.prompt.value),
            ]

            if len(self.reward_functions) != len(self.reward_weights):
                message = (
                    f"Length of reward function weights and reward functions do not match. "
                    f"Reward functions: {len(self.reward_functions)}, Reward weights: {len(self.reward_weights)}"
                )

                bt.logging.error(message)
                raise Exception(message)
            
            # Masking functions
            self.blacklist = (
                Blacklist() if not self.config.neuron.blacklist_off else MockRewardModel(RewardModelType.blacklist.value)
            )
            task_validator = (
                TaskValidator() if not self.config.neuron.task_validator_off
                else MockRewardModel(RewardModelType.task_validator.value)
            )
            relevance_model = (
                RelevanceRewardModel(device=self.device) if not self.config.neuron.relevance_off
                else MockRewardModel(RewardModelType.relevance.value)
            )
            self.diversity_model = (
                DiversityRewardModel(device=self.device) if not self.config.neuron.diversity_off
                else MockRewardModel(RewardModelType.diversity.value)
            )
            nsfw_model = (
                NSFWRewardModel(device=self.device) if not self.config.neuron.nsfw_off
                else MockRewardModel(RewardModelType.nsfw.value)              
            )

            self.masking_functions = [self.blacklist, task_validator, relevance_model, self.diversity_model, nsfw_model]
            bt.logging.debug(str(self.reward_functions))
            bt.logging.debug(str(self.masking_functions))

        # Init the event loop.
        self.loop = asyncio.get_event_loop()

        # Init wandb.
        if not self.config.wandb.off:
            bt.logging.debug("loading", "wandb")
            init_wandb(self)

        if self.config.neuron.epoch_length_override:
            self.config.neuron.epoch_length = self.config.neuron.epoch_length_override
        else:
            self.config.neuron.epoch_length = self.subtensor.validator_epoch_length(self.config.netuid)

        self.prev_block = ttl_get_block(self)
        self.step = 0


def main():
    neuron().run()


if __name__ == "__main__":
    main()
