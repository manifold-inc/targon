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

import bittensor as bt
from dataclasses import dataclass
from typing import List, Optional

from targon.validator.reward import RewardModelType


@dataclass
class EventSchema:
    completions: List[str]  # List of completions received for a given prompt
    completion_times: List[float]  # List of completion times for a given prompt
    name: str  # Prompt type, e.g. 'followup', 'answer'
    block: float  # Current block at given step
    gating_loss: float  # Gating model loss for given step
    uids: List[int]  # Queried uids
    prompt: str  # Prompt text string
    step_length: float  # Elapsed time between the beginning of a run step to the end of a run step
    best: str  # Best completion for given prompt

    # Reward data
    rewards: List[float]  # Reward vector for given step
    dahoas_reward_model: Optional[List[float]] # Output vector of the dahoas reward model
    blacklist_filter: Optional[List[float]]  # Output vector of the blacklist filter
    nsfw_filter: Optional[List[float]]  # Output vector of the nsfw filter
    reciprocate_reward_model: Optional[List[float]]  # Output vector of the reciprocate reward model
    diversity_reward_model: Optional[List[float]]  # Output vector of the diversity reward model
    dpo_reward_model: Optional[List[float]]  # Output vector of the dpo reward model
    rlhf_reward_model: Optional[List[float]]  # Output vector of the rlhf reward model
    prompt_reward_model: Optional[List[float]]  # Output vector of the prompt reward model
    relevance_filter: Optional[List[float]]  # Output vector of the relevance scoring reward model
    task_validator_filter: Optional[List[float]]

    dahoas_reward_model_normalized: Optional[List[float]] # Output vector of the dahoas reward model
    nsfw_filter_normalized: Optional[List[float]]  # Output vector of the nsfw filter
    reciprocate_reward_model_normalized: Optional[List[float]]  # Output vector of the reciprocate reward model
    diversity_reward_model_normalized: Optional[List[float]]  # Output vector of the diversity reward model
    dpo_reward_model_normalized: Optional[List[float]]  # Output vector of the dpo reward model
    rlhf_reward_model_normalized: Optional[List[float]]  # Output vector of the rlhf reward model
    prompt_reward_model_normalized: Optional[List[float]]  # Output vector of the prompt reward model
    relevance_filter_normalized: Optional[List[float]]  # Output vector of the relevance scoring reward model
    task_validator_filter_normalized: Optional[List[float]]
    
    # Weights data
    set_weights: Optional[List[List[float]]]

    @staticmethod
    def from_dict(event_dict: dict, disable_log_rewards: bool) -> 'EventSchema':
        """Converts a dictionary to an EventSchema object."""
        rewards = {
            'blacklist_filter': event_dict.get(RewardModelType.blacklist.value),
            'dahoas_reward_model': event_dict.get(RewardModelType.dahoas.value),
            'task_validator_filter': event_dict.get(RewardModelType.task_validator.value),
            'nsfw_filter': event_dict.get(RewardModelType.nsfw.value),
            'relevance_filter': event_dict.get(RewardModelType.relevance.value),
            'reciprocate_reward_model': event_dict.get(RewardModelType.reciprocate.value),
            'diversity_reward_model': event_dict.get(RewardModelType.diversity.value),
            'dpo_reward_model': event_dict.get(RewardModelType.dpo.value),
            'rlhf_reward_model': event_dict.get(RewardModelType.rlhf.value),
            'prompt_reward_model': event_dict.get(RewardModelType.prompt.value),
            
            'dahoas_reward_model_normalized': event_dict.get(RewardModelType.dahoas.value + '_normalized'),
            'task_validator_filter_normalized': event_dict.get(RewardModelType.task_validator.value + '_normalized'),
            'nsfw_filter_normalized': event_dict.get(RewardModelType.nsfw.value + '_normalized'),
            'relevance_filter_normalized': event_dict.get(RewardModelType.relevance.value + '_normalized'),
            'reciprocate_reward_model_normalized': event_dict.get(RewardModelType.reciprocate.value + '_normalized'),
            'diversity_reward_model_normalized': event_dict.get(RewardModelType.diversity.value + '_normalized'),
            'dpo_reward_model_normalized': event_dict.get(RewardModelType.dpo.value + '_normalized'),
            'rlhf_reward_model_normalized': event_dict.get(RewardModelType.rlhf.value + '_normalized'),
            'prompt_reward_model_normalized': event_dict.get(RewardModelType.prompt.value + '_normalized'),
        }

        # Logs warning that expected data was not set properly
        if not disable_log_rewards and any(value is None for value in rewards.values()):
            for key, value in rewards.items():
                if value is None:
                    bt.logging.warning(f'EventSchema.from_dict: {key} is None, data will not be logged')

        return EventSchema(
            completions=event_dict['completions'],
            completion_times=event_dict['completion_times'],
            name=event_dict['name'],
            block=event_dict['block'],
            gating_loss=event_dict['gating_loss'],
            uids=event_dict['uids'],
            prompt=event_dict['prompt'],
            step_length=event_dict['step_length'],
            best=event_dict['best'],
            rewards=event_dict['rewards'],
            **rewards,
            set_weights=None,
        )
