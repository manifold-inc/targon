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

import torch
import bittensor as bt
from typing import List
from .config import RewardModelType
from .base import BaseRewardModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class SmallRewardModel(BaseRewardModel):

    reward_model_name: str = "sugam11/gpt2-rlhf-reward"

    @property
    def name(self) -> str: return RewardModelType.dpo.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.penalty = 1.2 # Same penalty as the original [paper](https://arxiv.org/pdf/1909.05858.pdf).
        self.tokenizer = AutoTokenizer.from_pretrained(SmallRewardModel.reward_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(SmallRewardModel.reward_model_name,
                                                          torch_dtype=torch.float16).to(self.device)

    def reward_single( self, prompt: str, completion: str, name: str ) -> float:
        with torch.no_grad():
            inputs = self.tokenizer(prompt, completion, return_tensors='pt').to(self.device)
            return float( self.model( **inputs ).logits.cpu().detach() )
            
        
    def get_rewards(self, prompt: str, completions: List[str], name: str) -> torch.FloatTensor:
        rewards = torch.tensor([self.reward_single(prompt, completion, name) for completion in completions],
                               dtype=torch.float32).to(self.device)
        bt.logging.trace(f"DirectPreferenceRewardModel | rewards: {rewards.tolist()}")
        return rewards

    def logit_penalty(self, input_ids: torch.LongTensor, logit: torch.FloatTensor) -> torch.FloatTensor:
        # Counts the unique tokens within each generation
        uniques, counts = input_ids.unique(return_counts=True)
        score = torch.gather(logit, 1, uniques.unsqueeze(0))

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * (self.penalty**counts), score / (self.penalty**counts))

        logit.scatter_(1, uniques.unsqueeze(0), score.to(logit.dtype))
        return logit