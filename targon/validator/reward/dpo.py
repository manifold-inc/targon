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
from .reward import BaseRewardModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class DirectPreferenceRewardModel(BaseRewardModel):

    reward_model_name: str = "cerebras/btlm-3b-8k-base"

    @property
    def name(self) -> str: return RewardModelType.dpo.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.penalty = 1.2 # Same penalty as the original [paper](https://arxiv.org/pdf/1909.05858.pdf).
        self.tokenizer = AutoTokenizer.from_pretrained(DirectPreferenceRewardModel.reward_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(DirectPreferenceRewardModel.reward_model_name,
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.float16).to(self.device)

    def reward_single(self, prompt: str, completion: str, name: str ,with_penalty=True) -> float:
        r""" Calculates a direct preference optimization (DPO) style reward for a completion,
        which is a reference model's average log-probability for completion tokens given a prompt.
        Uses guidance from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py.
        """
        with torch.no_grad():

            # Check if completion is 
            if completion.strip() == '' or len(completion) <= 5:
                return -11 # exp(-11)=1.67e-5 < 2e-5=1/50257 (typical vocab size)
            
            # Tokenize the combined prompt + completion.
            combined = self.tokenizer(prompt + completion, return_tensors="pt").input_ids[0].to(self.device)  # [seq_len]
            # Tokenize only the prompt, to help determine prompt token length.
            prompt_part = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)  # [prompt_len]

            # Completion doesn't fit into model sequence, so return lowest reward.
            if self.tokenizer.model_max_length <= len(prompt_part):
                return -11.  # exp(-11)=1.67e-5 < 2e-5=1/50257 (typical vocab size)

            # Truncate combined to fit into model max sequence length.
            if self.tokenizer.model_max_length < len(combined):
                combined = combined[:self.tokenizer.model_max_length]

            labels = combined.clone()  # [seq_len]
            # Ignore prompt part for calculating reward.
            labels[:len(prompt_part)] = -100
            # Label only each next token prediction ground-truth.
            labels = labels[1:]  # [seq_len-1]
            loss_mask = (labels != -100)  # [seq_len-1]
            # Dummy token to allow for indexing, but loss will be ignored.
            labels[labels == -100] = 0
            # Reshape for gather operation.
            labels = labels.unsqueeze(0).unsqueeze(2)  # [batch_size=1, seq_len-1, :]

            # Forward pass to calculate logit predictions for each sequence position.
            logits = self.model(combined.unsqueeze(0)).logits  # [batch_size=1, seq_len, vocab_len]
            # Predict only where labels are available.
            logits = logits[:, :-1, :]  # [batch_size=1, seq_len-1, vocab_len]

            if with_penalty:
                # Apply penalty for repeated generation
                for i in range(len(prompt_part)+1, len(combined)-1):
                    logit = logits[:,i,:].clone()
                    inputs = combined[len(prompt_part):i].clone()
                    logits[:,i,:] =  self.logit_penalty(input_ids=inputs, logit=logit)

            # Rescale via log(softmax(logits)).
            logits = logits.log_softmax(-1)
            # Calculate the model's log-probability for each actual completion token.
            per_token_logps = torch.gather(logits, dim=2, index=labels).squeeze(2)  # [batch_size=1, seq_len-1]
            # Average log-probability over completion sequence.
            reward = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)  # [batch_size=1]
            reward = reward[0].cpu().detach()

            # NaNs can possibly arise through log(0)=-inf, replace with suitably small logits.
            if torch.isnan(reward) or torch.isinf(reward):
                return -11.  # exp(-11)=1.67e-5 < 2e-5=1/50257 (typical vocab size)
            return reward.item()
        
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