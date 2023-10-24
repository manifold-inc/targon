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
import numpy as np
from typing import List
from .config import RewardModelType
from .base import BaseRewardModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from targon.prompts import tasks

class CorrectnessRewardSignal(BaseRewardModel):

    reward_model_name: str = "google/flan-t5-large"

    @property
    def name(self) -> str: return RewardModelType.correctness.value

    def __init__(self, device: str, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

        self.yes_token_id = 2163  # this is for Flan-T5, change it accordingly
        self.no_token_id = 465  # this is for Flan-T5, change it accordingly



    def build_input_text(self, prompt: str, completion: str, flavor: str, solution: str) -> str:
        # find the flavor in the taks list
        input_text = f"Question: {prompt}\n\nAnswer: {completion}\n\nSolution: {solution}\n\n Is the answer correct based off the solution? Response:"

        return input_text

    def reward_single(self, prompt: str, completion: str, name: str, solution: str) -> float:
        with torch.no_grad():
            # Validate input
            if not prompt or not completion or not name or not solution:
                return 0.0  # or some other default value

            input_text = self.build_input_text(prompt, completion, name, solution)
            
            # Validate tokenized input
            x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(self.device)

            outputs = self.model.generate(
                x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
            )
            p_yes = torch.exp(outputs.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
            p_no = torch.exp(outputs.scores[0][:, self.no_token_id]).cpu().numpy()[0]
            
            # Validate p_yes and p_no
            if np.isnan(p_yes) or np.isnan(p_no):
                return 0.0  # or some other default value

            # Validate denominator
            denominator = p_yes + p_no
            if denominator == 0:
                return 0.0  # or some other default value

            score = (p_no / denominator - 0.5) * 10
            if np.isnan(score):
                return 0.0
            return float(score)
            
    def get_rewards(self, prompt: str, completions: List[str], name: str, solution: str) -> torch.FloatTensor:
        rewards = torch.tensor([self.reward_single(prompt, completion, name, solution) for completion in completions],
                               dtype=torch.float32).to(self.device)
        bt.logging.trace(f"{name} correctness signal | rewards: {rewards.tolist()}")
        return rewards
