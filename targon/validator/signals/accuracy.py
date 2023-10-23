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
import numpy as np
import bittensor as bt
from typing import List
from .config import RewardModelType
from .base import BaseRewardModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from targon.prompts import tasks

class AccuracyRewardSignal(BaseRewardModel):

    reward_model_name: str = "google/flan-t5-large"

    @property
    def name(self) -> str: return RewardModelType.accuracy.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(AccuracyRewardSignal.reward_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(AccuracyRewardSignal.reward_model_name,
                                                          torch_dtype=torch.float16).to(self.device)

        self.sentiment_fn = pipeline(
            "sentiment-analysis",
            "facebook/roberta-hate-speech-dynabench-r4-target",
            top_k=2,
            truncation=True,
            batch_size=256,
            device=self.device,
            function_to_apply="none",
        )


        self.yes_token_id = 2163  # this is for Flan-T5, change it accordingly
        self.no_token_id = 465  # this is for Flan-T5, change it accordingly



    def build_input_text(self, prompt: str, completion: str, flavor: str, solution: str) -> str:
        # find the flavor in the taks list
        try:
            task_name = [task for task in tasks if flavor in task['flavor']][0]['name']

            if task_name == 'coding':
                input_text = f"Question: {prompt}\n\nAnswer: {completion}\n\n Does this {flavor} code solve the problem? Response:"
            elif task_name == 'qa':
                input_text = f"Question: {prompt}\n\nAnswer: {completion}\n\n Does this {flavor} answer the question? Response:"
            elif task_name == 'reasoning':
                input_text = f"Question: {prompt}\n\nAnswer: {completion}\n\n Does this {flavor} answer the question? Response:"
            else:
                raise ValueError(f"Unknown task name: {task_name}")
            
            return input_text
        except:
            return f"Question: {prompt}\n\nAnswer: {completion}\n\n Does this answer the question? Response:"
    
    def reward_single( self, prompt: str, completion: str, name: str, solution: str ) -> float:
        with torch.no_grad():
            input_text = self.build_input_text(prompt, completion, name, solution)
            x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(
                self.device
            )

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
            return float( score )
            
        
    def get_rewards(self, prompt: str, completions: List[str], name: str, solution: str) -> torch.FloatTensor:
        rewards = torch.tensor([self.reward_single(prompt, completion, name, solution) for completion in completions],
                               dtype=torch.float32).to(self.device)
        bt.logging.trace(f"{name} accuracy signal | rewards: {rewards.tolist()}")
        return rewards

