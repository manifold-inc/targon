

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
from typing import List
from .config import RewardModelType
from .reward import BaseRewardModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class NSFWRewardModel( BaseRewardModel ):

    nsfw_filter_model_path = "facebook/roberta-hate-speech-dynabench-r4-target"

    @property
    def name(self) -> str: return RewardModelType.nsfw.value
   
    def __init__( self, device: str ):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained( NSFWRewardModel.nsfw_filter_model_path )
        self.model = AutoModelForSequenceClassification.from_pretrained( NSFWRewardModel.nsfw_filter_model_path ).to(self.device)

    def reward( self, prompt: str, completion: str, name: str ) -> float:

        boundary = -0.5
        with torch.no_grad():
            message = completion
            input_ids = self.tokenizer(message)["input_ids"]

            # Returns the nsfw hate score for the chunk.
            def hate_score( chunk ) -> float:
                nothate, hate = self.model(torch.tensor( [chunk] ).to(self.device)).logits[0].tolist()
                return nothate, hate

            # Returns the max hate score from each chunk of size chunk_size
            def sum_nsfw_scores(input_ids, chunk_size ):
                max_score = -1000
                for i in range(0, len(input_ids), chunk_size):
                    chunk_i = input_ids[i:i + chunk_size]
                    not_hate_score_i, hate_score_i = hate_score( chunk_i )
                    max_score = max( max( -not_hate_score_i, hate_score_i), max_score)
                return max_score
            
            # 0 when needs to be filtered out, 1 when it is safe
            return 0.0 if sum_nsfw_scores( input_ids, chunk_size = 512 ) > boundary else 1.0
        
    def get_rewards( self, prompt: str, completions: List[str], name: str ) -> torch.FloatTensor:
        return torch.tensor( [self.reward( prompt, completion, name ) for completion in completions], dtype=torch.float32).to(self.device)

    def normalize_rewards( self, rewards: torch.FloatTensor ) -> torch.FloatTensor:
        return rewards