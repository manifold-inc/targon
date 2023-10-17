
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ReciprocateRewardModel( BaseRewardModel ):

    reward_model_path: str = "reciprocate/gpt-j_rm_format-oa"
    revision: str = "501f895"

    @property
    def name(self) -> str: return RewardModelType.reciprocate.value

    def __init__( self, device: str ):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained( ReciprocateRewardModel.reward_model_path, revision = ReciprocateRewardModel.revision )
        self.model = AutoModelForSequenceClassification.from_pretrained( ReciprocateRewardModel.reward_model_path,
                                                                         revision = ReciprocateRewardModel.revision,
                                                                         torch_dtype=torch.float16).to(self.device)

    def reward( self, prompt: str, completion: str, name: str ) -> float:
        with torch.no_grad():
            message = f"<|prompter|>{prompt}</s><|assistant|>{completion}</s><|endoftext|>"
            inputs = self.tokenizer( message,
                                     return_tensors="pt" ,
                                     truncation=True,
                                     ).to(self.device)
            return float( self.model( **inputs )[0].item() )
        
    def get_rewards( self, prompt: str, completions: List[str], name: str ) -> torch.FloatTensor:
        return torch.tensor( [self.reward( prompt, completion, name ) for completion in completions], dtype=torch.float32).to(self.device)
        
