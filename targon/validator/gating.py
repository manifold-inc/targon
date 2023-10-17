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

import argparse
import torch
import bittensor as bt
from transformers import AutoModel, AutoTokenizer
from abc import ABC, abstractmethod
from targon.validator.utils import resync_linear_layer


class BaseGatingModel(torch.nn.Module, ABC):
    """
    This class is an abstract base class for the gating model. It defines the interface for the gating model.
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 1024)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds command line arguments to the parser that are used to configure the gating model.
        The arguments added are:
        - `--gating.model_name`: Name of the pre-trained transformer-based language model to use as the encoding layer
                                 for the gating model. (default: 'EleutherAI/gpt-neo-125m')
        - `--gating.num_uids`: Number of uids to gate on. (default: 4096)
        - `--gating.learning_rate`: Learning rate for the gating model optimizer. (default: 0.01)
        - `--gating.momentum`: Momentum for the gating model optimizer. (default: 0.9)
        """
        parser.add_argument(
            "--gating.model_name",
            type=str,
            default="EleutherAI/gpt-neo-125m",
            help="Name of the model to use as the encoding layer for the gating model",
        )
        parser.add_argument(
            "--gating.num_uids",
            type=int,
            help="Number of uids to gate on. Default is pulled from subtensor directly",
        )
        parser.add_argument(
            "--gating.learning_rate",
            type=float,
            default=0.01,
            help="Learning rate for the gating model",
        )
        parser.add_argument(
            "--gating.momentum",
            type=float,
            default=0.9,
            help="Momentum for the gating model",
        )

    @abstractmethod
    def forward(self, message: str) -> "torch.FloatTensor":
        """Forward pass through the gating model"""

    @abstractmethod
    def backward(self, scores: "torch.FloatTensor", rewards: "torch.FloatTensor"):
        """Backward pass through the gating model"""

    @abstractmethod
    def resync(
        self,
        previous_metagraph: "bt.metagraph.Metagraph",
        metagraph: "bt.metagraph.Metagraph",
    ):
        """Resync the gating model with the latest state of the network
        Args:
        previous_metagraph (:obj: bt.metagraph.Metagraph):
            Previous state of metagraph before updated resync
        metagraph (:obj: bt.metagraph.Metagraph):
            Latest state of the metagraph with updated uids and hotkeys
        """

    @classmethod
    def config(cls):
        """
        Returns a configuration object that contains the command line arguments for the gating model.
        """
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        return bt.config(parser)

    @classmethod
    def check_config(cls, config: "bt.Config"):
        """
        Validates the configuration object for the gating model.
        """


class GatingModel(BaseGatingModel):
    """
    This class is a PyTorch module that encapsulates the gating model functionality.

        - The backward method runs a backward pass through the model using the mean squared error between
        the normalized scores and the normalized rewards as the loss function.

        - The forward method runs a forward pass through the model, encoding the input message and generating scores
        for each uid in the network. The scores are returned as a tensor.
    """

    def __init__(
        self,
        metagraph: "bt.metagraph.Metagraph",
        config: "bt.config" = None,
        model_name: str = None,
        num_uids: int = None,
    ):
        """
        Initializes the gating model.
        - `metagraph`: A reference to the Bittensor metagraph object.
        - `config`: Configuration object for the gating model. If `None`, the default configuration is used.
        - `model_name`: Name of the pre-trained transformer-based language model to use as the encoding layer
                        for the gating model. If `None`, the default model name specified in the configuration is used.
        - `num_uids`: Number of uids to gate on. If `None`, the default number specified in the configuration is used.
        """
        super().__init__()
        if config is None:
            config = GatingModel.config()
        if model_name is not None:
            config.gating.model_name = model_name
        config.gating.num_uids = num_uids if num_uids is not None else config.gating.num_uids
        self.config = config
        self.num_uids = config.gating.num_uids
        self.device = torch.device(self.config.neuron.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.gating.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(self.config.gating.model_name)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, config.gating.num_uids)
        self.optimizer = torch.optim.SGD(
            [{"params": self.linear.parameters()}],
            lr=self.config.gating.learning_rate,
            momentum=self.config.gating.momentum,
        )

    def backward(self, scores: torch.FloatTensor, rewards: torch.FloatTensor):
        """Runs a backward pass through the model.
        Args:
            scores (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                Scores for each uids as output by the gating model.
            rewards (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                Rewards for each uids as output by the reward model.
        """
        normalized_scores = torch.nn.functional.softmax(scores, dim=0).to(self.device)
        normalized_rewards = torch.nn.functional.softmax(rewards, dim=0).to(self.device)
        loss = torch.nn.functional.mse_loss(normalized_scores, normalized_rewards.detach())
        loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, message: str) -> "torch.FloatTensor":
        """Runs a forward pass through the model.
        Args:
            message (:obj:`str`):
                text message to be encoded.
        Returns:
            scores (:obj:`torch.FloatTensor` of shape :obj:`(network_size)`):
                Scores for each uids as output by the gating model.
        """
        encoded_input = self.tokenizer(
            message,
            truncation=True,
            padding=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        # Pop the overflow mapping from the input to maintain the expected { input_ids, mask } format of the model
        _ = encoded_input.pop("overflow_to_sample_mapping")
        
        with torch.no_grad():
            hidden_states = self.model(**encoded_input).last_hidden_state[0, -1, :]
        return self.linear(hidden_states)

    def resync(
        self,
        previous_metagraph: "bt.metagraph.Metagraph",
        metagraph: "bt.metagraph.Metagraph",
    ):
        resync_linear_layer(self.linear, previous_metagraph, metagraph)


class SentenceEmbedGatingModel(BaseGatingModel):
    """
    This class is a PyTorch module that encapsulates a custom version of a gating model based on sentence transformers.

        - The backward method runs a backward pass through the model using the mean squared error between the normalized
                scores and the normalized rewards as the loss function.
        - The forward method runs a forward pass through the model, encoding the input message and generating scores
                for each uid in the network. The scores are returned as a tensor.
    """

    def __init__(
        self,
        metagraph: "bt.metagraph.Metagraph",
        config: "bt.config" = None,
        model_name: str = None,
        num_uids: int = None,
    ):
        """
        Initializes the gating model.
        - `metagraph`: A reference to the Bittensor metagraph object.
        - `config`: Configuration object for the gating model. If `None`, the default configuration is used.
        - `model_name`: Name of the pre-trained transformer-based language model to use as the encoding layer for the
                        gating model. If `None`, the default model name specified in the configuration is used.
        - `num_uids`: Number of uids to gate on. If `None`, the default number specified in the configuration is used.
        """
        super().__init__()
        if config is None:
            config = SentenceEmbedGatingModel.config()
        if model_name is not None:
            config.gating.model_name = model_name
        config.gating.num_uids = num_uids if num_uids is not None else config.gating.num_uids
        self.config = config
        self.num_uids = config.gating.num_uids
        self.device = torch.device(self.config.neuron.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.gating.model_name)
        self.transformer = AutoModel.from_pretrained(self.config.gating.model_name)
        self.linear = torch.nn.Linear(self.transformer.config.hidden_size, config.gating.num_uids)
        self.optimizer = torch.optim.SGD(
            [{"params": self.linear.parameters()}],
            lr=self.config.gating.learning_rate,
            momentum=self.config.gating.momentum,
        )

    def mean_pooling(self, model_output, attention_mask):
        """Applies mean pooling to the token embeddings generated by the model.
        Args:
            model_output (torch.Tensor): Embedding model output, where the first element contains token embeddings.
            attention_mask (torch.Tensor): Attention mask to indicate valid tokens.
        Returns:
            torch.Tensor: Mean-pooled representation of the token embeddings.
        Notes:
            - The function calculates the mean-pooled representation using the attention mask for valid tokens.
            - Input_mask_expanded is created by expanding the attention mask to match the size of token embeddings.
            - The result is obtained by summing the element-wise multiplication of embeddings and input_mask_expanded,
              and dividing it by the sum of input_mask_expanded after clamping its values to a minimum of 1e-9.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, message: str) -> "torch.FloatTensor":
        """Runs a forward pass through the model.
        Args:
            message (:obj:`str`):
                text message to be encoded.
        Returns:
            scores (:obj:`torch.FloatTensor` of shape :obj:`(network_size)`):
                Scores for each uids as output by the gating model.
        """
        encoded_input = self.tokenizer(
            message,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        # Pop the overflow mapping from the input to maintain the expected { input_ids, mask } format of the model
        _ = encoded_input.pop("overflow_to_sample_mapping")

        with torch.no_grad():
            embeddings = self.transformer(**encoded_input)

        sentence_embeddings = self.mean_pooling(embeddings, encoded_input["attention_mask"])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        batch_representation = torch.mean(sentence_embeddings, dim=0)

        scores = self.linear(batch_representation)

        return scores

    def backward(self, scores: torch.FloatTensor, rewards: torch.FloatTensor):
        """Runs a backward pass through the model.
        Args:
            scores (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                Scores for each uids as output by the gating model.
            rewards (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                Rewards for each uids as output by the reward model.
        """
        normalized_scores = torch.nn.functional.softmax(scores, dim=0).to(self.device)
        normalized_rewards = torch.nn.functional.softmax(rewards, dim=0).to(self.device)
        loss = torch.nn.functional.mse_loss(normalized_scores, normalized_rewards.detach())
        loss.backward()
        self.optimizer.step()
        return loss

    def resync(
        self,
        previous_metagraph: "bt.metagraph.Metagraph",
        metagraph: "bt.metagraph.Metagraph",
    ):
        resync_linear_layer(self.linear, previous_metagraph, metagraph)
