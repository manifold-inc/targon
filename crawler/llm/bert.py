from transformers import BertModel, BertTokenizer
import torch
from llm.base import BaseLanguageModel

MODEL_NAME = "bert-base-uncased"
MAX_TOKEN_LEN = 512
EMBED_SIZE = 768


class BertLanguageModel(BaseLanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertModel.from_pretrained(MODEL_NAME).to(device)

    @property
    def max_token_length(self):
        return MAX_TOKEN_LEN

    @property
    def embed_size(self):
        return EMBED_SIZE
