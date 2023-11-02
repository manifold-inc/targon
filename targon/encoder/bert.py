from transformers import BertModel, BertTokenizer

from .base import BaseEncoder

MODEL_NAME = "bert-base-uncased"
MAX_TOKEN_LEN = 512
EMBED_SIZE = 768


class BertEncoder(BaseEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertModel.from_pretrained(MODEL_NAME)

    @property
    def max_token_length(self):
        return MAX_TOKEN_LEN

    @property
    def embed_size(self):
        return EMBED_SIZE