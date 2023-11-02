from .base import BaseEncoder
from .bert import BertEncoder   

ENCODER_REGISTRY = {"bert-base-uncased": BertEncoder}