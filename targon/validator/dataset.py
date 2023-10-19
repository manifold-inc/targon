import random
import bittensor as bt
from datasets import load_dataset
from collections.abc import Iterator

class Dataset(Iterator):
    def __init__(self):
        super().__init__()
        seed = random.randint(0,1000)
        self.openwebtext = iter( load_dataset("openwebtext", split="train", streaming=True).shuffle(seed=seed, buffer_size=10000) )
        self.red_pajama = iter( load_dataset("cerebras/SlimPajama-627B", 'default', split='train', streaming=True).shuffle(seed=seed, buffer_size=10000) )

    def __next__(self):         
         while True:
            bt.logging.debug('Retrieving data from dataset...')
            if random.random() < 0.5:
                text = next(self.openwebtext)["text"]
            else:
                text = next(self.red_pajama)["text"]

            # Check if the text is not empty or does not consist only of newline characters
            if text.strip():
                return {"text": text}


class MockDataset(Iterator):
    def __next__(self):
        return {"text": "What is the capital of Texas?"}


   