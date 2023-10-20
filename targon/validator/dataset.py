import random
import bittensor as bt
from datasets import load_dataset
from collections.abc import Iterator

class Dataset(Iterator):
    def __init__(self):
        super().__init__()
        seed = random.randint(0,1000)
        self.red_pajama = iter( load_dataset("togethercomputer/RedPajama-Data-1T", 'default', split='train', streaming=True).shuffle(seed=seed, buffer_size=100000) )

    def __next__(self):         
        #  while True:
        bt.logging.debug('Retrieving data from dataset...')

        text = next(self.red_pajama)["text"]
        bt.logging.debug('Retrieved data from dataset.')
        # Check if the text is not empty or does not consist only of newline characters
        if text.strip():
            return {"text": text}


class MockDataset(Iterator):
    def __next__(self):
        return {
            "text": '''Asynchronously processes the input text and sends back tokens as a streaming response.

This function takes an input text, tokenizes it using the GPT-2 tokenizer, and then
uses the simulated model to decode token IDs into strings. It then sends each token
back to the client as a streaming response, with a delay between tokens to simulate
the effect of real-time streaming.

Args:
    text (str): The input text message to be processed.
    send (Send): An asynchronous function that allows sending back the streaming response.

Usage:
    This function can be adjusted based on the streaming requirements, speed of
    response, or the model being used. Developers can also introduce more sophisticated
    processing steps or modify how tokens are sent back to the client.
'''
        }


   