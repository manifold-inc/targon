import random
import bittensor as bt
from datasets import load_dataset
from collections.abc import Iterator

from targon.prompts import process_javascript, process_python, javascript_prompt

class CodingDataset(Iterator):
    def __init__(self):
        super().__init__()
        seed = random.randint(0, 10000)

        ## coding
        self.python = iter( load_dataset("Nan-Do/reason_code-search-net-python", split="train", streaming=True).shuffle(seed=seed, buffer_size=100))
        self.javascript = iter( load_dataset("CM/codexglue_code2text_javascript", split="train", streaming=True).shuffle(seed=seed, buffer_size=100))
       
    def __next__(self):         
         while True:
            bt.logging.debug('Retrieving data from dataset...')
            if random.random() < 0.4:
                data = next(self.javascript)
                code, docstring = process_javascript( data )
                instruction = javascript_prompt.format( code=code )
                return {"question": instruction, "task": "javascript", "solution": docstring}
            else:
                data = next(self.python)
                instruction, response = process_python( data )
                return {"question": instruction, "task": "python", "solution": response}
            # Check if the text is not empty or does not consist only of newline characters
            # if question.strip():
            


class MockCodingDataset(Iterator):
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


   