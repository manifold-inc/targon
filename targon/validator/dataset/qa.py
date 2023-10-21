import random
import bittensor as bt
from datasets import load_dataset
from collections.abc import Iterator

from targon.prompts import process_math_qa, process_open_orca, math_qa_prompt

class QADataset(Iterator):
    def __init__(self):
        super().__init__()
        seed = random.randint(0, 10000)

        ## QA
        self.open_orca = iter( load_dataset("Open-Orca/OpenOrca", split="train", streaming=True) .shuffle(seed=seed, buffer_size=100))
        self.math_qa = iter( load_dataset("math_qa", split="train", streaming=True).shuffle(seed=seed, buffer_size=100))

    def __next__(self):         
         while True:
            bt.logging.debug('Retrieving data from dataset...')
            if random.random() < 0.2:
                data = next(self.open_orca)
                problem, options, rationale, correct_option = process_math_qa( data )
                question = math_qa_prompt.format( problem=problem, options=options )
                return {"question": question, "task": "math_qa", "solution": rationale}
            else:
                data = next(self.math_qa)
                system_prompt, question, response = process_open_orca( data )
                if system_prompt is not None:
                    question = system_prompt + "\n\n" + question
                return {"question": question, "task": "open_orca", "solution": response}

            # Check if the text is not empty or does not consist only of newline characters
            # if question.strip():
            


class MockQADataset(Iterator):
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


   