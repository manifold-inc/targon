import os
import time
import torch
import openai
import shutil
import asyncio
import tempfile
import argparse
import bittensor as bt

from threading import Thread
from functools import partial
from starlette.types import Send
from targon.miner.miner import Miner 
from transformers import GPT2Tokenizer
from typing import List, Optional, Union
from targon.protocol import TargonStreaming
from transformers import TextIteratorStreamer
from torchvision.transforms import ToPILImage, Resize, Compose
from transformers import StoppingCriteria, StoppingCriteriaList


class SybilMiner( Miner ):
    def config(self) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Sybil Configs")
        self.add_args(parser)
        return bt.config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds custom arguments to the command line parser.

        Developers can introduce additional command-line arguments specific to the miner's
        functionality in this method. These arguments can then be used to configure the miner's operation.

        Args:
            parser (argparse.ArgumentParser):
                The command line argument parser to which custom arguments should be added.
        """
        parser.add_argument('--sybil.model', type=str, default="teknium/OpenHermes-2-Mistral-7B", help='Model to use for generation.')
        parser.add_argument('--sybil.max_new_tokens', type=int, default=300, help='Maximum number of tokens to generate.')
        parser.add_argument('--sybil.num_beams', type=int, default=1, help='Number of beams to use for beam search.')
        parser.add_argument('--sybil.min_length', type=int, default=1, help='Minimum number of tokens to generate.')
        parser.add_argument('--sybil.top_p', type=float, default=0.9, help='Top p for nucleus sampling.')
        parser.add_argument('--sybil.repetition_penalty', type=float, default=1.0, help='Repetition penalty.')
        parser.add_argument('--sybil.length_penalty', type=float, default=1.0, help='Length penalty.')
        parser.add_argument('--sybil.temperature', type=float, default=1.0, help='Temperature for sampling.')
        parser.add_argument('--sybil.max_length', type=int, default=2000, help='Maximum number of tokens to generate.')
        parser.add_argument('--sybil.device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run the model on.')
        parser.add_argument('--sybil.api_url', type=str, default="http://0.0.0.08000", help='URL for the API server.')

    def __init__(self, *args, **kwargs):
        super(SybilMiner, self).__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # get the directory this file is in
        base_path = os.path.dirname(os.path.realpath(__file__))

        openai.api_base = self.config.sybil.api_url
        openai.api_key = 'asdasd'



    def prompt(self, synapse: TargonStreaming) -> TargonStreaming:
        """
        Generates a streaming response for the provided synapse.

        This function serves as the main entry point for handling streaming prompts. It takes
        the incoming synapse which contains messages to be processed and returns a streaming
        response. The function uses the GPT-2 tokenizer and a simulated model to tokenize and decode
        the incoming message, and then sends the response back to the client token by token.

        Args:
            synapse (TargonStreaming): The incoming TargonStreaming instance containing the messages to be processed.

        Returns:
            TargonStreaming: The streaming response object which can be used by other functions to
                            stream back the response to the client.

        Usage:
            This function can be extended and customized based on specific requirements of the
            miner. Developers can swap out the tokenizer, model, or adjust how streaming responses
            are generated to suit their specific applications.
        """



        async def _prompt(messages: str, send: Send):
            """
            Asynchronously processes the input text and sends back tokens as a streaming response.

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
            """
            messages = [{"role": role, "content": message} for role, message in zip(synapse.roles, synapse.messages)]

            results_generator = openai.ChatCompletion.create(
                model=self.config.sybil.model,
                messages=messages,
                temperature=0.7,
                stream=True  # this time, we set stream=True
            )
            

            bt.logging.info(f"results generator {results_generator}")

            buffer = []
            output_text = ""
            for token in results_generator:
                output_text += token

                
                N = 3  # Number of tokens to send back to the client at a time
                buffer.append(token)
                # If buffer has N tokens, send them back to the client.
                if len(buffer) == N:
                    joined_buffer = "".join(buffer)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    bt.logging.debug(f"Streamed tokens: {joined_buffer}")
                    buffer = []  # Clear the buffer for next batch of tokens
                    # await asyncio.sleep(0.08)  # Simulate streaming delay
            
            # Send any remaining tokens in the buffer
            if buffer:
                joined_buffer = "".join(buffer)
                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": False,  # No more tokens to send
                    }
                )
                bt.logging.trace(f"Streamed tokens: {joined_buffer}")

        # message = synapse.messages[0]
        last_text  = synapse.messages[-1]
        token_streamer = partial(_prompt, last_text)
        return synapse.create_streaming_response(token_streamer)



if __name__ == "__main__":
    """
    Entry point for executing the StreamingTemplateMiner.

    This block initializes the StreamingTemplateMiner and runs it, effectively connecting
    it to the Bittensor network. Once connected, the miner will continuously listen for
    incoming requests from the Bittensor network. For every request, it responds with a
    static message processed as per the logic defined in the 'prompt' method of the
    StreamingTemplateMiner class.

    The main loop at the end serves to keep the miner running indefinitely. It periodically
    prints a "running..." message to the console, providing a simple indication that the miner
    is operational and active.

    Developers looking to extend or customize the miner's behavior can modify the
    StreamingTemplateMiner class and its methods. However, this block itself usually
    remains unchanged unless there's a need for specific startup behaviors or configurations.

    To start the miner:
    Simply execute this script. Ensure all dependencies are properly installed and network
    configurations are correctly set up.
    """
    bt.debug()
    with SybilMiner():
        while True:
            print("running...", time.time())
            time.sleep(1)
