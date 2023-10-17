import os
import time
import json
import torch
import openai
import shutil
import asyncio
import requests
import tempfile
import argparse
import bittensor as bt

from threading import Thread
from functools import partial
from starlette.types import Send
from targon.miner.miner import Miner 
from transformers import GPT2Tokenizer
from targon.protocol import TargonStreaming
from transformers import TextIteratorStreamer
from typing import List, Optional, Union, Iterable
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


    def post_http_request(self,
                        prompt: str,
                        api_url: str,
                        n: int = 1,
                        stream: bool = False) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        pload = {
            "prompt": prompt,
            "n": n,
            "use_beam_search": True,
            "temperature": 0.0,
            "max_tokens": 16,
            "stream": stream,
        }
        response = requests.post(api_url, headers=headers, json=pload, stream=True)
        return response


    def get_streaming_response(self, response: requests.Response) -> Iterable[List[str]]:
        for chunk in response.iter_lines(chunk_size=8192,
                                        decode_unicode=False,
                                        delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"]
                yield output

    def get_response(self, response: requests.Response) -> List[str]:
        data = json.loads(response.content)
        output = data["text"]
        return output

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

        messages = [{"role": role, "content": message} for role, message in zip(synapse.roles, synapse.messages)]


        def get_prompt(messages: List[dict]) -> str:
            """
            Constructs the prompt for the GPT-2 model.

            This function takes the incoming messages and constructs a prompt for the GPT-2 model.
            It uses the GPT-2 tokenizer to tokenize the messages and then decodes the token IDs
            into strings. It then constructs the prompt by concatenating the decoded messages
            together.

            Args:
                messages (List[dict]): The incoming messages to be processed.

            Returns:
                str: The prompt to be used for the GPT-2 model.

            Usage:
                This function can be adjusted based on the model being used. Developers can also
                introduce more sophisticated processing steps or modify how the prompt is
                constructed.
            """
            prompt = ""
            for message in messages:
                if message["role"] == "system":
                    prompt += f'''<imstart>system
{message["content"]}<|im_end|>'''
                elif message["role"] == "user":
                    prompt += f'''<imstart>user
{message["content"]}<|im_end|>'''
                elif message["role"] == "assistant":
                    prompt += f'''<imstart>assistant
{message["content"]}<|im_end|>'''
                else:
                    raise ValueError(f"Invalid role: {message['role']}")

        async def _prompt(prompt: str, send: Send):
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


            response = self.post_http_request(prompt, self.config.sybil.api_url, n=1, stream=True)

            if not synapse.stream:
                joined_buffer = self.get_response(response)
                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": False,  # No more tokens to send
                    }
                )
            else:

                buffer = []
                output_text = ""
                for token in self.get_streaming_response(response):
                    output_text += token
                    bt.logging.info(f"token", token)
                    
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
        prompt = get_prompt(messages)
        token_streamer = partial(_prompt, prompt)
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
