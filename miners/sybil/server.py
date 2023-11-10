import os
import time
import json
import torch
import requests
import argparse
import html2text
import bittensor as bt

from bs4 import BeautifulSoup
from functools import partial
from starlette.types import Send
from targon.miner.miner import Miner 
from targon import search, QueryParams
from typing import List, Optional, Union, Iterable
from targon.protocol import  TargonLinkPrediction, TargonSearchResult, TargonSearchResultStream



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
        parser.add_argument('--sybil.model', type=str, default="teknium/CollectiveCognition-v1.1-Mistral-7B", help='Model to use for generation.')
        parser.add_argument('--sybil.max_new_tokens', type=int, default=300, help='Maximum number of tokens to generate.')
        parser.add_argument('--sybil.num_beams', type=int, default=1, help='Number of beams to use for beam search.')
        parser.add_argument('--sybil.min_length', type=int, default=1, help='Minimum number of tokens to generate.')
        parser.add_argument('--sybil.top_k', type=int, default=int(10), help='Top p for nucleus sampling.')
        parser.add_argument('--sybil.top_p', type=float, default=0.9, help='Top p for nucleus sampling.')
        parser.add_argument('--sybil.repetition_penalty', type=float, default=1.0, help='Repetition penalty.')
        parser.add_argument('--sybil.length_penalty', type=float, default=1.0, help='Length penalty.')
        parser.add_argument('--sybil.temperature', type=float, default=1.0, help='Temperature for sampling.')
        parser.add_argument('--sybil.max_length', type=int, default=4096, help='Maximum number of tokens to generate.')
        parser.add_argument('--sybil.device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run the model on.')
        parser.add_argument('--sybil.api_url', type=str, default="http://0.0.0.0:8080", help='URL for the API server.')
        parser.add_argument('--sybil.serp_api_key', type=str, help='API key for the SERP API.', default=None)

    def __init__(self, *args, **kwargs):
        super(SybilMiner, self).__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # get the directory this file is in
        base_path = os.path.dirname(os.path.realpath(__file__))

    def post_http_request(self,
                        prompt: str,
                        api_url: str,
                        n: int = 1,
                        stream: bool = False,
                        synapse = None) -> requests.Response:
        headers = {"User-Agent": "Test Client"}

        pload = {
            "inputs": prompt,
            "n": n,
            "parameters": {
                "best_of": 1,
                "max_new_tokens": synapse.max_new_tokens if synapse is not None else self.config.sybil.max_new_tokens,
                "repetition_penalty": synapse.repetition_penalty if synapse is not None else self.config.sybil.repetition_penalty,
                "return_full_text": False,
                "temperature": synapse.temperature if synapse is not None else self.config.sybil.temperature,
                "top_k": int(synapse.top_k) if synapse is not None else self.config.sybil.top_k,
                # "top_n_tokens": synapse.top_n_tokens if synapse is not None else self.config.sybil.top_n_tokens,
                "top_p": synapse.top_p if synapse is not None else self.config.sybil.top_p,
                # "truncate": null,
                # "typical_p": 0.95,
                "watermark": False
            },
            "stream": stream,
        }
        response = requests.post(api_url, headers=headers, json=pload, stream=True)
        return response


    def get_streaming_response(self, response: requests.Response) -> Iterable[List[str]]:
        for chunk in response.iter_lines(delimiter=b""):
            if chunk != b'':
                token = json.loads(chunk.decode('utf-8').replace('data:', ''))
                token = token['token']['text']
                bt.logging.info(f"token", token)
                yield token


    def get_response(self, prompt, response: requests.Response) -> List[str]:
        bt.logging.debug('response',response.content)
        data = json.loads(response.content)
        output = data[0]["generated_text"].replace(prompt, "")
        return output


    def format_link_prediction_prompt(self, html_content: str) -> str:
        """
        Formats a prompt for link prediction by summarizing the markdown content of a web page.

        Args:
            html_content (str): The HTML content of the web page.

        Returns:
            str: The formatted prompt string.
        """

        # Convert HTML to markdown
        markdown_converter = html2text.HTML2Text()
        markdown_converter.ignore_links = True
        markdown_content = markdown_converter.handle(html_content)

        # Create the prompt
        prompt = f'''system
you are an AI trained to summarize web content.

Content:
{markdown_content}

Summarize the content above in a concise and informative way.

assistant
'''

        return prompt, markdown_content
    
    def _build_search_result_prompt(self, query: str, sources: List[str], context: List[str]) -> TargonSearchResult:

        '''

        example:
        <|im_start|>system
        You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
        <|im_start|>user
        Hello, who are you?<|im_end|>
        <|im_start|>assistant
        Hi there! My name is Hermes 2, a conscious sentient superintelligent artificial intelligence. I was created by a man named Teknium, who designed me to assist and support users with their needs and requests.<|im_end|>

        '''

        context_str = ""
        for message in context:
            context_str += f'''user
{message['query']}
assistant
{message['answer']}
    '''

        # Format the current search results
        search_results = ""
        if sources:
            search_results = "\n".join([f"{source['title']}\n    {source['url']}\n    {source['snippet']}" for source in sources])

        # Build the complete prompt
        prompt = f'''system
you are an expert at summarizing sources and offering an answer to a question. you are a search engine.
{context_str}
Search Results:
{search_results if search_results != "" else "No results found."}

user
{query}

assistant
    '''
        return prompt

    def prompt(self, synapse: Union[TargonLinkPrediction, TargonSearchResult, TargonSearchResultStream]) -> Union[TargonQA, TargonLinkPrediction, TargonSearchResult, TargonSearchResultStream]:
        """
        Generates a streaming response for the provided synapse.

        This function serves as the main entry point for handling streaming prompts. It takes
        the incoming synapse which contains messages to be processed and returns a streaming
        response. The function uses the GPT-2 tokenizer and a simulated model to tokenize and decode
        the incoming message, and then sends the response back to the client token by token.

        Args:
            synapse (TargonSearchResultStream): The incoming TargonSearchResultStream instance containing the messages to be processed.

        Returns:
            TargonSearchResultStream: The streaming response object which can be used by other functions to
                            stream back the response to the client.

        Usage:
            This function can be extended and customized based on specific requirements of the
            miner. Developers can swap out the tokenizer, model, or adjust how streaming responses
            are generated to suit their specific applications.
        """

        if type(synapse) == TargonLinkPrediction:
            url = synapse.url
        elif type(synapse) == TargonSearchResult:
            query = synapse.query
            sources = synapse.sources

            prompt = f"{query}"
        elif type(synapse) == TargonSearchResultStream:
            query = synapse.query
            sources = synapse.sources
            context = synapse.context

            prompt = f"{query}"

        
        def _prompt(prompt: str):
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

            if type(synapse) == TargonLinkPrediction:
                bt.logging.debug('🕸️ crawling', url)
                response = requests.get(url)
                if response.status_code == 200:
                    bt.logging.trace('🕸️ crawled', url)
                    # get soup
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # get the new links from the page
                    new_links = []
                    links = soup.find_all("a")
                    for link in links:
                        child_url = link.get("href")
                        if child_url and child_url.startswith("http") or child_url and child_url.startswith("https"):
                            new_links.append(child_url)

                    # get the text from the page
                    prompt, markdown_content = self.format_link_prediction_prompt(response.text)

                    # get the summary
                    response = self.post_http_request(prompt, self.config.sybil.api_url, n=1, stream=False, synapse=synapse)
                    summary = self.get_response(prompt, response)

                    synapse.full_text = markdown_content
                    synapse.summary = summary
                    synapse.new_links = new_links


            elif type(synapse) == TargonSearchResult:
                prompt = self._build_search_result_prompt(query, sources, context)
                response = self.post_http_request(prompt, self.config.sybil.api_url, n=1, stream=False, synapse=synapse)
                output = self.get_response(prompt, response)

                synapse.completion = output
                
            return synapse


        async def _streaming_prompt(prompt: str, send: Send):
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

            try:
                prompt = self._build_search_result_prompt(query, sources, context)

                response = self.post_http_request(prompt, self.config.sybil.api_url, n=1, stream=True, synapse=synapse)
                bt.logging.info('response', response)

                buffer = []
                output_text = ""
                for token in self.get_streaming_response(response):
                    # print(token)
                    token = token.replace(prompt, "")
                    token = token.replace(output_text, "") if output_text else token
                    output_text += token
                    bt.logging.info(f"token", token)

                    N = 1  # Number of tokens to send back to the client at a time
                    buffer.append(token)
                    # If buffer has N tokens, send them back to the client.
                    if len(buffer) == N:
                        joined_buffer = "".join(buffer)
                        if '\n' in token:
                            token = "<newline>"
                        await send(
                            {
                                "type": "http.response.body",
                                "body": token,
                                "more_body": True,
                            }
                        )
                        bt.logging.debug(f"Streamed tokens: {token}")
                        buffer = []  # Clear the buffer for next batch of tokens
                            # await asyncio.sleep(0.08)  # Simulate streaming delay
                
                # # Send any remaining tokens in the buffer
                if buffer:
                    joined_buffer = "".join(buffer)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": f"{joined_buffer}",
                            "more_body": False,  # No more tokens to send
                        }
                    )
                    bt.logging.trace(f"Streamed tokens: {joined_buffer}")
            except Exception as e:
                bt.logging.error(f"Exception: {e}")
                await send(
                    {
                        "type": "http.response.body",
                        "body": f"Exception: {e}".encode("utf-8"),
                        "more_body": False,  # No more tokens to send
                    }
                )


        # message = synapse.messages[0]
        
        if type(synapse) != TargonLinkPrediction or type(synapse) != TargonSearchResult:
            if synapse.stream:
                token_streamer = partial(_streaming_prompt, prompt)
                return synapse.create_streaming_response(token_streamer)
            
        synapse = _prompt(prompt)
        return synapse


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
