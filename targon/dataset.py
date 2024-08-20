import os
import random
from datetime import datetime
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NAMES = [line.strip() for line in open(os.path.join(BASE_DIR, "names.txt")).readlines()]
COUNTRIES = [line.strip() for line in open(os.path.join(BASE_DIR, "countries.txt")).readlines()]

def create_search_prompt(query: str, increase_entropy=False) ->  Iterable[ChatCompletionMessageParam]:
    # Format the current date for inclusion in the prompt
    date = datetime.now().strftime("%Y-%m-%d")

    # Construct the system message with dynamic content
    system_message = f"""
### Current Date: {date}
### Instruction:
You are Sybil.com, an expert language model tasked with performing a search over the given query and search results.
You are running the text generation on Subnet 4, a bittensor subnet developed by Manifold Labs.
Your answer should be relevant to the query.
"""
    if increase_entropy:
        system_message = f"""
### Current Date: {date}
### Instruction:
You are to take on the role of {random.choice(NAMES)}, an expert language model developed in {random.choice(COUNTRIES)}, tasked with generating responses to user queries.
Your answer should be relevant to the query, and you must start all responses by briefly introducing yourself and stating today's date (which was provided above), then provide the response to the query.
"""

    # Compile the chat components into a structured format
    chats: Iterable[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    # Apply the chat template without tokenization
    return chats


def create_query_prompt(query: str) ->  Iterable[ChatCompletionMessageParam]:
    # Format the current date for inclusion in the prompt
    date = datetime.now().strftime("%Y-%m-%d")

    # Construct the system message with dynamic content
    system_message = f"""
### Current Date: {date}
### Instruction:
You are to take the query information that is passed from you and create a short search query for the query data. 
Do not answer the information, just create a search query. The search query should not be longer than a few words.
Assistant should always start the response with "Search query: "
"""

    # Compile the chat components into a structured format
    chats: Iterable[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    # Apply the chat template without tokenization
    return chats
