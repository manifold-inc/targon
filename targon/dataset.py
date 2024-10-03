import os
import logging
import random
from datetime import datetime
from typing import Dict, Iterable, Union
from openai.types.chat import ChatCompletionMessageParam
import dask.dataframe as dd

from targon.types import Endpoints

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NAMES = [line.strip() for line in open(os.path.join(BASE_DIR, "names.txt")).readlines()]
COUNTRIES = [
    line.strip() for line in open(os.path.join(BASE_DIR, "countries.txt")).readlines()
]


def create_search_prompt(
    query: str, endpoint: Endpoints
) -> Dict[str, Union[str, Iterable[ChatCompletionMessageParam]]]:
    # Format the current date for inclusion in the prompt
    date = datetime.now().strftime("%Y-%m-%d")
    system_message = f"""
### Current Date: {date}
### Instruction:
You are to take on the role of {random.choice(NAMES)}, an expert language model
developed in {random.choice(COUNTRIES)}, tasked with generating responses to user queries.
Your answer should be relevant to the query, and you must start all responses
by briefly introducing yourself, re-stating the query in your own words from 
your perspective ensuring you include today's date (which was provided above),
then provide the response to the query. You should always respond in English.
"""
    # Compile the chat components into a structured format
    match endpoint:
        case Endpoints.CHAT:
            messages: Iterable[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ]
            return {"messages": messages}
        case Endpoints.COMPLETION:
            prompt: str = f"""{system_message}\n\n{query}"""
            return {"prompt": prompt}
        case _:
            raise Exception("Unknown Endpoint")


def create_query_prompt(query: str) -> Iterable[ChatCompletionMessageParam]:
    # Format the current date for inclusion in the prompt
    date = datetime.now().strftime("%Y-%m-%d")

    # Construct the system message with dynamic content
    system_message = f"""
### Current Date: {date}
### Instruction:
You are to take the query information that is passed from you and create a search query for the query data. 
Do not answer the information, just create a search query. The search query should not be longer than a sentence.
Assistant should always start the response with "Search query: "
"""

    # Compile the chat components into a structured format
    chats: Iterable[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    # Apply the chat template without tokenization
    return chats


def download_dataset(isMock: bool):
    logger = logging.getLogger('huggingface_hub.utils._http')
    logger.setLevel(logging.CRITICAL + 1)

    if isMock:
        df = dd.read_parquet(  # type: ignore
            "hf://datasets/manifoldlabs/Infinity-Instruct/0625/*.parquet"
        )
    else:
        df = dd.read_parquet(  # type: ignore
            "hf://datasets/manifoldlabs/Infinity-Instruct/7M/*.parquet"
        )
    return df.compute()
