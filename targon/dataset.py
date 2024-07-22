import random

from os import urandom
from datetime import datetime

from openai.types.chat import ChatCompletion
from targon import protocol


async def generate_dataset(self):
    """
    Asynchronously generates a dataset for the verifier by sampling from the existing dataset,
    generating query prompts, and sourcing additional text generation based on these prompts.

    Returns:
        tuple: A tuple containing the final messages and the sampling parameters.
    """
    # Generate a random seed for reproducibility in sampling and text generation
    random.seed(urandom(100))
    seed = random.randint(10000, 10000000)

    # Determine the maximum number of new tokens to generate
    max_new_tokens = random.randint(16, 1024)

    # Create sampling parameters using the generated seed and token limit
    sampling_params = protocol.InferenceSamplingParams(
        seed=seed, max_new_tokens=max_new_tokens
    )

    # Sample a random row from the dataset and extract the text
    random_row_text = self.dataset.sample(n=1)["text"].iloc[0]

    # Generate a query from the sampled text and perform text generation
    messages = create_query_prompt(random_row_text)

    res: ChatCompletion
    res = self.client.chat.completions.create(
        model=self.config.neuron.model_name,
        messages=messages,
        stream=False,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        seed=sampling_params.seed,
        timeout=5
    )

    # Create a final search prompt using the query and sources
    completion = res.choices[0].message.content
    if completion is None:
        print(res)
        raise Exception("No completion")
    prompt = create_search_prompt(completion)

    return prompt, sampling_params


def create_search_prompt(query: str):
    """
    Creates a formatted search prompt for the verifier based on the provided query and sources.

    Args:
        query (str): The generated query text.
        sources (str): The generated sources text.

    Returns:
        str: A formatted prompt string ready for further processing.
    """
    # Format the current date for inclusion in the prompt
    date = datetime.now().strftime("%Y-%m-%d")

    # Construct the system message with dynamic content
    system_message = f"""
### Current Date: {date}
### Instruction: 
You are Sybil.com, an expert language model tasked with performing a search over the given query and search results.
You are running the text generation on Subnet 4, a bittensor subnet developed by Manifold Labs.
Your answer should be short, two paragraphs exactly, and should be relevant to the query.
"""

    # Compile the chat components into a structured format
    chats = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    # Apply the chat template without tokenization
    return chats


def create_query_prompt(query: str):
    """
    Creates a query prompt for the verifier based on the provided text.

    Args:
        query (str): The text to base the query prompt on.

    Returns:
        str: A formatted query prompt string.
    """
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
    chats = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    # Apply the chat template without tokenization
    return chats


def create_sources_prompt(query: str):
    """
    Creates a source prompt for the verifier based on the provided query.

    Args:
        query (str): The query text to base the source prompt on.

    Returns:
        str: A formatted source prompt string.
    """
    # Format the current date for inclusion in the prompt
    date = datetime.now().strftime("%Y-%m-%d")

    # Construct the system message with dynamic content
    system_message = f"""
### Current Date: {date}
### Instruction:
You are to take the query information that is passed from you and create a short search source for the query data. 
Do not answer the information, just create a search source.
Assistant should always start the response with "Search source: "
"""

    # Compile the chat components into a structured format
    chats = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    # Apply the chat template without tokenization
    return chats

