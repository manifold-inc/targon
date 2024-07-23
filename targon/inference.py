import time
import bittensor as bt
from typing import List
from targon import protocol
from pydantic import BaseModel
import json
import traceback


class InferenceStats(BaseModel):
    time_to_first_token: float
    time_for_all_tokens: float
    tokens_per_second: float
    tokens: List[str]
    response: str
    verified: bool
    uid: int


async def create_ground_truth(self, messages, sampling_params):
    ground_truth_tokens = []

    stream = self.client.chat.completions.create(
        model=self.config.neuron.model_name,
        messages=messages,
        stream=True,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        seed=sampling_params.seed,
        timeout=5,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if not token:
            continue
        ground_truth_tokens.append(token)

    ground_truth_output = "".join(ground_truth_tokens)

    return ground_truth_output


async def handle_inference(self, messages, sampling_params, uid, ground_truth):
    try:
        synapse = protocol.Inference(
            messages=json.dumps(messages),
            sampling_params=sampling_params,
        )

        response_tokens = []

        token_count = 0
        start_send_message_time = time.time()
        end_send_message_time = None
        start_token_time = 0

        async for token in await self.dendrite(
            self.metagraph.axons[uid],
            synapse,
            deserialize=False,
            timeout=self.config.neuron.timeout,
            streaming=True,
        ):
            if token_count == 1:
                end_send_message_time = time.time()
                start_token_time = time.time()
            if isinstance(token, protocol.Inference):
                continue
            for t in token:
                response_tokens.append(t)
            token_count += 1

        if token_count <= 1 or len(response_tokens) <= 1:
            return None

        if end_send_message_time is None:
            end_send_message_time = time.time()
            start_token_time = end_send_message_time

        end_token_time = time.time()

        time_to_first_token = end_send_message_time - start_send_message_time
        time_for_all_tokens = end_token_time - start_token_time

        tokens_per_second_partial = (
            token_count / time_for_all_tokens
            if token_count > 0 and time_for_all_tokens > 0
            else 0
        )
        tokens_per_second = tokens_per_second_partial

        bt.logging.info(f"Time to receive all tokens: {time_for_all_tokens}")
        bt.logging.info(f"Time to receive first token: {time_to_first_token}")
        bt.logging.info(f"Tokens per second: {tokens_per_second}")

        response = "".join(response_tokens)

        verified = check_tokens(self, response, ground_truth)

        # check if the response was pregenerated, meaning the time it takes to get the first token is much longer than the total generation
        if time_to_first_token > 1.8 * time_for_all_tokens:
            verified = False
            tokens_per_second = 0

        stats = InferenceStats(
            time_to_first_token=time_to_first_token,
            time_for_all_tokens=time_for_all_tokens,
            tokens_per_second=tokens_per_second,
            tokens=response_tokens,
            response=response,
            verified=verified,
            uid=uid,
        )

        return stats
    except Exception as e:
        bt.logging.error(f"Error in forward: {e}")
        bt.logging.error(traceback.format_exc())
        time.sleep(12)


def check_tokens(self, miner_output, ground_truth_output):
    # Tokenize the miner output and the ground truth output
    miner_tokenized = self.prompt_tokenizer(
        miner_output, return_tensors="pt", padding=True, truncation=True
    )
    ground_truth_tokenized = self.prompt_tokenizer(
        ground_truth_output, return_tensors="pt", padding=True, truncation=True
    )

    # Compare the list of tokens
    miner_tokens = miner_tokenized["input_ids"]
    ground_truth_tokens = ground_truth_tokenized["input_ids"]

    bt.logging.trace(miner_tokens)
    bt.logging.trace(ground_truth_tokens)

    # convert to list
    miner_tokens = miner_tokens[0].tolist()
    ground_truth_tokens = ground_truth_tokens[0].tolist()

    # make the tokenized outputs the same length, perferring the ground truth output length
    if len(miner_tokens) > len(ground_truth_tokens):
        miner_tokens = miner_tokens[: len(ground_truth_tokens)]
    elif len(miner_tokens) < len(ground_truth_tokens):
        return False

    # Calculate the score from 0 to 1
    score = sum([1 for token in miner_tokens if token in ground_truth_tokens]) / len(
        miner_tokens
    )

    if score < 0.60:
        return False

    return True
