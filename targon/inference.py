import time

import bittensor as bt

from typing import List
from targon import protocol
from pydantic import BaseModel

class InferenceStats(BaseModel):
    time_to_first_token: float
    time_for_all_tokens: float
    tokens_per_second: float
    tokens: List[str]
    response: str
    verified: bool
    uid: int



async def create_ground_truth(self, prompt, sampling_params):
    ground_truth_tokens = []

    async for token in await self.client.text_generation(
        prompt,
        best_of=sampling_params.best_of,
        max_new_tokens=sampling_params.max_new_tokens,
        seed=sampling_params.seed,
        do_sample=sampling_params.do_sample,
        repetition_penalty=sampling_params.repetition_penalty,
        temperature=sampling_params.temperature,
        top_k=sampling_params.top_k,
        top_p=sampling_params.top_p,
        truncate=sampling_params.truncate,
        typical_p=sampling_params.typical_p,
        watermark=sampling_params.watermark,
        details=False,
        stream=True,
    ):
        ground_truth_tokens.append(token)

    ground_truth_output = "".join(ground_truth_tokens)

    return ground_truth_output


async def handle_inference(self, prompt, sampling_params, uid, ground_truth):

    synapse = protocol.Inference(
        query=prompt,
        sources=[''],
        sampling_params=sampling_params,
    )

    response_tokens = []

    token_count = 0
    start_send_message_time = time.time()
    end_send_message_time = None

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
        if isinstance(token, list):
            response_tokens.append(token[0])
            token_count += 1
        elif isinstance(token, str):
            response_tokens.append(token)
            token_count += 1
        else:
            output_synapse = token
    
    if end_send_message_time is None:
        end_send_message_time = time.time()
        start_token_time = end_send_message_time

    end_token_time = time.time()

    time_to_first_token = end_send_message_time - start_send_message_time
    time_for_all_tokens = end_token_time - start_token_time

    tokens_per_second = token_count / time_for_all_tokens
    bt.logging.info(f"Time to receive all tokens: {time_for_all_tokens}")
    bt.logging.info(f"Time to receive first token: {time_to_first_token}")
    bt.logging.info(f"Tokens per second: {tokens_per_second}")

    response = "".join(response_tokens)
    
    verified = check_tokens(self, response, ground_truth)

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



def check_tokens(self, prover_output, ground_truth_output):
    # Tokenize the prover output and the ground truth output
    prover_tokenized = self.prompt_tokenizer(
        prover_output, return_tensors="pt", padding=True, truncation=True
    )
    ground_truth_tokenized = self.prompt_tokenizer(
        ground_truth_output, return_tensors="pt", padding=True, truncation=True
    )

    # Compare the list of tokens
    prover_tokens = prover_tokenized["input_ids"]
    ground_truth_tokens = ground_truth_tokenized["input_ids"]

    bt.logging.trace(prover_tokens)
    bt.logging.trace(ground_truth_tokens)

    # convert to list
    prover_tokens = prover_tokens[0].tolist()
    ground_truth_tokens = ground_truth_tokens[0].tolist()

    # make the tokenized outputs the same length, perferring the ground truth output length
    if len(prover_tokens) > len(ground_truth_tokens):
        prover_tokens = prover_tokens[: len(ground_truth_tokens)]
    elif len(prover_tokens) < len(ground_truth_tokens):
        return 0

    # Calculate the score from 0 to 1
    score = sum([1 for token in prover_tokens if token in ground_truth_tokens]) / len(
        prover_tokens
    )

    if score < 0.75:
        return False

    return True
