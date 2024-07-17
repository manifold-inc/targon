import time

import numpy as np
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


async def create_ground_truth(self, messages, sampling_params):
    ground_truth_tokens = []

    prompt = self.prompt_tokenizer.apply_chat_template(messages, tokenize=False)
    async for token in await self.client.text_generation(
        **sampling_params,
        prompt,
        details=False,
        stream=True,
    ):
        ground_truth_tokens.append(token)

    ground_truth_output = "".join(ground_truth_tokens)

    return ground_truth_output


async def handle_inference(self, messages, sampling_params, uid, ground_truth):
    synapse = protocol.Inference(
        messages=messages,
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
        if isinstance(token, list):
            response_tokens.append(token[0])
            token_count += 1
        elif isinstance(token, str):
            response_tokens.append(token)
            token_count += 1
    
    if end_send_message_time is None:
        end_send_message_time = time.time()
        start_token_time = end_send_message_time

    end_token_time = time.time()

    time_to_first_token = end_send_message_time - start_send_message_time
    time_for_all_tokens = end_token_time - start_token_time

    tokens_per_second_partial = token_count / time_for_all_tokens if token_count > 0 and time_for_all_tokens > 0 else 0
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
        return False

    # Calculate the score from 0 to 1
    score = sum([1 for token in prover_tokens if token in ground_truth_tokens]) / len(
        prover_tokens
    )

    if score < 0.60:
        return False

    return True


async def api_chat_completions(
    self,
    prompt: str,
    sampling_params: protocol.InferenceSamplingParams,
):
    """
    Handles a inference sent to a prover and verifies the response.

    Returns:
    - Tuple[bool, protocol.Inference]: A tuple containing the verification result and the inference.
    """
    try:
        synapse = protocol.Inference(
            query=prompt,
            sampling_params=sampling_params,
        )

        start_time = time.time()
        token_count = 0
        uid = select_highest_n_peers(1, self.metagraph)[0]
        res = ''
        async for token in await self.dendrite(
            self.metagraph.axons[uid],
            synapse,
            deserialize=False,
            timeout=self.config.neuron.timeout,
            streaming=True,
        ):
            if isinstance(token, list):
                res += token[0]
                yield token[0]
            elif isinstance(token, str):
                res += token
                yield token
            token_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_per_second = token_count / elapsed_time
        bt.logging.info(f"Token generation rate: {tokens_per_second} tokens/second")
        bt.logging.info(f"{res} | {token_count}")
    except Exception as e:
        bt.logging.error(str(e))


# get highest incentive axons from metagraph
def select_highest_n_peers(n: int, metagraph=None, return_all=False):
    """
    Selects the highest incentive peers from the metagraph.

    Parameters:
        n (int): number of top peers to return.

    Returns:
        int: uid of the selected peer from unique highest IPs.
    """
    assert metagraph is not None, "metagraph is None"
    # Get the top n indices based on incentive
    indices = np.argsort(metagraph.incentive)[-n:][::-1]
    # Get the corresponding uids
    uids_with_highest_incentives = metagraph.uids[indices].tolist()

    if return_all:
        return uids_with_highest_incentives

    # get the axon of the uids
    axons = [metagraph.axons[uid] for uid in uids_with_highest_incentives]

    # get the ip from the axons
    ips = [axon.ip for axon in axons]

    # get the coldkey from the axons
    coldkeys = [axon.coldkey for axon in axons]

    # Filter out the uids and ips whose coldkeys are in the blacklist
    uids_with_highest_incentives, ips = zip(
        *[
            (uid, ip)
            for uid, ip, _ in zip(uids_with_highest_incentives, ips, coldkeys)
        ]
    )
    # axons_with_highest_incentives = [metagraph.axons[uid] for uid in uids_with_highest_incentives]
    # unique_ip_to_uid = {ip: uid for ip, uid in zip(ips, uids_with_highest_incentives)}
    # uids = list(unique_ip_to_uid.values())
    return uids_with_highest_incentives
