# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Manifold Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import math
import torch
import random
import typing
import string
import asyncio
import plotext as plt
import bittensor as bt

from targon import protocol
from targon.utils.prompt import create_prompt
from targon.constants import CHALLENGE_FAILURE_REWARD
from targon.verifier.uids import get_random_uids
from targon.verifier.reward import hashing_function
from targon.verifier.state import EventSchema


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
    indices = torch.topk(metagraph.incentive, n).indices

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
            for uid, ip, coldkey in zip(uids_with_highest_incentives, ips, coldkeys)
        ]
    )
    # axons_with_highest_incentives = [metagraph.axons[uid] for uid in uids_with_highest_incentives]
    # unique_ip_to_uid = {ip: uid for ip, uid in zip(ips, uids_with_highest_incentives)}
    # uids = list(unique_ip_to_uid.values())
    return uids_with_highest_incentives


def _filter_verified_responses(uids, responses):
    """
    Filters out responses that have not been verified.

    Args:
    - uids (list): A list of user IDs.
    - responses (list): A list of tuples containing verification status and response.

    Returns:
    - tuple: Two tuples, one containing filtered user IDs and the other containing their corresponding responses.
    """
    not_none_responses = [
        (uid, response[0])
        for (uid, (verified, response)) in zip(uids, responses)
        if verified != None
    ]

    if len(not_none_responses) == 0:
        return (), ()

    uids, responses = zip(*not_none_responses)
    return uids, responses


def check_tokens(self, prover_output, ground_truth_output):
    # Tokenize the prover output and the ground truth output
    prover_tokenized = self.embedding_tokenizer(
        prover_output, return_tensors="pt", padding=True, truncation=True
    )
    ground_truth_tokenized = self.embedding_tokenizer(
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

    bt.logging.trace(score)
    return score


def verify(self, prover_output, ground_truth_output, prover_ss58):
    """
    Verifies the prover's output against the ground truth output.

    Args:
    - self: Reference to the current instance of the class.
    - prover_output (str): The output provided by the prover.
    - ground_truth_output (str): The expected output.
    - prover_ss58 (str): The prover's SS58 address.

    Returns:
    - bool: True if the outputs match or if the embedding check passes, False otherwise.
    """
    prover_output_hash = hashing_function(prover_output)
    ground_truth_hash = hashing_function(ground_truth_output)

    return check_tokens(self, prover_output, ground_truth_output)


async def api_chat_completions(
    self,
    prompt: str,
    sampling_params: protocol.InferenceSamplingParams,
) -> typing.Tuple[bool, protocol.Inference]:
    """
    Handles a inference sent to a prover and verifies the response.

    Returns:
    - Tuple[bool, protocol.Inference]: A tuple containing the verification result and the inference.
    """
    try:
        synapse = protocol.Inference(
            sources=[],
            query=prompt,
            sampling_params=sampling_params,
        )

        start_time = time.time()
        token_count = 0
        uid = select_highest_n_peers(1, self.metagraph)[0]
        async for token in await self.dendrite(
            self.metagraph.axons[uid],
            synapse,
            deserialize=False,
            timeout=self.config.neuron.timeout,
            streaming=True,
        ):
            if isinstance(token, list):
                yield token[0]
            elif isinstance(token, str):
                yield token
            token_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_per_second = token_count / elapsed_time
        bt.logging.info(f"Token generation rate: {tokens_per_second} tokens/second")
    except Exception as e:
        bt.logging.error(e)


async def handle_inference(
    self,
    uid: int,
    private_input: typing.Dict,
    ground_truth_output: str,
    sampling_params: protocol.InferenceSamplingParams,
) -> typing.Tuple[bool, protocol.Inference]:
    """
    Handles a inference sent to a prover and verifies the response.

    Parameters:
    - uid (int): The UID of the prover being inferenced.

    Returns:
    - Tuple[bool, protocol.Inference]: A tuple containing the verification result and the inference.
    """
    if not self.config.mock:
        synapse = protocol.Inference(
            sources=[private_input["sources"]],
            query=private_input["query"],
            sampling_params=sampling_params,
        )

        response_tokens = []

        try:
            start_time = time.time()
            token_count = 0
            async for token in await self.dendrite(
                self.metagraph.axons[uid],
                synapse,
                deserialize=False,
                timeout=self.config.neuron.timeout,
                streaming=True,
            ):
                if isinstance(token, list):
                    response_tokens.append(token[0])
                    token_count += 1
                elif isinstance(token, str):
                    response_tokens.append(token)
                    token_count += 1
                else:
                    output_synapse = token

            end_time = time.time()
            output = "".join(response_tokens)

            elapsed_time = end_time - start_time
            tokens_per_second = token_count / (
                elapsed_time if elapsed_time > 0 else 1000
            )
            bt.logging.info(f"Token generation rate: {tokens_per_second} tokens/second")

        except Exception as e:
            bt.logging.error(f"Error in handle_inference: {e}")
            return False, (synapse, uid, 0)

        # output_encoded = output.encode('utf-8')
        if output is not None:
            start_time = time.time()
            output_normalized = output.replace("\r\n", "\n")
            output_cleaned = " ".join(output_normalized.split())
            end_time = time.time()
            elapsed_time = end_time - start_time
            bt.logging.info(f"Output normalization rate: {elapsed_time} seconds")
            bt.logging.info(
                f"Output normalization rate: {tokens_per_second} tokens/second"
            )

            bt.logging.debug("output", output_cleaned)
            score = verify(
                self, output_cleaned, ground_truth_output, self.metagraph.hotkeys[uid]
            )

        else:
            score = 0

        output_dict = (output_synapse, uid, tokens_per_second)
        return score, output_dict

    else:
        prompt = create_prompt(private_input)

        synapse = protocol.Inference(
            sources=[private_input["sources"]],
            query=private_input["query"],
            sampling_params=sampling_params,
        )

        response_tokens = []

        start_time = time.time()
        token_count = 0
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
            response_tokens.append(token)
            token_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_per_second = token_count / elapsed_time

        response = "".join(response_tokens)

        synapse.completion = response

        verified = verify(
            self, response, ground_truth_output, self.metagraph.hotkeys[uid]
        )

        output_dict = (synapse, uid, tokens_per_second)
        return verified, output_dict


async def inference_data(self):
    """
    Orchestrates the inference process, from fetching inference data to applying rewards based on the verification results.

    This function performs several key steps:
    1. Fetches inference data from a configured URL.
    2. Generates a ground truth output using the inference data.
    3. Selects a set of UIDs (user identifiers) to inference.
    4. Sends the inference to each selected UID and collects their responses.
    5. Verifies the responses against the ground truth output.
    6. Applies rewards or penalties based on the verification results.
    7. Updates the event schema with the results of the inference.

    The function handles both real and mock inferences, allowing for testing without actual data.

    Returns:
    - EventSchema: An object containing detailed information about the inference, including which UIDs were successful, the rewards applied, and other metadata.
    """

    def remove_indices_from_tensor(tensor, indices_to_remove):
        # Sort indices in descending order to avoid index out of range error
        sorted_indices = sorted(indices_to_remove, reverse=True)
        for index in sorted_indices:
            tensor = torch.cat([tensor[:index], tensor[index + 1 :]])
        return tensor

    # --- Create the event
    event = EventSchema(
        task_name="inference",
        successful=[],
        completion_times=[],
        task_status_messages=[],
        task_status_codes=[],
        block=self.subtensor.get_current_block(),
        uids=[],
        step_length=0.0,
        best_uid=-1,
        best_hotkey="",
        rewards=[],
        set_weights=None,
        moving_averaged_scores=None,
    )

    bt.logging.info("Generating challenge data")
    challenge_data = {
        "query": "".join(random.choice(string.ascii_letters) for _ in range(12)),
        "sources": "".join(random.choice(string.ascii_letters) for _ in range(12)),
    }
    prompt = create_prompt(challenge_data)

    bt.logging.info("prompt created")
    seed = random.randint(10000, 10000000)

    sampling_params = protocol.InferenceSamplingParams(seed=seed)

    ground_truth_tokens = []

    start_time = time.time()
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

    # ground_truth_output_encoded = ground_truth_output.encode('utf-8')
    ground_truth_output_normalized = ground_truth_output.replace("\r\n", "\n")
    ground_truth_output_cleaned = " ".join(ground_truth_output_normalized.split())

    # --- Get the uids to query
    tasks = []
    # uids = await get_tiered_uids( self, k=self.config.neuron.sample_size )
    uids = get_random_uids(self, k=self.config.neuron.sample_size)

    bt.logging.debug(f"inference uids {uids}")

    responses = []
    for uid in uids:
        tasks.append(
            asyncio.create_task(
                handle_inference(
                    self,
                    uid,
                    challenge_data,
                    ground_truth_output_cleaned,
                    sampling_params,
                )
            )
        )
    responses = await asyncio.gather(*tasks)

    # Create a list of tuples (uid, tokens_per_second) for sorting
    uid_tokens_pairs = [
        (uid, tokens_per_second if verified >= 0.6 else 1e-7)
        for verified, (_, uid, tokens_per_second) in responses
    ]
    # Initialize or update moving averages dictionary
    if not hasattr(self, "moving_averages"):
        self.moving_averages = {uid: 0 for uid, _ in uid_tokens_pairs}

    for uid, tokens_per_second in uid_tokens_pairs:
        if uid in self.moving_averages:
            self.moving_averages[uid] = (
                self.config.neuron.moving_average_alpha * tokens_per_second
                + (1 - self.config.neuron.moving_average_alpha)
                * self.moving_averages[uid]
            )
        else:
            self.moving_averages[uid] = tokens_per_second

    # Sort the list by tokens_per_second in descending order
    sorted_uid_tokens_pairs = sorted(uid_tokens_pairs, key=lambda x: x[1])
    uids_sorted = [uid for uid, _ in sorted_uid_tokens_pairs]

    # Extract tokens_per_second for plotting
    tokens_per_second_sorted = [
        tokens_per_second for _, tokens_per_second in sorted_uid_tokens_pairs
    ]

    # rewards: torch.FloatTensor = torch.zeros(len(self.metagraph.uids), dtype=torch.float32).to(
    #     self.device
    # )

    # Calculate rewards based on the difference between the highest and lowest tokens_per_second using moving averages
    self.max_tokens_per_second = (
        max(tokens_per_second_sorted)
        if self.max_tokens_per_second < max(tokens_per_second_sorted)
        else self.max_tokens_per_second
    )
    self.min_tokens_per_second = (
        min(tokens_per_second_sorted)
        if self.min_tokens_per_second > min(tokens_per_second_sorted)
        else self.min_tokens_per_second
    )
    self.range_tokens_per_second = (
        self.max_tokens_per_second - self.min_tokens_per_second
    )
    # Calculate the current average tokens per second
    current_average_tokens_per_second = sum(tokens_per_second_sorted) / len(
        tokens_per_second_sorted
    )
    # Update the moving average for tokens per second
    self.average_tokens_per_second = (
        self.config.neuron.moving_average_alpha * current_average_tokens_per_second
        + (1 - self.config.neuron.moving_average_alpha) * self.average_tokens_per_second
    )
    # Initialize moving average for rewards

    for i, (uid, tokens_per_second) in enumerate(sorted_uid_tokens_pairs):
        if self.range_tokens_per_second > 0:
            normalized_difference = (
                tokens_per_second - self.average_tokens_per_second
            ) / self.range_tokens_per_second
            reward_multiplier = math.exp(
                normalized_difference * 10
            )  # Scale the difference to enhance reward disparity
        else:
            reward_multiplier = (
                1  # Avoid division by zero if all tokens_per_second are the same
            )
        self.rewards[uid] = reward_multiplier * tokens_per_second
        # Update moving average for rewards
        self.scores[uid] = (
            self.config.neuron.moving_average_alpha * self.rewards[uid]
            + (1 - self.config.neuron.moving_average_alpha) * self.scores[uid]
        )

    # Print the highest UID and its corresponding tokens_per_second and reward score
    # Find the highest UID and its corresponding tokens_per_second and reward score
    highest_uid = max(range(len(self.rewards)), key=lambda uid: self.rewards[uid])
    # Safely retrieve the tokens_per_second for the highest_uid
    highest_tokens_per_second = next(
        (tps for uid, tps in uid_tokens_pairs if uid == highest_uid), None
    )

    if highest_tokens_per_second is None:
        bt.logging.error(f"No tokens per second found for highest UID: {highest_uid}")
        # Handle the case where no matching UID is found
    else:
        highest_reward = self.rewards.tolist()[highest_uid]

        print(
            f"Highest UID: {highest_uid}, Tokens/Second: {highest_tokens_per_second}, Reward: {highest_reward}"
        )
        print(
            f"Highest UID: {highest_uid}, Tokens/Second: {highest_tokens_per_second}, Reward: {highest_reward}"
        )
        print(f"Average Tokens/Second: {self.average_tokens_per_second}")
    print(self.scores)
    # Plot moving average of rewards
    y = plt.scatter(
        uids_sorted, self.scores.to("cpu").numpy(), color="red"
    )  # Reduced marker size for a smaller plot
    plt.title("Sorted Tokens per Second")
    plt.xlabel("UID (sorted)")
    plt.ylabel("Reward Score")
    plt.plotsize(100, 20)
    plt.show()
    plt.clf()  # Clear the plot after showing
    mock_weights_measurments(self)


def mock_weights_measurments(self):
    # if torch.isnan(self.scores).any():
    #     bt.logging.warning(
    #         "Scores contain NaN values. This may be due to a lack of responses from provers, or a bug in your reward functions."
    #     )

    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(
        self.scores - self.scores.mean(dim=0), p=1, dim=0
    )

    # bt.logging.debug("raw_weights", raw_weights)
    # bt.logging.debug("raw_weight_uids", self.metagraph.uids)
    # # Process the raw weights to final_weights via subtensor limitations.
    (
        processed_weight_uids,
        processed_weights,
    ) = bt.utils.weight_utils.process_weights_for_netuid(
        uids=self.metagraph.uids,
        weights=raw_weights.to("cpu"),
        netuid=self.config.netuid,
        subtensor=self.subtensor,
        metagraph=self.metagraph,
    )
    # bt.logging.debug("processed_weights", processed_weights)
    # bt.logging.debug("processed_weight_uids", processed_weight_uids)

    # # Convert to uint16 weights and uids.
    # (
    #     uint_uids,
    #     uint_weights,
    # ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
    #     uids=processed_weight_uids, weights=processed_weights
    # )
    # bt.logging.debug("uint_weights", uint_weights)
    # bt.logging.debug("uint_uids", uint_uids)

    # Plotting the graph of processed_weights against processed_weight_uids
    plt.plotsize(100, 20)
    plt.scatter(processed_weight_uids, processed_weights, color="red")
    plt.title("Processed Weights vs UIDs")
    plt.xlabel("UIDs")
    plt.ylabel("Processed Weights")
    plt.grid(True)
    plt.show()
