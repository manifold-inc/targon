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
import torch
import random
import typing
import asyncio
import httpx
import bittensor as bt

from targon import protocol
from requests.auth import HTTPBasicAuth
from targon.verifier.event import EventSchema
from targon.utils.prompt import create_prompt
from targon.utils.misc import return_json_params
from torch.nn.functional import cosine_similarity
from targon.constants import CHALLENGE_FAILURE_REWARD
from targon.verifier.uids import get_tiered_uids, get_random_uids
from targon.verifier.reward import hashing_function, apply_reward_scores
from targon.verifier.bonding import update_statistics, get_tier_factor, get_similarity_threshold


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

def check_tokens( self, prover_output, ground_truth_output ):
    
    # Tokenize the prover output and the ground truth output
    prover_tokenized = self.embedding_tokenizer(prover_output, return_tensors="pt", padding=True, truncation=True)
    ground_truth_tokenized = self.embedding_tokenizer(ground_truth_output, return_tensors="pt", padding=True, truncation=True)

    # Compare the list of tokens
    prover_tokens = prover_tokenized['input_ids']
    ground_truth_tokens = ground_truth_tokenized['input_ids']

    bt.logging.info(prover_tokens)
    bt.logging.info(ground_truth_tokens)

    # convert to list
    prover_tokens = prover_tokens[0].tolist()
    ground_truth_tokens = ground_truth_tokens[0].tolist()

    # make the tokenized outputs the same length, perferring the ground truth output length
    if len(prover_tokens) > len(ground_truth_tokens):
        prover_tokens = prover_tokens[:len(ground_truth_tokens)]
    elif len(prover_tokens) < len(ground_truth_tokens):
        return 0

    # Calculate the score from 0 to 1
    score = sum([1 for token in prover_tokens if token in ground_truth_tokens]) / len(prover_tokens)

    bt.logging.info(score)
    return score

def verify( self, prover_output, ground_truth_output, prover_ss58 ):
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

    if not prover_output_hash == ground_truth_hash:
        bt.logging.debug(
            f"Output hash {prover_output_hash} does not match ground truth hash {ground_truth_hash}"
        )

        # check how t


        # return asyncio.run(embedding_check( self, prover_output, ground_truth_output, prover_ss58 ))
        return check_tokens( self, prover_output, ground_truth_output )
    
    bt.logging.debug(
        f"Output hash {prover_output_hash} matches ground truth hash {ground_truth_hash}"
    )
    return True

async def handle_inference( self, uid: int, private_input: typing.Dict, ground_truth_output: str, sampling_params: protocol.InferenceSamplingParams ) -> typing.Tuple[bool, protocol.Inference]:
    """
    Handles a inference sent to a prover and verifies the response.

    Parameters:
    - uid (int): The UID of the prover being inferenced.

    Returns:
    - Tuple[bool, protocol.Inference]: A tuple containing the verification result and the inference.
    """

    hotkey = self.metagraph.hotkeys[uid]
    keys = await self.database.hkeys(f"hotkey:{hotkey}")
    bt.logging.trace(f"{len(keys)} stats pulled for hotkey {hotkey}")

    if not self.config.mock:
        synapse = protocol.Inference(
            sources = [private_input["sources"]],
            query = private_input["query"],
            sampling_params=sampling_params,
        )


        response = await self.dendrite(
            self.metagraph.axons[uid],
            synapse,
            deserialize=False,
            timeout=self.config.neuron.timeout,
        )

        output = response.completion

        # output_encoded = output.encode('utf-8')
        if output is not None:
            output_normalized = output.replace('\r\n', '\n')
            output_cleaned = ' '.join(output_normalized.split())

        
            bt.logging.debug('output', output_cleaned)
            verified = verify( self, output_cleaned, ground_truth_output, self.metagraph.hotkeys[uid] )
        
        else:
            verified = False

        output_dict = (
            response,
            uid
        )
        return verified, output_dict
    
    else:
        prompt = create_prompt(private_input)

        synapse = protocol.Inference(
            sources = [private_input["sources"]],
            query = private_input["query"],
            sampling_params=sampling_params,
        )

        response_tokens = []

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
            stream=True
        ):
            response_tokens.append(token)
        
        response = ''.join(response_tokens)

        synapse.completion = response
        

        verified = verify( self, response, ground_truth_output, self.metagraph.hotkeys[uid] )

        output_dict = (
            synapse,
            uid
        )
        return verified, output_dict

async def inference_data( self ):
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

    
    bt.logging.info("Grabbing inference data")
    url = self.config.neuron.challenge_url # inference data url

    hotkey = self.wallet.hotkey.ss58_address # get the hotkey address
    signature = f"0x{self.wallet.hotkey.sign(hotkey).hex()}"

    private_input = httpx.get(url, auth=HTTPBasicAuth(hotkey, signature)).json()
    bt.logging.info(f"Inference data: {private_input}")
    prompt = create_prompt(private_input)

    bt.logging.info('prompt created')
    seed = random.randint(10000, 10000000)


    sampling_params = protocol.InferenceSamplingParams(
        seed=seed
    )


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
		stream=True
	):
        ground_truth_tokens.append(token)
    
    ground_truth_output = ''.join(ground_truth_tokens)

    # ground_truth_output_encoded = ground_truth_output.encode('utf-8')
    ground_truth_output_normalized = ground_truth_output.replace('\r\n', '\n')
    ground_truth_output_cleaned = ' '.join(ground_truth_output_normalized.split())

    # --- Get the uids to query
    start_time = time.time()
    tasks = []
    # uids = await get_tiered_uids( self, k=self.config.neuron.sample_size )
    uids = get_random_uids( self, k=self.config.neuron.sample_size )

    bt.logging.debug(f"inference uids {uids}")
    responses = []
    for uid in uids:
        tasks.append(asyncio.create_task(handle_inference(self, uid, private_input, ground_truth_output_cleaned, sampling_params)))
    responses = await asyncio.gather(*tasks)


    rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(
        self.device
    )

    remove_reward_idxs = []
    for i, (verified, (response, uid)) in enumerate(responses):
        bt.logging.trace(
            f"Inference iteration {i} uid {uid} response {str(response.completion if not self.config.mock else response)}"
        )

        hotkey = self.metagraph.hotkeys[uid]

        # Update the inference statistics
        await update_statistics(
            ss58_address=hotkey,
            success=verified,
            task_type="inference",
            database=self.database,
            current_block=self.block,
        )

        # Apply reward for this inference
        tier_factor = await get_tier_factor(hotkey, self.database)
        rewards[i] = 1.0 * tier_factor if verified else CHALLENGE_FAILURE_REWARD

        if self.config.mock:
            event.uids.append(uid)
            event.successful.append(verified)        
            event.completion_times.append(0.0)
            event.task_status_messages.append("mock")
            event.task_status_codes.append(0)
            event.rewards.append(rewards[i].item())
        else: 
            event.uids.append(uid)
            event.successful.append(verified)
            event.completion_times.append(response.dendrite.process_time)
            event.task_status_messages.append(response.dendrite.status_message)
            event.task_status_codes.append(response.dendrite.status_code)
            event.rewards.append(rewards[i].item())

    bt.logging.debug(
        f"inference_data() rewards: {rewards} | uids {uids} hotkeys {[self.metagraph.hotkeys[uid] for uid in uids]}"
    )

    event.step_length = time.time() - start_time

    if len(responses) == 0:
        bt.logging.debug(f"Received zero hashes from miners, returning event early.")
        return event

    # Remove UIDs without hashes (don't punish new miners that have no inferences yet)
    uids, responses = _filter_verified_responses(uids, responses)
    bt.logging.debug(
        f"inference_data() full rewards: {rewards} | uids {uids} | uids to remove {remove_reward_idxs}"
    )
    rewards = remove_indices_from_tensor(rewards, remove_reward_idxs)
    bt.logging.debug(f"inference_data() kept rewards: {rewards} | uids {uids}")

    bt.logging.trace("Applying inference rewards")
    apply_reward_scores(
        self,
        uids,
        responses,
        rewards,
        timeout=self.config.neuron.timeout,
        mode=self.config.neuron.reward_mode,
    )

    # Determine the best UID based on rewards
    if event.rewards:
        best_index = max(range(len(event.rewards)), key=event.rewards.__getitem__)
        event.best_uid = event.uids[best_index]
        event.best_hotkey = self.metagraph.hotkeys[event.best_uid]
    
    return event
