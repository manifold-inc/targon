import torch
import random
import bittensor as bt
from typing import List

from targon.validator import check_uid_availability
from .prompts import qa_prompt
from targon.protocol import TargonQA, TargonLinkPrediction, TargonSearchResult, TargonSearchResultStream

def get_random_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
                
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample([uid for uid in avail_uids if uid not in candidate_uids], k-len(candidate_uids))

    uids = torch.tensor(random.sample(available_uids, k), dtype=torch.int64)
    return uids


async def fetch(self, synapse, uids):
    responses = await self.dendrite_pool.async_forward(
        uids = uids,
        synapse = synapse,
        timeout = 12
    )
    return responses


async def _qa_forward(self, question: str, uids: List[int]):
    """Queries a list of uids for a question.
    Args:
        question (str): Question to query.
        uids (torch.LongTensor): Uids to query.
        timeout (float): Timeout for the query.
    Returns:
        responses (List[TargonQA]): List of responses.
    """
    # Check if we have any uids to query.
    qa_synapse = TargonQA(question=question)
    responses = await fetch(self, qa_synapse, uids)

    bt.logging.info('qa synapse', responses)

    answers = [response.answer for response in responses]
    return answers



async def _link_prediction_forward(self, question: str, uids: List[int]):
    """Queries a list of uids for a question.
    Args:
        question (str): Question to query.
        uids (torch.LongTensor): Uids to query.
        timeout (float): Timeout for the query.
    Returns:
        responses (List[TargonQA]): List of responses.
    """
    # Check if we have any uids to query.
    search_synapse = TargonLinkPrediction( query=question )
    responses = await fetch( self, search_synapse, uids )

    sources = [response.results for response in responses]

    return sources


async def _search_result_forward(self, question: str, sources: List[dict], uids: List[int]):
    """Queries a list of uids for a question.
    Args:
        question (str): Question to query.
        uids (torch.LongTensor): Uids to query.
        timeout (float): Timeout for the query.
    Returns:
        responses (List[TargonQA]): List of responses.
    """
    # Check if we have any uids to query.
    search_synapse = TargonSearchResult( query=question, sources=sources )
    responses = await fetch( self, search_synapse, uids )

    completions = [response.completion for response in responses]

    return completions


async def forward_fn(self, validation=True, stream=False):
    """Queries a list of uids for a question.
    Args:
        question (str): Question to query.
        uids (torch.LongTensor): Uids to query.
        timeout (float): Timeout for the query.
    Returns:
        responses (List[TargonQA]): List of responses.
    """
    k = self.config.neuron.followup_sample_size
    if validation:
            uids = get_random_uids(self, k=k).to(self.device)
            data = next(self.dataset)["text"]

            random_cutoff = random.randint(15, 30)
            # Truncate context to a limited set of sentences.
            base_text = ".".join(data.split(".", maxsplit=random_cutoff)[:-1])
            prompt = qa_prompt(base_text)

            questions = await _qa_forward(self, prompt, uids)

            # TODO: select most relevant question from questions
            top_question = questions[0]
            bt.logging.info('top_question', top_question)
            # sources = await _link_prediction_forward(self, top_question, uids)

            sources = []
            completions = await _search_result_forward(self, top_question, sources, uids)
            bt.logging.info("completions", completions)

            name = "search-result"
            # Compute the rewards for the responses given the prompt.
            rewards: torch.FloatTensor = torch.zeros(len(completions), dtype=torch.float32).to(self.device)
            for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
                reward_i, reward_i_normalized = reward_fn_i.apply(prompt, completions, name)
                rewards += weight_i * reward_i_normalized.to(self.device)
                bt.logging.trace(str(reward_fn_i.name), reward_i.tolist())
                bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())

            for masking_fn_i in self.masking_functions:
                mask_i, mask_i_normalized = masking_fn_i.apply(prompt, completions, name)
                rewards *= mask_i_normalized.to(self.device)  # includes diversity
                bt.logging.trace(str(masking_fn_i.name), mask_i_normalized.tolist())

            
            scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(0, uids, rewards).to(self.device)

            # Update moving_averaged_scores with rewards produced by this step.
            # shape: [ metagraph.n ]
            alpha: float = self.config.neuron.moving_average_alpha
            self.moving_averaged_scores: torch.FloatTensor = alpha * scattered_rewards + (1 - alpha) * self.moving_averaged_scores.to(
                self.device
            )

            bt.logging.info("rewards", rewards.tolist())

            





            






