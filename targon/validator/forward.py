# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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
# DEALINGS IN
#  THE SOFTWARE.

import time
import torch
import random
import bittensor as bt
import random

from loguru import logger
from typing import List
from dataclasses import asdict
from targon.validator.event import EventSchema
from targon.validator.misc import ttl_get_block
from targon.validator.prompts import followup_prompt, answer_prompt, augment_prompt
from targon.validator.utils import check_uid_availability


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


async def run_step(self, prompt: str, k: int, timeout: float, name: str, exclude: list = [], base_prompt = None):

    if base_prompt == None:
        base_prompt = prompt

    bt.logging.debug("run_step", name)

    # Record event start time.
    event = {"name": name}
    start_time = time.time()

    # Get the list of uids to query for this step.
    uids = get_random_uids(self, k=k, exclude=exclude).to(self.device)

    # Make calls to the network with the prompt.
    responses: List[bt.DendriteCall] = await self.dendrite_pool.async_forward(
        uids=uids,
        roles=["user"],
        messages=[prompt],
        timeout=timeout,
    )

    # Compute the rewards for the responses given the prompt.
    rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(self.device)
    for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
        reward_i, reward_i_normalized = reward_fn_i.apply(prompt, responses, name)
        rewards += weight_i * reward_i_normalized.to(self.device)
        if not self.config.neuron.disable_log_rewards:
            event[reward_fn_i.name] = reward_i.tolist()
            event[reward_fn_i.name + '_normalized'] = reward_i_normalized.tolist()
        bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())

    for masking_fn_i in self.masking_functions:
        mask_i, mask_i_normalized = masking_fn_i.apply(base_prompt, responses, name)
        rewards *= mask_i_normalized.to(self.device)  # includes diversity
        if not self.config.neuron.disable_log_rewards:
            event[masking_fn_i.name] = mask_i.tolist()
            event[masking_fn_i.name + '_normalized'] = mask_i_normalized.tolist()
        bt.logging.trace(str(masking_fn_i.name), mask_i_normalized.tolist())

    # Train the gating model based on the predicted scores and the actual rewards.
    gating_scores: torch.FloatTensor = self.gating_model(prompt).to(self.device)
    gating_loss: torch.FloatTensor = self.gating_model.backward(scores=gating_scores[uids], rewards=rewards)

    # Find the best completion given the rewards vector.
    completions: List[str] = [comp.completion for comp in responses]
    best: str = completions[rewards.argmax(dim=0)].strip()

    # Get completion times
    completion_times: List[float] = [comp.elapsed_time for comp in responses]

    # Compute forward pass rewards, assumes followup_uids and answer_uids are mutually exclusive.
    # shape: [ metagraph.n ]
    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(0, uids, rewards).to(self.device)

    # Update moving_averaged_scores with rewards produced by this step.
    # shape: [ metagraph.n ]
    alpha: float = self.config.neuron.moving_average_alpha
    self.moving_averaged_scores: torch.FloatTensor = alpha * scattered_rewards + (1 - alpha) * self.moving_averaged_scores.to(
        self.device
    )

    # Log the step event.
    event.update(
        {
            "block": ttl_get_block(self),
            "step_length": time.time() - start_time,
            "prompt": prompt,
            "uids": uids.tolist(),
            "completions": completions,
            "completion_times": completion_times,
            "rewards": rewards.tolist(),
            "gating_loss": gating_loss.item(),
            "best": best,
        }
    )
    bt.logging.debug("event:", str(event))
    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event)

    # Log the event to wandb.
    wandb_event = EventSchema.from_dict(event, self.config.neuron.disable_log_rewards)
    if not self.config.wandb.off:
        self.wandb.log(asdict(wandb_event))

    # Return the event.
    return event


async def forward(self):

    # Obtain a unique context from the dataset.
    data = next(self.dataset)["text"]

    random_cutoff = random.randint(15, 30)
    # Truncate context to a limited set of sentences.
    base_text = ".".join(data.split(".", maxsplit=random_cutoff)[:-1])
    aug_prompt = augment_prompt(base_text)

    # Reset Blacklist reward model
    self.blacklist.reset()

    # Request a summary, given the original context.
    augment_event = await run_step(
        self,
        prompt=aug_prompt,
        name="augment",
        k=self.config.neuron.followup_sample_size,
        timeout=self.config.neuron.followup_timeout,
    )

    base_text = augment_event["best"]
    base_prompt = augment_event["best"]
    exclude = augment_event["uids"]
    for k in range(self.config.neuron.num_followup_steps):

        # Get a followup question, given the summarized context.
        prompt = followup_prompt(base_text, i=k)
        followup_event = await run_step(
            self,
            prompt=prompt,
            name="followup" + str(k),
            k=self.config.neuron.followup_sample_size,
            timeout=self.config.neuron.followup_timeout,
            exclude=exclude,
            base_prompt=base_prompt
        )
        exclude += followup_event["uids"]

        # Ask the followup question, given the original context.
        prompt = answer_prompt(base_text, followup_event["best"])
        answer_event = await run_step(
            self,
            prompt=prompt,
            name="answer" + str(k),
            k=self.config.neuron.answer_sample_size,
            timeout=self.config.neuron.answer_timeout,
            exclude=exclude,
            base_prompt=followup_event["best"]
        )
        exclude += answer_event["uids"]

        self.blacklist.question_blacklist.append(followup_event["best"])
        self.blacklist.answer_blacklist.append(answer_event["best"])

        if k == 0:
            # Extend the base text with the best answer.
            base_text = (
                base_text + "\nPrevious Question \nQuestion:" + followup_event["best"] + "\nAnswer:" + answer_event["best"]
            )
        else:
            base_text = base_text + "\nQuestion:" + followup_event["best"] + "\nAnswer:" + answer_event["best"]
    