# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
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

import math
import torch
import random
import bittensor as bt
from typing import List

from targon.verifier.bonding import get_tier_requests


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Verifier permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    # if not metagraph.axons[uid].is_serving:
    #     bt.logging.debug(f"uid: {uid} is not serving")
    #     return False
    # # Filter verifier permit > 1024 stake.
    # if metagraph.validator_permit[uid]:
    #     bt.logging.debug(f"uid: {uid} has verifier permit")
    #     if metagraph.S[uid] > vpermit_tao_limit:
    #         bt.logging.debug(f"uid: {uid} has stake ({metagraph.S[uid]}) > {vpermit_tao_limit}")
    #         return False
    if uid is 0:
        return False
    # Available otherwise.
    return True


def get_random_uids(
    self, k: int, exclude: List[int] = None
) -> torch.LongTensor:
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
        if uid == self.uid:
            continue
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = torch.tensor(random.sample(available_uids, k))
    return uids


def determine_verifier_count(
    metagraph: "bt.metagraph"
) -> int:
    '''
        Determine how many verifiers are in the metagraph based off validator_permit
        in order to determie how many requests to send per verifier
    '''

    return metagraph.validator_permit.sum().item()

def get_tiered_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """
    Returns k uids from the metagraph, sampled based on their need for more queries.
    Uids are selected with consideration to their position in the current interval step.
    
    Args:
        k (int): Number of uids to return.
        exclude (List[int], optional): List of uids to exclude from the sampling. Defaults to None.
    
    Returns:
        torch.LongTensor: Sampled uids.
    """

    if exclude is None:
        exclude = set()
    else:
        exclude = set(exclude)

    candidate_uids = []
    for uid in range(self.metagraph.n.item()):
        if uid in exclude or uid == self.uid:
            continue

        if check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit):
            candidate_uids.append(uid)

    if len(candidate_uids) < k:
        k = len(candidate_uids)  # Adjust k to the number of available candidate uids

    verifier_count = determine_verifier_count(self.metagraph)
    if verifier_count == 0:
        raise ValueError("No verifiers available in metagraph")

    # Calculate requests needed based on step in the interval
    step_proportion = self.step / 360
    requests_needed = lambda uid: step_proportion * get_tier_requests(self, self.metagraph.hotkeys[uid]) / verifier_count

    # Sort candidate uids based on their requests needed, descending order
    candidate_uids.sort(key=requests_needed, reverse=True)

    # Select the top k uids
    selected_uids = candidate_uids[:k]

    return torch.tensor(selected_uids, dtype=torch.long)