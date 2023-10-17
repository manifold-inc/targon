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
# DEALINGS IN THE SOFTWARE.

import torch
import asyncio
import bittensor as bt
from typing import List


class AsyncDendritePool:
    def __init__(self, keypair, metagraph):
        self.dendrites = [bt.text_prompting(axon=axon, keypair=keypair, uid=uid) for uid, axon in enumerate(metagraph.axons)]

    async def async_forward(
        self,
        uids: List[int],
        roles: List[str],
        messages: List[str],
        timeout: float = 12,
        return_call=True,
    ):
        # The following asyncio definition queries a single endpoint with the message
        # prompt and returns the response.
        def call_single_uid(uid: int):
            return self.dendrites[uid].async_forward(roles=roles, messages=messages, return_call=return_call, timeout=timeout)

        # The following asyncio definition gathers the responses
        # from multiple coroutines for each uid.
        async def query():
            coroutines = [call_single_uid(uid) for uid in uids]
            all_responses = await asyncio.gather(*coroutines)
            return all_responses

        return await query()

    async def async_backward(
        self,
        uids: List[int],
        roles: List[str],
        messages: List[str],
        completions: List[str],
        rewards: torch.FloatTensor,
        timeout: float = 1,  # response not required
    ):
        # Async call single endpoint with reward for its completion.
        def call_single_uid(uid: int, completion: str, reward: torch.Tensor):
            return self.dendrites[uid].async_backward(
                roles=roles, messages=messages, completion=completion, rewards=[reward.item()], timeout=timeout
            )

        # The following asyncio definition gathers the responses
        # from multiple coroutines for each uid.
        async def query():
            coroutines = [
                call_single_uid(uid, completion, reward) for uid, completion, reward in zip(uids, completions, rewards)
            ]
            all_responses = await asyncio.gather(*coroutines)
            return all_responses

        return await query()

    def resync(self, metagraph: "bt.metagraph.Metagraph"):
        """
        Resyncs the dendrites with the latest updated version of the metagraph.

        Parameters:
            metagraph (:`bittensor.metagraph.Metagraph`): latest updated version of the metagraph.
        """
        metagraph_uids_axons_state = dict(zip(metagraph.uids.tolist(), metagraph.axons))

        # Iterate over all dendrites and check if the metagraph has updated.
        for index, dendrite in enumerate(self.dendrites):
            current_uid_axon_info = metagraph_uids_axons_state.pop(dendrite.uid)

            if dendrite.axon_info != current_uid_axon_info:
                self.dendrites[index] = bt.text_prompting(
                    axon=current_uid_axon_info,
                    keypair=dendrite.keypair,
                    uid=dendrite.uid,
                )

        # If there are any new axons, add them to the dendrites.
        if len(metagraph_uids_axons_state) > 0:
            for uid, axon in metagraph_uids_axons_state.items():
                self.dendrites.append(bt.text_prompting(axon=axon, keypair=self.dendrites[0].keypair, uid=uid))
