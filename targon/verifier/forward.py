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
import asyncio
import bittensor as bt
from pprint import pformat

from targon.verifier.state import log_event
from targon.verifier.challenge import challenge_data
from targon.verifier.inference import inference_data
from targon.verifier.bonding import compute_all_tiers
from targon.verifier.database import get_prover_statistics, total_verifier_requests

async def forward(self):
    """
    Verifier forward pass. Consists of:
    - Generating the query
    - Querying the provers
    - Getting the responses
    - Rewarding the provers
    - Updating the scores
    """
    try:
        start_time = time.time()
        bt.logging.info(f"forward block: {self.block}")

        # --- Generate the query.
        event = await challenge_data(self)

        # --- Log the event
        log_event(self, event)

        if not self.config.mock:
            if self.block >= self.next_adjustment_block and self.step > 0:
                bt.logging.info("initiating compute stats")
                await compute_all_tiers(self.database)

                # Update prover statistics and usage data.
                stats = await get_prover_statistics(self.database)
                bt.logging.debug(f"prover stats: {pformat(stats)}")

                self.last_interval_block = self.get_last_adjustment_block()
                self.adjustment_interval = self.get_adjustment_interval()
                self.next_adjustment_block = self.last_interval_block + self.adjustment_interval
                self.step = 0
        else:
            if self.step % self.config.neuron.compute_stats_interval == 0 and self.step > 0:
                bt.logging.info("initiating compute stats")
                await compute_all_tiers(self.database)

                # Update prover statistics and usage data.
                stats = await get_prover_statistics(self.database)
                bt.logging.debug(f"prover stats: {pformat(stats)}")
                self.step = 0


        total_request_size = await total_verifier_requests(self.database)
        bt.logging.info(f"total verifier requests: {total_request_size}")
        try:
            block = self.substrate.subscribe_block_headers(self.subscription_handler)
        except:
            sleep_time = 12 - (time.time() - start_time)
            if sleep_time > 0:
                bt.logging.info(f"Sleeping for {sleep_time} seconds")
                await asyncio.sleep(sleep_time)
        
    except Exception as e:
        bt.logging.error(f"Error in forward: {e}")
        pass



