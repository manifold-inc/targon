
import time
import asyncio
import bittensor as bt
from pprint import pformat

from targon.verifier.state import log_event
from targon.verifier.challenge import challenge_data
from targon.verifier.bonding import compute_all_tiers
from targon.verifier.database import get_prover_statistics, total_verifier_requests

def subscribe_to_next_block(self):
    """
    Subscribes to block headers and waits for the next block.
    Returns a future that gets completed upon the arrival of the next block.
    """
    future = asyncio.Future()

    def next_block_handler(obj, update_nr, subscription_id):
        future.set_result(obj["header"]["number"])
        self.subscription_substrate.unsubscribe_block_headers(subscription_id)

    self.subscription_substrate.subscribe_block_headers(next_block_handler)
    return future

async def forward(self):
    """
    Verifier forward pass. Consists of:
    - Generating the query
    - Querying the provers
    - Getting the responses
    - Rewarding the provers
    - Updating the scores
    """
    bt.logging.info(f"forward step: {self.step}")

    # --- Generate the query.
    event = await challenge_data(self)

    # --- Log the event
    log_event(self, event)

    if self.block >= self.next_adjustment_block and self.step > 0:
        bt.logging.info("initiating compute stats")
        await compute_all_tiers(self.database)

        # Update miner statistics and usage data.
        stats = await get_prover_statistics(self.database)
        bt.logging.debug(f"miner stats: {pformat(stats)}")

        self.last_interval_block = self.get_last_adjustment_block()
        self.adjustment_interval = self.get_adjustment_interval()
        self.next_adjustment_block = self.last_interval_block + self.adjustment_interval


    total_request_size = await total_verifier_requests(self.database)
    bt.logging.info(f"total verifier requests: {total_request_size}")
    
    next_block_number = await self.subscribe_to_next_block()
    bt.logging.info(f"Next block arrived: {next_block_number}")


