import torch
import targon
import asyncio
import bittensor as bt
from traceback import print_exception

from targon import autoupdate
from targon.validator import should_checkpoint, checkpoint, should_reinit_wandb, reinit_wandb, load_state, save_state, ttl_get_block
from .forward import forward_fn
# from openvalidators.weights import should_set_weights, set_weights

def should_set_weights(self) -> bool:
    # Check if enough epoch blocks have elapsed since the last epoch.
    if self.config.neuron.disable_set_weights:
        return False

    return ttl_get_block(self) % self.config.neuron.epoch_length < self.prev_block % self.config.neuron.epoch_length


def set_weights(self):
    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(self.moving_averaged_scores, p=1, dim=0)
    bt.logging.trace("raw_weights", raw_weights)
    bt.logging.trace("top10 values", raw_weights.sort()[0])
    bt.logging.trace("top10 uids", raw_weights.sort()[1])

    # Process the raw weights to final_weights via subtensor limitations.
    (processed_weight_uids, processed_weights,) = bt.utils.weight_utils.process_weights_for_netuid(
        uids=self.metagraph.uids.to("cpu"),
        weights=raw_weights.to("cpu"),
        netuid=self.config.netuid,
        subtensor=self.subtensor,
        metagraph=self.metagraph,
    )
    bt.logging.trace("processed_weights", processed_weights)
    bt.logging.trace("processed_weight_uids", processed_weight_uids)

    # Set the weights on chain via our subtensor connection.
    self.subtensor.set_weights(
        wallet=self.wallet,
        netuid=self.config.netuid,
        uids=processed_weight_uids,
        weights=processed_weights,
        wait_for_finalization=False,
        version_key=targon.__spec_version__,
    )

# Neuron run loop.`
def run(self):
    bt.logging.info("run()")
    load_state(self)
    checkpoint(self)
    try:
        while True:
            bt.logging.info(f"step({self.step}) block({ttl_get_block( self )})")

            # Run multiple forwards.
            async def run_forward():
                await forward_fn(self)

            self.loop.run_until_complete(run_forward())
            
            # Resync the network state
            if should_checkpoint(self):
                checkpoint(self)

            # Set the weights on chain.
            if should_set_weights(self):
                set_weights(self)
                save_state(self)
            
            # Check if we should update.
            autoupdate()

            # Rollover wandb to a new run.
            if should_reinit_wandb(self):
                reinit_wandb(self)

            self.prev_block = ttl_get_block(self)
            self.step += 1

    except Exception as e:
        bt.logging.error("Error in training loop", str(e))
        bt.logging.debug(print_exception(value=e))