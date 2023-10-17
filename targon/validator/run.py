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

import asyncio
import bittensor as bt
from traceback import print_exception

from targon.validator.forward import forward
from targon.validator.utils import should_checkpoint, checkpoint, should_reinit_wandb, reinit_wandb, load_state, save_state
from targon.validator.weights import should_set_weights, set_weights
from targon.validator.misc import ttl_get_block

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
                coroutines = [forward(self) for _ in range(self.config.neuron.num_concurrent_forwards)]
                await asyncio.gather(*coroutines)

            self.loop.run_until_complete(run_forward())

            # Resync the network state
            if should_checkpoint(self):
                checkpoint(self)

            # Set the weights on chain.
            if should_set_weights(self):
                set_weights(self)
                save_state(self)

            # Rollover wandb to a new run.
            if should_reinit_wandb(self):
                reinit_wandb(self)

            self.prev_block = ttl_get_block(self)
            self.step += 1

    except Exception as e:
        bt.logging.error("Error in training loop", str(e))
        bt.logging.debug(print_exception(value=e))
