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
import random
import asyncio
import bittensor as bt
from pprint import pformat

from targon.verifier.inference import inference_data

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
        bt.logging.info(f"forward block: {self.block if not self.config.mock else self.block_number} step: {self.step}")

        # --- Perform coin flip

            # --- Generate the query.
        event = await inference_data(self)
        

        if not self.config.mock:
            try:
                block = self.substrate.subscribe_block_headers(self.subscription_handler)
            except:
                sleep_time = 12 - (time.time() - start_time)
                if sleep_time > 0:
                    bt.logging.info(f"Sleeping for {sleep_time} seconds")
                    await asyncio.sleep(sleep_time)
        else:
            time.sleep(1)
        
    except Exception as e:
        bt.logging.error(f"Error in forward: {e}")
        time.sleep(12)
        pass




