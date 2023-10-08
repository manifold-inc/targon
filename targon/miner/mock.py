# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2023 Manifold Labs

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
import json
import bittensor as bt
from typing import List, Optional


class MockSubtensor:
    def __init__(self, config: "bt.Config"):
        self.config = config
        self.mock_metagraph = bt.subtensor(self.config).metagraph(config.netuid)
        self.start_time = time.time()

    def serve_axon(self, netuid: int, axon: "bt.axon"):
        return True

    def register(self, netuid: int, wallet: "bt.Wallet"):
        return True

    def get_current_block(self):
        return (
            int((time.time() - self.start_time) / 12) + self.mock_metagraph.block.item()
        )

    def metagraph(self, netuid: int) -> "bt.Metagraph":
        return self.mock_metagraph
