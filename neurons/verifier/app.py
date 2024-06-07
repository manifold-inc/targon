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

import os
import time
from bittensor.axon import FastAPIThreadedServer
import uvicorn
import bittensor as bt

from targon import protocol
from targon.verifier.forward import forward
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from targon.base.verifier import BaseVerifierNeuron
from targon.verifier.inference import api_chat_completions
from targon.verifier.uids import check_uid_availability
from fastapi import Request
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("HUB_SECRET_TOKEN")


class Verifier(BaseVerifierNeuron):
    """
    Text prompt verifier neuron.
    """

    async def safeParseAndCall(self, req: Request):
        data = await req.json()
        if data.get("api_key") != TOKEN and TOKEN is not None:
            return "", 401

        print("Received an organic request!")
        messages = data.get("messages")
        if not isinstance(messages, list):
            return "", 403
        prompt = "\n".join([msg["role"] + ": " + msg["content"] for msg in messages])
        try:
            return EventSourceResponse(
                api_chat_completions(
                    self,
                    prompt,
                    protocol.InferenceSamplingParams(
                        max_new_tokens=data.get("max_tokens", 1024)
                    ),
                ),
                media_type="text/event-stream",
            )
        except Exception as e:
            print(f"Failed due to: {e}")
            return "", 500

    def __init__(self, config=None):
        super(Verifier, self).__init__(config=config)

        self.restart_required = False

        bt.logging.info("load_state()")
        if not self.config.mock:
            self.load_state()
            for i, axon in enumerate(self.metagraph.axons):
                bt.logging.info(f"axons[{i}]: {axon}")
                check_uid_availability(
                    self.metagraph, i, self.config.neuron.vpermit_tao_limit
                )

        # inference client
        # --- Block
        self.app = FastAPI()
        self.app.router.add_api_route(
            "/api/chat/completions", self.safeParseAndCall, methods=["POST"]
        )
        self.fast_config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.config.neuron.proxy.port
        )
        self.fast_server = FastAPIThreadedServer(config=self.fast_config)
        self.fast_server.start()

        self.last_interval_block = self.get_last_adjustment_block()
        self.adjustment_interval = self.get_adjustment_interval()
        self.next_adjustment_block = self.last_interval_block + self.adjustment_interval

    async def forward(self):
        """
        Verifier forward pass. Consists of:
        - Generating the query
        - Querying the provers
        - Getting the responses
        - Rewarding the provers
        - Updating the scores
        """
        print("forward()")
        if self.config.neuron.api_only:
            bt.logging.info("Running in API only mode, sleeping for 12 seconds.")
            time.sleep(12)
            return
        return await forward(self)

    def __enter__(self):
        if self.config.no_background_thread:
            bt.logging.warning("Running verifier in main thread.")
            self.run()
        else:
            self.run_in_background_thread()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the verifier's background operations upon exiting the context.
        This method facilitates the use of the verifier in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping verifier in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")


# The main function parses the configuration and runs the verifier.
if __name__ == "__main__":
    with Verifier() as verifier:
        while True:
            bt.logging.info("Verifier running...", time.time())
            time.sleep(5)
