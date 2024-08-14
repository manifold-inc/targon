from functools import partial
import traceback
import time
from typing import Tuple
from starlette.types import Send
import json
from neurons.base import BaseNeuron, NeuronType
from targon.utils import print_info
import uvicorn
import bittensor as bt
from neurons.miner_app import app

from bittensor.axon import FastAPIThreadedServer
from targon.protocol import Inference


class Miner(BaseNeuron):
    neuron_type = NeuronType.Miner
    fast_api: FastAPIThreadedServer

    def shutdown(self):
        if self.fast_api:
            self.fast_api.stop()

    def __init__(self, config=None):
        super().__init__(config)
        ## Typesafety
        assert self.config.netuid
        assert self.config.neuron
        assert self.config.logging
        assert self.config.axon

        ## BITTENSOR INITIALIZATION
        self.axon = bt.axon(
            wallet=self.wallet,
            port=self.config.axon.port,
            external_ip=self.config.axon.external_ip,
            config=self.config,
        )
        bt.logging.info(
            "\N{grinning face with smiling eyes}", "Successfully Initialized!"
        )

    async def blacklist(self, synapse: Inference) -> Tuple[bool, str]:
        assert synapse.dendrite
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: Inference) -> float:
        assert synapse.dendrite
        assert synapse.dendrite.hotkey
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", str(priority)
        )
        return priority

    async def forward(self, synapse: Inference):
        bt.logging.info("\u2713", "Getting Inference request!")

        async def _prompt(synapse: Inference, send: Send) -> None:
            assert self.config.neuron
            assert synapse.sampling_params
            messages = json.loads(synapse.messages)
            stream = self.client.chat.completions.create(
                model=self.config.neuron.model_name,
                messages=messages,
                stream=True,
                temperature=synapse.sampling_params.temperature,
                top_p=synapse.sampling_params.top_p,
                seed=synapse.sampling_params.seed,
                timeout=5,
                max_tokens=synapse.sampling_params.max_new_tokens,
            )
            full_text = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_text += token
                    await send(
                        {
                            "type": "http.response.body",
                            "body": token.encode("utf-8"),
                            "more_body": True,
                        }
                    )
            bt.logging.info("\N{grinning face}", "Successful Prompt")

        token_streamer = partial(_prompt, synapse)
        return synapse.create_streaming_response(token_streamer)

    def run(self):
        assert self.config.netuid
        assert self.config.subtensor
        assert self.config.axon
        assert self.config.neuron

        # Check that miner is registered on the network.
        self.sync_metagraph()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.

        #TODO make this logging better
        bt.logging.info(
            f"Serving miner endpoint on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        serve_success = self.subtensor.serve(
            wallet=self.wallet,
            ip=self.config.axon.ip,
            port=self.config.axon.port,
            netuid=self.config.netuid,
            protocol=4,
        )
        if not serve_success:
            bt.logging.error("Failed to serve endpoint")
            return

        # Start  starts the miner's endpoint, making it active on the network.
        # change the config in the axon
        fast_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.config.axon.port,
            log_level="info",
            loop="asyncio",
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        bt.logging.info(f"Miner starting at block: {self.subtensor.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                # Print Logs for Miner
                print_info(
                    self.metagraph,
                    self.wallet.hotkey.ss58_address,
                    self.subtensor.block,
                )
                # Wait before checking again.
                time.sleep(12)

                # Sync metagraph if stale
                self.sync_metagraph()
        except Exception as e:
            bt.logging.error(str(e))
            bt.logging.error(traceback.format_exc())
        self.shutdown()


if __name__ == "__main__":
    try:
        miner = Miner()
        miner.run()
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
