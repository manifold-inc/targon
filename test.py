import asyncio
import bittensor as bt

from targon.protocol import TargonStreaming
subtensor = bt.subtensor( network = 'finney' )
metagraph = subtensor.metagraph( netuid = 4 )

# find all hotkeys with an axon ip

bt.debug()
wallet = bt.wallet( name="lilith", hotkey="A4" )

dendrite = bt.dendrite( wallet = wallet )

prompt = """
            Asynchronously processes the input text and sends back tokens as a streaming response.

            This function takes an input text, tokenizes it using the GPT-2 tokenizer, and then
            uses the simulated model to decode token IDs into strings. It then sends each token
            back to the client as a streaming response, with a delay between tokens to simulate
            the effect of real-time streaming.

            Args:
                text (str): The input text message to be processed.
                send (Send): An asynchronous function that allows sending back the streaming response.

            Usage:
                This function can be adjusted based on the streaming requirements, speed of
                response, or the model being used. Developers can also introduce more sophisticated
                processing steps or modify how tokens are sent back to the client.
            """

# find all hotkeys with an axon ip that is not none

axons = [axon for axon in metagraph.axons if axon.ip == '184.105.4.10']
synapse = TargonStreaming(roles=['user'], messages=[prompt])


async def fetch():
    responses = await dendrite(
        axons=axons,
        synapse=synapse,
        timeout=10
    )

    # r = await synapse.process_streaming_response(responses)
    return responses

import code; code.interact(local=dict(globals(), **locals()))
responses = asyncio.run(fetch())

print(responses)