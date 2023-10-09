import asyncio
import bittensor as bt

from targon.protocol import TargonStreaming
subtensor = bt.subtensor( network = 'finney' )
metagraph = subtensor.metagraph( netuid = 4 )

# find all hotkeys with an axon ip

bt.debug()
wallet = bt.wallet( name="lilith", hotkey="A4" )

dendrite = bt.dendrite( wallet = wallet )

prompt = 'ding ding'

# find all hotkeys with an axon ip that is not none

axons = [axon for axon in metagraph.axons if axon.ip == '184.105.4.10']
synapse = TargonStreaming(roles=['user'], messages=[prompt])


async def fetch():
    responses = await dendrite(
        axons=axons,
        synapse=synapse,
        timeout=10
    )

    return responses

import code; code.interact(local=dict(globals(), **locals()))
responses = asyncio.run(fetch())

print(responses)