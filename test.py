import asyncio
import bittensor as bt

from targon.protocol import TargonStreaming
subtensor = bt.subtensor( network = 'finney' )
metagraph = subtensor.metagraph( netuid = 4 )

# find all hotkeys with an axon ip


wallet = bt.wallet( name="targon" )

dendrite = bt.dendrite( wallet = wallet )

prompt = 'ding ding'

# find all hotkeys with an axon ip that is not none

axons = [axon for axon in metagraph.axons if axon.ip == '141.193.30.26']
synapse = TargonStreaming(roles=['user'], messages=[prompt])


async def fetch():
    responses = await dendrite(
        axons=axons,
        synapse=synapse,
        timeout=10
    )

    return responses


asyncio.run(fetch())