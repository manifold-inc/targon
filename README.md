# Targon - A Bittensor Subnetwork

Project Targon - is a subnetwork of the bittensor finney protocol. It is designed to faciliate the inference of multi modality LLMs for high throughput search.

## Installation
```bash
pip install -e .
```

## Experimental
This project is currently in an experimental state with rapid development occuring. 

## Running

### Run as a validator
```bash
python targon/validator/neuron.py --wallet.name YOUR_NAME --wallet.hotkey YOUR_HOTKEY --netuid 4
```

### Run as a miner

run the model endpoint script so you can run vllm endpoint for the model. remember to change the --sybil.api_url to the correct url for your vllm instance.
```bash
./miners/sybil/model_endpoint.sh
```
then once this is running, start the miner.

If you are wanting to run link prediction right now (currently experimental and not validated, include your serp api key like --sybil.serp_api_key)
```bash
python miners/sybil/server.py --wallet.name YOUR_NAME --wallet.hotkey YOUR_HOTKEY --netuid 4 --sybil.api_url YOUR_URL 
```

### usage:

```python
import asyncio
import bittensor as bt
import torchvision.transforms as transforms
from PIL import Image

from targon.protocol import TargonQA, TargonLinkPrediction, TargonSearchResult, TargonDendrite, TargonSearchResultStream
subtensor = bt.subtensor( network = 'finney' )
metagraph = subtensor.metagraph( netuid = 4 )

# find all hotkeys with an axon ip

bt.debug()
bt.trace()
wallet = bt.wallet( name="YOUR_NAME", hotkey="HOTKEY_NAME" )

dendrite = TargonDendrite( wallet = wallet )

async def fetch(synapse):
    responses = await dendrite(
        axons=axons,
        synapse=synapse,
        timeout=60,
        streaming=synapse.stream
    )


    if type(synapse) == TargonSearchResultStream:
        async for token in responses:
            print(token, end="", flush=True)
    # else:
    #     print(responses)
    # async for token in responses:
    #     print(token, end="", flush=True)  # or handle the token as needed

    return responses



axons = [axon for axon in metagraph.axons]

question = "what is happening in the sbf trial?"

sources_synapse = TargonLinkPrediction( query=question )

sources_response = asyncio.run(fetch(sources_synapse))
sources = sources_response[0].results

search_synapse = TargonSearchResultStream( query=question, sources=sources, stream=True )

asyncio.run(fetch(search_synapse))
```