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
wallet = bt.wallet( name="lilith", hotkey="A4" )

dendrite = TargonDendrite( wallet = wallet )

prompt = """describe this image."""

# # find all hotkeys with an axon ip that is not none
# # Open the image
# image_path = "neurons/miniGPT4/icbm_bicycle.png"
# image = Image.open(image_path)

# # Convert the image to a tensor, then convert to float and scale to [0, 1]
# tensor_transform = transforms.Compose([
#     transforms.PILToTensor(),
#     transforms.Lambda(lambda x: x.float() / 255.0)
# ])
# image_tensor_float = tensor_transform(image)

# # Now normalize
# normalized_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# normalized_image_tensor = normalized_transform(image_tensor_float)

# serialized_tensor = bt.Tensor.serialize(normalized_image_tensor)

axons = [axon for axon in metagraph.axons if axon.ip == '160.202.128.179']

synapse = TargonQA(question=prompt)



# synapse = TargonLinkPrediction( query="who killed dumbledore?" )
# synapse = TargonSearchResult( query="who killed dumbledore?", sources=["http://"] )
synapse = TargonSearchResultStream( query="Q: who killed dumbledore?\nA:", sources=["http://"], stream=True )

# synapse = TargonSearchResultStream(roles=['user'], messages=[prompt])

async def fetch():
    responses = await dendrite(
        axons=axons,
        synapse=synapse,
        timeout=60,
        streaming=synapse.stream
    )


    if type(synapse) == TargonSearchResultStream:
        async for token in responses:
            print(token, end="", flush=True)
    else:
        print(responses)
    # async for token in responses:
    #     print(token, end="", flush=True)  # or handle the token as needed

    return responses


# def model():
#     for token in dendrite(axons=axons, synapse=synapse, timeout=60):
#         print( token )


asyncio.run(fetch())

# dendrite(axons=axons, synapse=synapse, timeout=60)
# import code; code.interact(local=dict(globals(), **locals()))
# responses = asyncio.run(fetch())

# print(responses)

