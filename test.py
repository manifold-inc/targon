import asyncio
import bittensor as bt
import torchvision.transforms as transforms
from PIL import Image

from targon.protocol import TargonStreaming, TargonDendrite
subtensor = bt.subtensor( network = 'finney' )
metagraph = subtensor.metagraph( netuid = 4 )

# find all hotkeys with an axon ip

bt.debug()
wallet = bt.wallet( name="lilith", hotkey="A4" )

dendrite = TargonDendrite( wallet = wallet )

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
# Open the image
image_path = "neurons/miniGPT4/icbm_bicycle.png"
image = Image.open(image_path)

# Convert the image to a tensor, then convert to float and scale to [0, 1]
tensor_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Lambda(lambda x: x.float() / 255.0)
])
image_tensor_float = tensor_transform(image)

# Now normalize
normalized_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
normalized_image_tensor = normalized_transform(image_tensor_float)

serialized_tensor = bt.Tensor.serialize(normalized_image_tensor)

axons = [axon for axon in metagraph.axons if axon.ip == '184.105.87.189']


synapse = TargonStreaming(roles=['user'], messages=[prompt], images=[serialized_tensor])


async def fetch():
    responses = await dendrite(
        axons=axons,
        synapse=synapse,
        timeout=60,
        streaming=True
    )
    async for token in responses:
        print(token, end="", flush=True)  # or handle the token as needed

    return responses


# def model():
#     for token in dendrite(axons=axons, synapse=synapse, timeout=60):
#         print( token )


asyncio.run(fetch())

# dendrite(axons=axons, synapse=synapse, timeout=60)
import code; code.interact(local=dict(globals(), **locals()))
# responses = asyncio.run(fetch())

# print(responses)