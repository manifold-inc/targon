import os
import time
import torch
import asyncio
import argparse
import bittensor as bt

from functools import partial
from starlette.types import Send
from min.minigpt4 import MiniGPT4
from targon.miner.miner import Miner 
from transformers import GPT2Tokenizer
from targon.protocol import TargonStreaming
from min.conversation import Chat, CONV_VISION
from min.blip_processor import Blip2ImageEvalProcessor



class MiniGPT4Miner( Miner ):
    def config(self) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        self.add_args(parser)
        return bt.config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds custom arguments to the command line parser.

        Developers can introduce additional command-line arguments specific to the miner's
        functionality in this method. These arguments can then be used to configure the miner's operation.

        Args:
            parser (argparse.ArgumentParser):
                The command line argument parser to which custom arguments should be added.
        """
        pass

    def __init__(self, *args, **kwargs):
        super(MiniGPT4Miner, self).__init__(*args, **kwargs)
        
        # get the directory this file is in
        base_path = os.path.dirname(os.path.realpath(__file__))

        self.model = MiniGPT4(
            vision_model_path=os.path.join(base_path, "models/eva_vit_g.pth"), #"models/eva_vit_g.pth",
            llama_model=os.path.join(base_path, "models/vicuna13b_v0/"),
            q_former_model=os.path.join(base_path, "models/blip2_pretrained_flant5xxl.pth"),
        )

        # ckpt_path = "models/pretrained_minigpt4.pth"
        ckpt_path = os.path.join(base_path, "models/pretrained_minigpt4.pth")

        print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt['model'], strict=False)

        torch.compile(self.model)

        self.vis_processor = Blip2ImageEvalProcessor()

        self.chat = Chat(self.model, self.vis_processor, device='cuda:0')
        bt.logging.info('model loaded, ready to go!')

    def prompt(self, synapse: TargonStreaming) -> TargonStreaming:
        """
        Generates a streaming response for the provided synapse.

        This function serves as the main entry point for handling streaming prompts. It takes
        the incoming synapse which contains messages to be processed and returns a streaming
        response. The function uses the GPT-2 tokenizer and a simulated model to tokenize and decode
        the incoming message, and then sends the response back to the client token by token.

        Args:
            synapse (TargonStreaming): The incoming TargonStreaming instance containing the messages to be processed.

        Returns:
            TargonStreaming: The streaming response object which can be used by other functions to
                            stream back the response to the client.

        Usage:
            This function can be extended and customized based on specific requirements of the
            miner. Developers can swap out the tokenizer, model, or adjust how streaming responses
            are generated to suit their specific applications.
        """

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        bt.logging.info('synapse', synapse)

        # Simulated function to decode token IDs into strings. In a real-world scenario,
        # this can be replaced with an actual model inference step.
        def model(ids):
            return (tokenizer.decode(id) for id in ids)

        async def _prompt(text: str, send: Send):
            """
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
            input_ids = tokenizer(text, return_tensors="pt").input_ids.squeeze()
            buffer = []
            N = 3  # Number of tokens to send back to the client at a time
            for token in model(input_ids):
                buffer.append(token)
                # If buffer has N tokens, send them back to the client.
                if len(buffer) == N:
                    joined_buffer = "".join(buffer)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    bt.logging.debug(f"Streamed tokens: {joined_buffer}")
                    buffer = []  # Clear the buffer for next batch of tokens

            # Send any remaining tokens in the buffer
            if buffer:
                joined_buffer = "".join(buffer)
                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": False,  # No more tokens to send
                    }
                )
                bt.logging.trace(f"Streamed tokens: {joined_buffer}")

        message = synapse.messages[0]
        token_streamer = partial(_prompt, message)
        return synapse.create_streaming_response(token_streamer)



if __name__ == "__main__":
    """
    Entry point for executing the StreamingTemplateMiner.

    This block initializes the StreamingTemplateMiner and runs it, effectively connecting
    it to the Bittensor network. Once connected, the miner will continuously listen for
    incoming requests from the Bittensor network. For every request, it responds with a
    static message processed as per the logic defined in the 'prompt' method of the
    StreamingTemplateMiner class.

    The main loop at the end serves to keep the miner running indefinitely. It periodically
    prints a "running..." message to the console, providing a simple indication that the miner
    is operational and active.

    Developers looking to extend or customize the miner's behavior can modify the
    StreamingTemplateMiner class and its methods. However, this block itself usually
    remains unchanged unless there's a need for specific startup behaviors or configurations.

    To start the miner:
    Simply execute this script. Ensure all dependencies are properly installed and network
    configurations are correctly set up.
    """
    bt.debug()
    with MiniGPT4Miner():
        while True:
            print("running...", time.time())
            time.sleep(1)

# def main():
#     print("Loading models...")

#     t0 = time.time()

#     model = MiniGPT4(
#         vision_model_path="models/eva_vit_g.pth",
#         llama_model="models/vicuna13b_v0/",
#         q_former_model="models/blip2_pretrained_flant5xxl.pth",
#     )

#     ckpt_path = "models/pretrained_minigpt4.pth"

#     print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     model.load_state_dict(ckpt['model'], strict=False)

#     torch.compile(model)

#     vis_processor = Blip2ImageEvalProcessor()

#     chat = Chat(model, vis_processor, device='cuda:0')

#     t1 = time.time()

#     print("Models loaded in {} seconds".format(t1-t0))

#     for i in range(5):
#         print("Loading image...")

#         t0 = time.time()

#         chat_state = CONV_VISION.copy()
#         img_list = []
#         chat.upload_img("icbm_bicycle.png", chat_state, img_list)

#         t1 = time.time()

#         print("Image loaded in {} seconds".format(t1-t0))

#         t0 = time.time()

#         num_beams = 1
#         temperature = 0.01

#         chat.ask("Tell me what you see on the road.", chat_state)

#         # Callback for each word generated by the LLM
#         def callback_function(word):
#             print(word, end='', flush=True)

#         print("Live output: ", end='', flush=True)

#         output_text = chat.answer_async(conv=chat_state,
#                                     img_list=img_list,
#                                     num_beams=num_beams,
#                                     temperature=temperature,
#                                     max_new_tokens=1024,
#                                     max_length=2048,
#                                     text_callback=callback_function)

#         print("")

#         t1 = time.time()

#         print("LLM response: {}".format(output_text))
#         print(chat_state)
#         print("Generated LLM response in {} seconds".format(t1-t0))

# if __name__ == "__main__":
#     main()