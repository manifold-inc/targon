import os
import time
import torch
import shutil
import asyncio
import tempfile
import argparse
import bittensor as bt

from threading import Thread
from functools import partial
from starlette.types import Send
from targon.miner.miner import Miner 
from transformers import GPT2Tokenizer
from targon.protocol import TargonStreaming
from transformers import TextIteratorStreamer
from torchvision.transforms import ToPILImage, Resize, Compose
from code.model.anyToImageVideoAudio import NextGPTModel
from code.config import load_config
from transformers import StoppingCriteria, StoppingCriteriaList


class NextMiner( Miner ):
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
        parser.add_argument('--next.max_new_tokens', type=int, default=300, help='Maximum number of tokens to generate.')
        parser.add_argument('--next.num_beams', type=int, default=1, help='Number of beams to use for beam search.')
        parser.add_argument('--next.min_length', type=int, default=1, help='Minimum number of tokens to generate.')
        parser.add_argument('--next.top_p', type=float, default=0.9, help='Top p for nucleus sampling.')
        parser.add_argument('--next.repetition_penalty', type=float, default=1.0, help='Repetition penalty.')
        parser.add_argument('--next.length_penalty', type=float, default=1.0, help='Length penalty.')
        parser.add_argument('--next.temperature', type=float, default=1.0, help='Temperature for sampling.')
        parser.add_argument('--next.max_length', type=int, default=2000, help='Maximum number of tokens to generate.')
        parser.add_argument('--next.device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run the model on.')

    def __init__(self, *args, **kwargs):
        super(NextMiner, self).__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # get the directory this file is in
        base_path = os.path.dirname(os.path.realpath(__file__))
        g_cuda = torch.Generator(device='cuda').manual_seed(1337)
        args = {'model': 'nextgpt',
                'nextgpt_ckpt_path': os.path.join(base_path, 'tiva_v0'),
                'max_length': 128,
                'stage': 3,
                'root_dir': '../',
                'mode': 'validate',
                }
        args.update(load_config(args))

        self.model = NextGPTModel(**args)
        delta_ckpt = torch.load(os.path.join(args['nextgpt_ckpt_path'], 'pytorch_model.pt'), map_location=torch.device('cuda'))

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
        images = [bt.Tensor.deserialize(image) for image in synapse.images]
        decoded_tensor_list = []
        if len(synapse.images) > 0:
            image_list = []
            image_transform = Compose([
                ToPILImage(),
                Resize((224, 224))
            ])
            # to_pil_image = ToPILImage()
            # image_list = [image_transform(bt.Tensor.deserialize(image)) for image in synapse.images]
            # bt.logging.info('image detected!!!!!!', image_list[0].shape)

        
            chat_state = CONV_VISION.copy()

            # Deserialize the tensors, apply the transformation, and save to the temp directory
            for idx, image_tensor in enumerate(images):
                image = image_transform(image_tensor)
                self.chat.upload_img(image, chat_state, image_list)



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
            if len(chat_state.messages) > 0 and chat_state.messages[-1][0] == chat_state.roles[0] \
                    and chat_state.messages[-1][1][-6:] == '</Img>':  # last message is image.
                chat_state.messages[-1][1] = ' '.join([chat_state.messages[-1][1], text])
            else:
                chat_state.append_message(chat_state.roles[0], text)

            chat_state.append_message(chat_state.roles[1], None)
            embs = self.chat.get_context_emb(chat_state, image_list)

            current_max_len = embs.shape[1] + self.config.minigpt4.max_new_tokens
            if current_max_len > self.config.minigpt4.max_length:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                    'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - self.config.minigpt4.max_length)

            embs = embs[:, begin_idx:]

            streamer = TextIteratorStreamer(self.model.llama_tokenizer)

            generation_kwargs = dict(streamer=streamer,
                inputs_embeds=embs,
                max_new_tokens=self.config.minigpt4.max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=self.config.minigpt4.num_beams,
                do_sample=True,
                min_length=self.config.minigpt4.min_length,
                top_p=self.config.minigpt4.top_p,
                repetition_penalty=self.config.minigpt4.repetition_penalty,
                length_penalty=self.config.minigpt4.length_penalty,
                temperature=self.config.minigpt4.temperature)

            thread = Thread(target=self.model.llama_model.generate, kwargs=generation_kwargs)
            thread.start()

            buffer = []
            output_text = ""
            for token in streamer:
                output_text += token

                
                N = 3  # Number of tokens to send back to the client at a time
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
                    # await asyncio.sleep(0.08)  # Simulate streaming delay
            
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
    bt.debug()
    with NextMiner():
        while True:
            print("running...", time.time())
            time.sleep(1)