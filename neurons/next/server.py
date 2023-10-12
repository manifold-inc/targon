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
from typing import List, Tuple, Any


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


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
        parser.add_argument('--next.max_tgt_len', type=int, default=150)
        parser.add_argument('--next.top_p', type=float, default=1.0)
        parser.add_argument('--next.temperature', type=float, default=0.4)
        parser.add_argument('--next.modality_cache', type=str, default=None)
        parser.add_argument('--next.filter_value', type=float, default=-float('Inf'))
        parser.add_argument('--next.min_word_tokens', type=int, default=10)
        parser.add_argument('--next.gen_scale_factor', type=float, default=4.0)
        parser.add_argument('--next.max_num_imgs', type=int, default=1)
        parser.add_argument('--next.stops_id', type=list, default=[[835]])
        parser.add_argument('--next.load_sd', type=bool, default=True)
        parser.add_argument('--next.guidance_scale_for_img', type=float, default=7.5)
        parser.add_argument('--next.num_inference_steps_for_img', type=int, default=50)
        parser.add_argument('--next.guidance_scale_for_vid', type=float, default=7.5)
        parser.add_argument('--next.num_inference_steps_for_vid', type=int, default=50)
        parser.add_argument('--next.max_num_vids', type=int, default=1)
        parser.add_argument('--next.height', type=int, default=320)
        parser.add_argument('--next.width', type=int, default=576)
        parser.add_argument('--next.num_frames', type=int, default=24)
        parser.add_argument('--next.guidance_scale_for_aud', type=float, default=7.5)
        parser.add_argument('--next.num_inference_steps_for_aud', type=int, default=50)
        parser.add_argument('--next.max_num_auds', type=int, default=1)
        parser.add_argument('--next.audio_length_in_s', type=int, default=4)
        parser.add_argument('--next.ENCOUNTERS', type=int, default=1)


    def __init__(self, *args, **kwargs):
        super(NextMiner, self).__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # get the directory this file is in
        base_path = os.path.dirname(os.path.realpath(__file__))
        self.g_cuda = torch.Generator(device='cuda').manual_seed(1337)
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

        self.model.load_state_dict(delta_ckpt, strict=False)
        self.model = self.model.eval().half().cuda()

        self.max_tgt_length = 150
        self.top_p = 1.0
        self.temperature = 0.4
        self.modality_cache = None


        self.history = []
        bt.logging.success('Loaded nextgpt from: {}'.format(args['nextgpt_ckpt_path']))

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
        # images = [bt.Tensor.deserialize(image) for image in synapse.images]
        # decoded_tensor_list = []
        # if len(synapse.images) > 0:
        #     image_list = []
        #     image_transform = Compose([
        #         ToPILImage(),
        #         Resize((224, 224))
        #     ])
        #     # to_pil_image = ToPILImage()
        #     # image_list = [image_transform(bt.Tensor.deserialize(image)) for image in synapse.images]
        #     # bt.logging.info('image detected!!!!!!', image_list[0].shape)

        
        #     chat_state = CONV_VISION.copy()

        #     # Deserialize the tensors, apply the transformation, and save to the temp directory
        #     for idx, image_tensor in enumerate(images):
        #         image = image_transform(image_tensor)
        #         self.chat.upload_img(image, chat_state, image_list)


        image_path = None
        audio_path = None
        video_path = None
        thermal_path = None


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


            inputs = {
                    'prompt': text,
                    'image_paths': [image_path] if image_path else [],
                    'audio_paths': [audio_path] if audio_path else [],
                    'video_paths': [video_path] if video_path else [],
                    'thermal_paths': [thermal_path] if thermal_path else [],
                    'top_p': self.config.next.top_p,
                    'temperature': self.config.next.temperature,
                    'max_tgt_len': self.config.next.max_tgt_len,
                    'modality_embeds': self.config.next.modality_cache,
                    'filter_value': self.config.next.filter_value, 'min_word_tokens': self.config.next.min_word_tokens,
                    'gen_scale_factor': self.config.next.gen_scale_factor, 'max_num_imgs': self.config.next.max_num_imgs,
                    'stops_id': self.config.next.stops_id,
                    'load_sd': self.config.nextload_sd,
                    'generator': self.g_cuda,
                    'guidance_scale_for_img': self.config.next.guidance_scale_for_img,
                    'num_inference_steps_for_img': self.config.next.num_inference_steps_for_img,

                    'guidance_scale_for_vid': self.config.next.guidance_scale_for_vid,
                    'num_inference_steps_for_vid': self.config.next.num_inference_steps_for_vid,
                    'max_num_vids': self.config.next.max_num_vids,
                    'height': self.config.next.height,
                    'width': self.config.next.width,
                    'num_frames': self.config.next.num_frames,

                    'guidance_scale_for_aud': self.config.next.guidance_scale_for_aud,
                    'num_inference_steps_for_aud': self.config.next.num_inference_steps_for_aud,
                    'max_num_auds': self.config.next.max_num_auds,
                    'audio_length_in_s': self.config.next.audio_length_in_s,
                    'ENCOUNTERS': self.config.next.ENCOUNTERS,

                }

            input_embeds = self.model.prepare_generation_embedding(inputs)
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=inputs['stops_id'], encounters=1)])


            streamer = TextIteratorStreamer(self.model.llama_tokenizer)

            # generation_kwargs = dict(
            #     streamer=streamer,
            #     inputs_embeds=input_embeds,
            #     max_new_tokens=inputs['max_tgt_len'],
            #     top_p=inputs['top_p'],
            #     temperature=inputs['temperature'],
            #     # repeat_pen,
            #     do_sample=True,
            #     use_cache=True,
            #     stopping_criteria=stopping_criteria,
            #     output_hidden_states=True,
            #     return_dict_in_generate=True,
            #     output_attentions=True
            # )

            generation_kwargs = dict(
                streamer=streamer,
                inputs_embeds=input_embeds,
                max_new_tokens=inputs['max_tgt_len'],
                stopping_criteria=stopping_criteria,
                do_sample=True,
                min_length=inputs['min_word_tokens'],
                top_p=inputs['top_p'],
                temperature=inputs['temperature'],
                repetition_penalty=1.0,
                length_penalty=1.0,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_attentions=True
            )


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