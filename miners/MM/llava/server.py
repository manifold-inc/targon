import os
import time
import torch
import shutil
import asyncio
import tempfile
import argparse
import traceback
import bittensor as bt

from peft import PeftModel
from threading import Thread
from functools import partial
from starlette.types import Send
from targon.miner.miner import Miner 
from transformers import GPT2Tokenizer
from targon.protocol import TargonStreaming
from transformers import TextIteratorStreamer
from llava.conversation import default_conversation, conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from torchvision.transforms import ToPILImage, Resize, Compose
from transformers import StoppingCriteria, StoppingCriteriaList
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN



class LlavaMiner( Miner ):
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
        parser.add_argument('--llava.max_new_tokens', type=int, default=300, help='Maximum number of tokens to generate.')
        parser.add_argument('--llava.num_beams', type=int, default=1, help='Number of beams to use for beam search.')
        parser.add_argument('--llava.min_length', type=int, default=1, help='Minimum number of tokens to generate.')
        parser.add_argument('--llava.top_p', type=float, default=0.9, help='Top p for nucleus sampling.')
        parser.add_argument('--llava.repetition_penalty', type=float, default=1.0, help='Repetition penalty.')
        parser.add_argument('--llava.length_penalty', type=float, default=1.0, help='Length penalty.')
        parser.add_argument('--llava.temperature', type=float, default=1.0, help='Temperature for sampling.')
        parser.add_argument('--llava.max_length', type=int, default=2000, help='Maximum number of tokens to generate.')
        parser.add_argument('--llava.device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run the model on.')
        parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-13b")
        parser.add_argument("--model_base", type=str, default=None)
        parser.add_argument("--model_name", type=str, default="liuhaotian/llava-v1.5-13b")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--multi_modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
        parser.add_argument("--limit_model_concurrency", type=int, default=5)
        parser.add_argument("--stream_interval", type=int, default=1)
        parser.add_argument("--no_register", action="store_true")
        parser.add_argument("--load_8bit", action="store_true")
        parser.add_argument("--load_4bit", action="store_true")

    def __init__(self, *args, **kwargs):
        super(LlavaMiner, self).__init__(*args, **kwargs)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(base_dir, "models", "sft_model")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.config.model_path, self.config.model_base, self.config.model_name, self.config.load_8bit, self.config.load_4bit, device=self.device)

        lora_path = os.path.join(base_dir, "models", "rlhf_lora_adapter_model")

        self.model = PeftModel.from_pretrained(
            self.model,
            lora_path,
        )

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
        # images_list = [bt.Tensor.deserialize(image) for image in synapse.images]
        message = synapse.messages[0]
        
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles


        message = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + message
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        image_args = {}
        images = None

        max_context_length = getattr(self.model.config, 'max_position_embeddings', 2048)
        num_image_tokens = 0

        if len(synapse.images) > 0:
            image_list = []
            image_transform = Compose([
                ToPILImage(),
                # Resize((224, 224))
            ])
            # to_pil_image = ToPILImage()
            image_list = [image_transform(bt.Tensor.deserialize(image)) for image in synapse.images]
            # bt.logging.info('image detected!!!!!!', image_list[0].shape)

            images = process_images(image_list, self.image_processor, self.model.config)

            if type(images) is list:
                images = [image.to(self.model.device, dtype=torch.float16) for image in images]
            else:
                images = images.to(self.model.device, dtype=torch.float16)


            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            bt.logging.debug('new message', prompt)

            num_image_tokens = prompt.count(replace_token) * self.model.get_vision_tower().num_patches

            image_args = {"images": images}

        

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
            try:
                max_new_tokens = self.config.llava.max_new_tokens
                if len(synapse.images) > 0:
                    input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
                else:
                    input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
                keywords = [None]
                # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

                max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
                do_sample = True

                bt.logging.info('images object', images)
                thread = Thread(target=self.model.generate, kwargs=dict(
                    inputs=input_ids,
                    images=images,
                    do_sample=do_sample,
                    temperature=self.config.llava.temperature,
                    top_p=self.config.llava.top_p,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    # stopping_criteria=[stopping_criteria],
                    use_cache=True
                ))
                thread.start()


                buffer = []
                output_text = ""
                for token in streamer:
                    output_text += token

                    
                    N = 1  # Number of tokens to send back to the client at a time
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
            except:
                traceback.print_exc()

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
    with LlavaMiner():
        while True:
            print("running...", time.time())
            time.sleep(1)

# def main():
#     print("Loading models...")

#     t0 = time.time()

#     model = Llava(
#         vision_model_path="models/eva_vit_g.pth",
#         llama_model="models/vicuna13b_v0/",
#         q_former_model="models/blip2_pretrained_flant5xxl.pth",
#     )

#     ckpt_path = "models/pretrained_llava.pth"

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

#         output_text = chat.answer_async(chat_state=chat_state,
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