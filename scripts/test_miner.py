from typing import List
from httpx import Timeout
import traceback
import httpx
import openai
from openai.types.chat import ChatCompletionMessageParam
from neurons.validator import Validator
from targon.epistula import generate_header
from targon.protocol import Endpoints


MINER_UID = -1

messages: List[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": f"What is the deffinition of the x y problem ",
    },
]
model = "NousResearch/Meta-Llama-3.1-8B-Instruct"
prompt = "def print_hello_world():"


def create_header_hook(hotkey, axon_hotkey):
    def add_headers(request: httpx.Request):
        for key, header in generate_header(hotkey, request.read(), axon_hotkey).items():
            request.headers[key] = header

    return add_headers


def main():
    try:
        validator = Validator(load_dataset=False)
        axon_info = validator.metagraph.axons[MINER_UID]
        miner = openai.OpenAI(
            base_url=f"http://{axon_info.ip}:{axon_info.port}/v1",
            api_key="sn4",
            max_retries=0,
            timeout=Timeout(12, connect=5, read=5),
            http_client=openai.DefaultHttpxClient(
                event_hooks={
                    "request": [
                        create_header_hook(validator.wallet.hotkey, axon_info.hotkey)
                    ]
                }
            ),
        )
        res = miner.chat.completions.create(
            messages=messages, model=model, stream=True, logprobs=True, max_tokens=200
        )
        tokens = []
        for chunk in res:
            if chunk.choices[0].delta.content is None:
                continue
            choice = chunk.choices[0]
            if choice.model_extra is None:
                continue
            token_ids = choice.model_extra.get("token_ids") or []
            token_id = token_ids[0] if len(token_ids) > 0 else -1
            tokens.append(
                (
                    choice.delta.content or "",
                    token_id,
                )
            )
            print(choice.delta.content, token_id)
        print(
            validator.check_tokens({"messages": messages[:20]}, tokens, Endpoints.CHAT)
        )
        print(
            validator.check_tokens({"messages": messages[:20]}, tokens, Endpoints.CHAT)
        )
        print(
            validator.check_tokens({"messages": messages[:20]}, tokens, Endpoints.CHAT)
        )
        print(
            validator.check_tokens({"messages": messages[:20]}, tokens, Endpoints.CHAT)
        )
        print(validator.check_tokens({"messages": messages}, tokens, Endpoints.CHAT))
        print(validator.check_tokens({"messages": messages}, tokens, Endpoints.CHAT))
        print(validator.check_tokens({"messages": messages}, tokens, Endpoints.CHAT))
        print(validator.check_tokens({"messages": messages}, tokens, Endpoints.CHAT))
        print(validator.check_tokens({"messages": messages}, tokens, Endpoints.CHAT))
    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
