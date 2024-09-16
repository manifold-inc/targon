from httpx import Timeout
import traceback
import asyncio
import openai
from neurons.validator import Validator
from targon.protocol import Endpoints
from targon.utils import create_header_hook


MINER_UID = -1


async def main():
    try:
        validator = Validator(load_dataset=False)
        axon_info = validator.metagraph.axons[MINER_UID]
        miner = openai.AsyncOpenAI(
            base_url=f"http://{axon_info.ip}:{axon_info.port}/v1",
            api_key="sn4",
            max_retries=0,
            timeout=Timeout(12, connect=5, read=5),
            http_client=openai.DefaultAsyncHttpxClient(
                event_hooks={
                    "request": [
                        create_header_hook(validator.wallet.hotkey, axon_info.hotkey)
                    ]
                }
            ),
        )
        prompt = "What is the x y problem"
        res = await miner.completions.create(
            prompt=prompt,
            model="NousResearch/Meta-Llama-3.1-8B-Instruct",
            stream=True,
        )
        tokens = []
        async for chunk in res:
            if chunk.choices[0].text is None:
                continue
            choice = chunk.choices[0]
            if choice.model_extra is None:
                continue
            token_ids = choice.model_extra.get("token_ids") or []
            token_id = token_ids[0] if len(token_ids) > 0 else -1
            tokens.append(
                (
                    choice.text or "",
                    token_id,
                    choice.model_extra.get("powv") or -1,
                )
            )
            print(choice.text, token_id, choice.model_extra.get("powv") or -1)
        print(validator.check_tokens({"prompt": prompt}, tokens, Endpoints.COMPLETION))
        print(validator.check_tokens({"prompt": prompt}, tokens, Endpoints.COMPLETION))
        print(validator.check_tokens({"prompt": prompt}, tokens, Endpoints.COMPLETION))
    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
