from httpx import Timeout
import openai
from neurons.validator import Validator
from targon.utils import create_header_hook


MINER_UID= -1

if __name__ == "__main__":
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
    res = miner.completions.create(
        prompt="What is the x y problem",
        model="NousResearch/Meta-Llama-3.1-8B-Instruct",
        stream=True,
    )
    for chunk in res:
        print(chunk)
