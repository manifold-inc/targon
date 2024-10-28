import json
from neurons.validator import Validator
from targon.dataset import download_dataset
from targon.types import Endpoints

miners = []
model_name = "NTQAI/Nxcode-CQ-7B-orpo"
endpoint = Endpoints.CHAT

if __name__ == "__main__":
    validator = Validator(run_init=False)
    validator.dataset = download_dataset(True)
    res = validator.loop.run_until_complete(
        validator.query_miners(miners, model_name, endpoint)
    )
    with open("results.json", "w") as file:
        json.dump(
            res,
            file,
        )
        file.flush()
    exit()
