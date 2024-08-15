from random import shuffle
import asyncio
from neurons.validator import Validator


MINER_UIDS = []

if __name__ == "__main__":
    validator = Validator()
    miner_uids = validator.get_miner_uids()
    shuffle(miner_uids)
    miner_uids = miner_uids[:48]
    res = asyncio.get_event_loop().run_until_complete(
        validator.query_miners(miner_uids)
    )

    if res is None:
        print("No response from miner")
        exit()
    stats, ground_truth, sampling_params, messages = res
    if ground_truth is None:
        ground_truth = ""
    with open("output.txt", "w") as res:
        print(sampling_params.model_dump())
        res.write(f"Sampling Params: {sampling_params.model_dump()}\n")
        res.write(f"Query: {messages}\n\n")
        for uid, stat in stats:
            wps = (
                min(len(stat.response.split(" ")), len(ground_truth.split(" ")))
                / stat.total_time
            )
            blob = f"UID: {uid:>3}  \t"
            # blob += f"Ground Truth: {ground_truth}\n\n"
            # blob += f"Miner response: {stat.response}\n"
            blob += f"WPS: {wps:>5.2f} "
            blob += f"Total Time: {stat.total_time:>5.2f} "
            blob += f"Jaro Score: {stat.jaros}"
            res.write(blob + "\n")
            print(blob)
