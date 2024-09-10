import asyncio
from neurons.validator import Validator
from targon.protocol import Endpoints


MINER_UIDS = []

if __name__ == "__main__":
    validator = Validator()
    miner_uids = []
    res = asyncio.get_event_loop().run_until_complete(
        validator.query_miners(miner_uids, Endpoints.CHAT)
    )

    if res is None:
        print("No response from miner")
        exit()
    stats, sampling_params, messages, endpoint = res
    print(sampling_params.model_dump())
    for uid, stat in stats:
        if not sampling_params.max_tokens:
            continue
        tps = (
            min(len(stat.tokens), sampling_params.max_tokens)
            / stat.total_time
        )
        blob = f"UID: {uid:>3}  \t"
        # blob += f"Ground Truth: {ground_truth}\n\n"
        # blob += f"Miner response: {stat.response}\n"
        blob += f"WPS: {tps:>5.2f} "
        blob += f"Total Time: {stat.total_time:>5.2f} "
        blob += f"Verified: {stat.verified}"
        print(blob)
