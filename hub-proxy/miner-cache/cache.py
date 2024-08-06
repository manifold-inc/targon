import asyncio
from typing import List, Tuple
import bittensor as bt
from redis import Redis
from redis.commands.json.path import Path
import numpy


async def sync_miners():
    metagraph = subtensor.metagraph(netuid=4)
    indices = numpy.argsort(metagraph.incentive)[-10:]
    print(metagraph.incentive)
    print(indices)

    # Get the corresponding uids
    uids_with_highest_incentives: List[int] = metagraph.uids[indices].tolist()

    # get the axon of the uids
    axons: List[Tuple[bt.AxonInfo, int]] = [
        (metagraph.axons[uid], uid) for uid in uids_with_highest_incentives
    ]
    ips = [
        {
            "ip": axon.ip,
            "port": axon.port,
            "hotkey": axon.hotkey,
            "coldkey": axon.coldkey,
        }
        for (axon, _) in axons
    ]
    print(ips, flush=True)
    r.json().set("miners", obj=ips, path=Path.root_path())
    await asyncio.sleep(60 * 12)


if __name__ == "__main__":
    subtensor = bt.subtensor("ws://subtensor.sybil.com:9944")
    r = Redis(host="cache", port=6379, decode_responses=True)
    while True:
        asyncio.run(sync_miners())
