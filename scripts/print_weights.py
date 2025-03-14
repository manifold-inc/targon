import asyncio
import json
import numpy as np

from bittensor.utils.weight_utils import process_weights_for_netuid
from neurons.validator import Validator
from targon.jugo import get_global_stats
from targon.math import get_weights


async def main():
    vali = Validator(standalone=True)
    organic_metadata = await get_global_stats(vali.wallet)
    if organic_metadata is None:
        print("Cannot get weights, failed getting metadata from jugo")
        return
    miner_models = {x: {} for x in range(256)}
    uids, raw_weights, _ = get_weights(miner_models, vali.organics, organic_metadata)
    (
        processed_weight_uids,
        processed_weights,
    ) = process_weights_for_netuid(
        uids=np.asarray(uids),
        weights=np.asarray(raw_weights),
        netuid=4,
        subtensor=vali.subtensor,
        metagraph=vali.metagraph,
    )
    processed_weights = [float(x) for x in processed_weights]
    processed_weight_uids = [int(x) for x in processed_weight_uids]
    final = {}
    for uid, w in zip(processed_weight_uids, processed_weights):
        final[uid] = w
    print("Final Weights: " + json.dumps(final, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
