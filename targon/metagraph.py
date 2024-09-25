from typing import Callable, Dict, List, Tuple
import numpy as np
import bittensor as bt
from bittensor.utils.weight_utils import process_weights_for_netuid

from targon.utils import fail_with_none

import threading


def get_miner_uids(
    metagraph: "bt.metagraph", self_uid: int, vpermit_tao_limit: int
) -> List[int]:
    available_uids = []
    for uid in range(int(metagraph.n.item())):
        if uid == self_uid:
            continue

        # Filter non serving axons.
        if not metagraph.axons[uid].is_serving:
            continue
        # Filter validator permit > 1024 stake.
        if metagraph.validator_permit[uid]:
            if metagraph.S[uid] > vpermit_tao_limit:
                continue
        available_uids.append(uid)
        continue
    return available_uids


@fail_with_none("Failed resyncing hotkeys")
def resync_hotkeys(metagraph: "bt.metagraph", miner_tps: Dict):
    bt.logging.info(
        "re-syncing hotkeys"
    )
    # Zero out all hotkeys that have been replaced.
    for uid, hotkey in enumerate(metagraph.hotkeys):
        if miner_tps.get(uid) is None:
            miner_tps[uid] = {}
        if hotkey != metagraph.hotkeys[uid]:
            miner_tps[uid] = {}


def create_set_weights(version: int, netuid):
    @fail_with_none("Failed setting weights")
    def set_weights(
        wallet: "bt.wallet",
        metagraph: "bt.metagraph",
        subtensor: "bt.subtensor",
        weights: Tuple[List[int], List[float]],
    ):
        if weights is None:
            return None
        uids, raw_weights = weights
        if not len(uids):
            bt.logging.info("No UIDS to score")
            return

        # Set the weights on chain via our subtensor connection.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=np.asarray(uids),
            weights=np.asarray(raw_weights),
            netuid=netuid,
            subtensor=subtensor,
            metagraph=metagraph,
        )

        bt.logging.info("Setting Weights: " + str(processed_weights))
        bt.logging.info("Weight Uids: " + str(processed_weight_uids))
        result, message = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=processed_weight_uids,  # type: ignore
            weights=processed_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=version,
            max_retries=1,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error(f"set_weights failed {message}")

    return set_weights


def create_subscription_handler(substrate, callback: Callable):
    def inner(obj, update_nr, _):
        substrate.get_block(block_number=obj["header"]["number"])

        if update_nr >= 1:
            return callback(obj["header"]["number"])  # type: int

    return inner


def start_subscription(substrate, callback: Callable):
    while True:
        return substrate.subscribe_block_headers(
            create_subscription_handler(substrate, callback)
        )


def run_block_callback_thread(substrate, callback: Callable):
    subscription_thread = threading.Thread(
        target=start_subscription, args=[substrate, callback], daemon=True
    )
    subscription_thread.start()
    bt.logging.info("Block subscription started in background thread.")
