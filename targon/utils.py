
def print_info(metagraph, hotkey, step, block):
    uid = metagraph.hotkeys.index(hotkey)

    log = (
        "Miner | "
        f"Step:{step} | "
        f"UID:{uid} | "
        f"Block:{block} | "
        f"Stake:{metagraph.S[uid]} | "
        f"Trust:{metagraph.T[uid]} | "
        f"Incentive:{metagraph.I[uid]} | "
        f"Emission:{metagraph.E[uid]}"
    )

    return log
