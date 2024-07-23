
def miner_print_info(metagraph, hotkey, step, block):
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


def vali_print_info(metagraph, hotkey, step, block):
    uid = metagraph.hotkeys.index(hotkey)

    log = (
        "Validator | "
        f"Step:{step} | "
        f"UID:{uid} | "
        f"Block:{block} | "
        f"Stake:{metagraph.S[uid]} | "
        f"VTrust:{metagraph.Tv[uid]} | "
        f"Dividend:{metagraph.D[uid]} | "
        f"Emission:{metagraph.E[uid]}"
    )

    return log
