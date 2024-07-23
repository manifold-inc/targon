
def print_info(metagraph, hotkey, step, block, isMiner=True):
    uid = metagraph.hotkeys.index(hotkey)

    log = (
        f"Step:{step} | "
        f"UID:{uid} | "
        f"Block:{block} | "
        f"Stake:{metagraph.S[uid]} | "
        f"VTrust:{metagraph.Tv[uid] if not isMiner else metagraph.T[uid]} | "
        f"Dividend:{metagraph.D[uid]} | "
        f"Emission:{metagraph.E[uid]}"
    )

    return log
