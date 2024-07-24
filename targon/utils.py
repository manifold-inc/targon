def print_info(metagraph, hotkey, block, isMiner=True):
    uid = metagraph.hotkeys.index(hotkey)
    log = f"UID:{uid} | Block:{block} | Consensus:{metagraph.C[uid]} | "
    if isMiner:
        return (
            log
            + f"Stake:{metagraph.S[uid]} | Trust:{metagraph.T[uid]} | Dividend:{metagraph.D[uid]} | Emission:{metagraph.E[uid]}"
        )
    return log + f"VTrust:{metagraph.Tv[uid]} | "
