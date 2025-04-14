import traceback
import bittensor as bt


def print_info(metagraph, hotkey, block, isMiner=True):
    uid = metagraph.hotkeys.index(hotkey)
    log = f"UID:{uid} | Block:{block} | Consensus:{metagraph.C[uid]} | "
    if isMiner:
        bt.logging.info(
            log
            + f"Stake:{metagraph.S[uid]} | Trust:{metagraph.T[uid]} | Incentive:{metagraph.I[uid]} | Emission:{metagraph.E[uid]}"
        )
        return
    bt.logging.info(log + f"VTrust:{metagraph.Tv[uid]} | ")


def fail_with_none(message: str = ""):
    def outer(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                bt.logging.error(message)
                bt.logging.error(str(e))
                bt.logging.error(traceback.format_exc())
                return None

        return inner

    return outer


class ExitContext:
    """
    Using this as a class lets us pass this to other threads
    """
    isExiting: bool = False

    def startExit(self, *_):
        if self.isExiting:
            exit()
        self.isExiting = True

    def __bool__(self):
        return self.isExiting
