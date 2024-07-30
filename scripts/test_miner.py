from neurons.validator import Validator


MINER_UID = 0

if __name__ == "__main__":
    validator = Validator()
    validator.sync_metagraph()
    validator.resync_hotkeys()
    validator.run()
    stats = validator.query_miners([0])
    print(stats)
