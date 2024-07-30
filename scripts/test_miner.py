from neurons.validator import Validator


MINER_UID = 0

if __name__ == "__main__":
    validator = Validator()
    validator.sync_metagraph()
    validator.resync_hotkeys()
    res = validator.query_miners([0])

    if res is None:
        print("No response from miner")
        exit()
    stats, ground_truth, sampling_params = res
    for uid, stat in stats:
        print(uid)
        print(f"Ground Truth: {ground_truth}")
        print(f"Miner response: {stat.response}")
        print(f"Total Time: {stat.total_time}")
        print(f"Jaro Score: {stat.jaro_score}")
        print()
