from neurons.validator import Validator


MINER_UIDS = []

if __name__ == "__main__":
    validator = Validator()
    validator.sync_metagraph()
    validator.resync_hotkeys()
    res = validator.query_miners(MINER_UIDS)

    if res is None:
        print("No response from miner")
        exit()
    stats, ground_truth, sampling_params = res
    with open("responses.txt", "w") as res:
        for uid, stat in stats:
            blob = f"{uid}\n"
            blob += f"Ground Truth: {ground_truth}\n"
            blob += f"Miner response: {stat.response}\n"
            blob += f"Total Time: {stat.total_time}\n"
            blob +=f"Jaro Score: {stat.jaro_score}\n"
            res.write(blob)
            print(blob)
