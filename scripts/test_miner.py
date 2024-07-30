from neurons.validator import Validator


MINER_UIDS = []

if __name__ == "__main__":
    validator = Validator()
    res = validator.query_miners(MINER_UIDS)

    if res is None:
        print("No response from miner")
        exit()
    stats, ground_truth, sampling_params = res
    with open("output.txt", "w") as res:
        print(sampling_params.model_dump())
        res.write(f"{sampling_params.model_dump()}\n\n")
        for uid, stat in stats:
            blob = f"UID: {uid}\n"
            blob += f"Ground Truth: {ground_truth}\n"
            blob += f"Miner response: {stat.response}\n"
            blob += f"Total Time: {stat.total_time}\n"
            blob += f"Jaro Score: {stat.jaro_score}\n\n"
            res.write(blob)
            print(blob)
