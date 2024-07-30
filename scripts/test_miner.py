from neurons.validator import Validator


MINER_UIDS = []

if __name__ == "__main__":
    validator = Validator()
    res = validator.query_miners(MINER_UIDS)

    if res is None:
        print("No response from miner")
        exit()
    stats, ground_truth, sampling_params, messages = res
    with open("output.txt", "w") as res:
        print(sampling_params.model_dump())
        res.write(f"Sampling Params: {sampling_params.model_dump()}\n")
        res.write(f"Query: {messages}\n\n")
        for uid, stat in stats:
            blob = f"UID: {uid}\n"
            blob += f"\nGround Truth: {ground_truth}\n\n"
            blob += f"Miner response: {stat.response}\n"
            blob += f"Total Time: {stat.total_time}\n"
            blob += f"Jaro Score: {stat.jaro_score}\n\n"
            res.write(blob)
            print(blob)
