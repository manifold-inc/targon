from neurons.validator import Validator


MINER_UIDS = []

if __name__ == "__main__":
    validator = Validator()
    res = validator.query_miners(MINER_UIDS)

    if res is None:
        print("No response from miner")
        exit()
    stats, ground_truth, sampling_params, messages = res
    if ground_truth is None:
        ground_truth = ""
    with open("output.txt", "w") as res:
        print(sampling_params.model_dump())
        res.write(f"Sampling Params: {sampling_params.model_dump()}\n")
        res.write(f"Query: {messages}\n\n")
        for uid, stat in stats:
            wps = min(len(stat.response.split(" ")), len(ground_truth.split(" "))) / stat.total_time
            blob = f"UID: {uid}\n"
            blob += f"Ground Truth: {ground_truth}\n\n"
            blob += f"Miner response: {stat.response}\n"
            blob += f"WPS: {wps}\n"
            blob += f"Total Time: {stat.total_time}\n"
            blob += f"Jaro Score: {stat.jaro_score}\n\n"
            res.write(blob)
            print(blob)
