import os
import random
def get_inital_links():
    pre_links = []
    if not os.path.exists(os.path.expanduser("~/.bittensor/links.csv")):
        os.makedirs(os.path.expanduser("~/.bittensor/"), exist_ok=True)
        os.system("wget https://tranco-list.eu/download/JXYJY/full -O ~/.bittensor/links.txt")

    with open(os.path.expanduser("~/.bittensor/links.txt"), "r") as f:
        for line in f.readlines():
            pre_links.append(line.split(",")[1].strip())

    # shuffle links
    links = random.shuffle(pre_links)

    return links
