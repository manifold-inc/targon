import pickle
import json


with open("cache.pickle", "rb") as file:
    loaded_data = pickle.load(file)

with open("cache.json", "w") as file:
    json.dump(loaded_data, file)

print("Done!")
