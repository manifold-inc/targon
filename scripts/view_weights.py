from neurons.validator import Validator
import plotext as plt

MINER_UIDS = []

if __name__ == "__main__":
    validator = Validator()
    uids, weights = validator.get_weights()
    weights = sorted(weights)
    plt.scatter(weights)
    plt.title("Weights")  # to apply a title
    plt.show()
