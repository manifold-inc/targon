set dotenv-load

# This file is mostly for testing, but can be used as reference for what commands should look like
default:
  @just list

validator:
  python3 neurons/validator.py --wallet.name validator --netuid 40 --subtensor.network test --neuron.model_endpoint http://localhost:9000/v1 --neuron.port 8080 --neuron.epoch_length 101 --logging.trace --database.url $DATABASE_URL

miner num="0":
  python neurons/miner.py --wallet.name miner --netuid 40 --wallet.hotkey new-miner{{num}} --subtensor.network test --neuron.model_endpoint http://localhost:9001/v1 --axon.port 900{{num}}
