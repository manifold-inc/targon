# This file is mostly for testing, but can be used as reference for what commands should look like
default:
  @just list

validator:
  python3 neurons/validator.py --wallet.name validator --netuid 40 --subtensor.network test --neuron.model_endpoint http://localhost:8000/v1 --neuron.proxy.port 8080 --neuron.api_key aksjdalskj --neuron.epoch_length 101

miner num="0":
  python neurons/miner.py --wallet.name miner --netuid 40 --wallet.hotkey new-miner{{num}} --subtensor.network test --neuron.model_endpoint http://localhost:8001/v1 --nueron.api_key 1234 --axon.port 900{{num}}
