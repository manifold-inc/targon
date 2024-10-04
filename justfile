set dotenv-load
# This file is mostly for testing, but can be used as reference for what commands should look like

default:
  @just --list

validator:
  python3 neurons/validator.py --wallet.name validator --netuid 40 --subtensor.network test --epoch-length 101 --logging.trace --autoupdate-off --mock --models.mode endpoint --models.endpoint https://targon.sybil.com/api/models

miner num="0":
  python neurons/miner.py --wallet.name miner --netuid 40 --wallet.hotkey new-miner{{num}} --subtensor.network test --model-endpoint http://localhost:9000/v1 --axon.port 700{{num}} --api_key abc123 --mock

script script_name opts="":
  python3 scripts/{{script_name}}.py --wallet.name validator --netuid 40 --subtensor.network test --neuron.port 8080 --epoch-length 101 --logging.trace {{opts}}

up:
  docker compose -f docker-compose.testnet.yml build
  docker compose -f docker-compose.testnet.yml up -d

build_verifier:
  cd verifier && docker build -t manifoldlabs/sn4-verifier .
  docker push manifoldlabs/sn4-verifier
