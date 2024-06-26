# TARGON: A Redundant Deterministic Verification of Large Language Models

TARGON (Bittensor Subnet 4) is a redundant deterministic verification mechanism that can be used to interpret and analyze ground truth sources and a query. 


NOTICE: Using this software, you must agree to the Terms and Agreements provided in the terms and conditions document. By downloading and running this software, you implicitly agree to these terms and conditions.

> v1.9.9 - removed tier system, implemented exponential tok/s reward scaling

> v1.0.6 - Runpod is now supported 🎉. Check out the runpod docs in docs/runpod/verifier.md and docs/runpod/prover.md for more information.

> ~~v1.0.0 - Runpod is not currently supported on this version of TARGON for verifiers. An easy alternative can be found in the [Running on TensorDock](#running-on-tensordock) section.~~


Currently supporting python>=3.9,<3.11.

> Note: The storage subnet is in an alpha stage and is subject to rapid development.


# Table of Contents
1. [Compute Requirements](#compute-requirements)
1. [Roadmap](#roadmap)
1. [Installation](#installation)
    - [Install Docker](#install-docker)
    - [Install PM2](#install-pm2)
    - [Install TARGON](#install-targon)
1. [What is a Redundant Deterministic Verification Network?](#what-is-a-redundant-deterministic-verification-network)
   - [Role of a Prover](#role-of-a-prover)
   - [Role of a Verifier](#role-of-a-verifier)
1. [Features of TARGON](#features-of-targon)
    - [Challenge Request](#challenge-request)
    - [Inference Request ](#inference-request)
1. [How to Run TARGON](#how-to-run-targon)
    - [Run a Prover](#run-a-prover)
    - [Run a Verifier](#run-a-verifier)
1. [How to Contribute](#how-to-contribute)


# Compute Requirements
The following table shows the VRAM, Storage, RAM, and CPU minimum requirements for running a verifier or prover.

GPU - A100
| Provider   | VRAM   | Storage |   RAM   | CPU  |
|------------|--------|---------|---------|------|
| TensorDock |  80GB  | 200GB   |   16GB  | 4    |
| Latitude   |  80GB  | 200GB   |   16GB  | 4    |
| Paperspace |  80GB  | 200GB   |   16GB  | 4    |
| GCP        |  80GB  | 200GB   |   16GB  | 4    |
| Azure      |  80GB  | 200GB   |   16GB  | 4    |
| AWS        |  80GB  | 200GB   |   16GB  | 4    |
| Runpod     |  80GB  | 200GB   |   16GB  | 4    |


# Recommended Compute Providers
The following table shows the suggested compute providers for running a verifier or prover.

| Provider   | Cost  | Location |   Machine Type   | Rating |
|------------|-------|----------|--------------|--------|
| TensorDock | Low   | Global   | VM & Container   | 4/5    |
| Latitude   | Medium| Global   | Bare Metal       | 5/5    |
| Paperspace | High  | Global   | VM & Bare Metal  | 4.5/5  |
| GCP        | High  | Global   | VM & Bare Metal  | 3/5    |
| Azure      | High  | Global   | VM & Bare Metal  | 3/5    |
| AWS        | High  | Global   | VM & Bare Metal  | 3/5    |
| Runpod     | Low   | Global   | VM & Container   | 5/5    |

# Roadmap

<details open>
<summary>Completed</summary>

- [x] Challenge Request
- [x] Reward System
- [x] Bonding
- [x] Database
- [x] Auto Update
- [x] Forwarding
- [x] Tiered Requests to match throughput
- [x] Inference Request

</details>

<details>
<summary>In Progress</summary>

- [] Metapool
- [] flashbots' style block space for verifier bandwidth
- [] metrics dashboard
</details>

# Installation

## Overview
In order to run TARGON, you need to install Docker, PM2, and TARGON package. The following instructions apply only to Ubuntu OSes. For your specific OS, please refer to the official documentation.

<details>
<summary>Install Docker</summary>

Install docker on your machine. Follow the instructions [here](https://docs.docker.com/engine/install/). The following instructions apply only to Ubuntu OSes.

### Set up Docker's apt repository.
Before you install Docker Engine for the first time on a new host machine, you need to set up the Docker repository. Afterward, you can install and update Docker from the repository.x
```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

### Install Docker Engine
To install the latest version, run:
```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Post Installation Steps
To run docker commands without sudo, create a docker group and add your user.
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
exit
```
Log back in and run the following command to verify that you can run docker commands without sudo.
```bash
docker ps
```
You have now installed Docker.
</details>

<details>

<summary>Install PM2</summary>

Install PM2 on your machine.

### Download NVM
To install or update nvm, you should run the install script. To do that, you may either download and run the script manually, or use the following cURL or Wget command:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```
### Add NVM to bash profile
Running either of the above commands downloads a script and runs it. The script clones the nvm repository to ~/.nvm, and attempts to add the source lines from the snippet below to the correct profile file (~/.bash_profile, ~/.zshrc, ~/.profile, or ~/.bashrc).
```bash
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
```
### Install Node
```bash
nvm install node
```

### Install PM2
```bash
npm install pm2@latest -g
```
You have now installed PM2.

</details>

### Install TARGON

### Clone the repository
```bash
git clone https://github.com/manifold-inc/targon.git
cd targon
```

### Install dependencies
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

You have now installed TARGON. You can now run a prover or verifier.


# What is a Redundant Deterministic Verification Network?
Using existing datasets that are public poses certain challenges to rewarding models for work. Those who seek to win overfit their models on these inputs, or can front run the input by using a lookup for the output. A solution to this challenge can be done using prompt generation using a  query generation model, and a private input. 

The private input will be sourced from an api ran by Manifold, and this input is rotated every twelve seconds, and is authenticated with a signature using the validator’s keys. The private input is fed into the query generation model, which can be run by the validator or as a light client by manifold. The data source can be either from a crawl or from RedPajama.

The query, private input, and a deterministic seed are used to generate a ground truth output with the specified model, which can be run by the validator or as a light client. The validator then sends requests to miners with the query, private input, and deterministic seed. The miner output are compared to the ground truth output. If the tokens are equal, the miner has successfully completed the challenge.

## Role of a Prover
A prover is a node that is responsible for generating a output from a query, private input, and a deterministic sampling params. 

## Role of a Verifier
A verifier is a node that is responsible for verifying a prover's output. The verifier will send a request to a prover with a query, private input, and deterministic sampling params. The miner will then send back a response with the output. The verifier will then compare the output to the ground truth output. If the outputs are equal, then the prover has completed the challenge.

# Features of TARGON

## Challenge Request
A challenge request is a request sent by a verifier to a prover. The challenge request contains a query, private input, and deterministic sampling params. The prover will then generate an output from the query, private input, and deterministic sampling params. The prover will then send the output back to the verifier.

## Inference Request
An inference request is a request sent by a verifier to a prover. The inference request contains a query, private input, and inference sampling params. The prover will then generate an output from the query, private input, and deterministic sampling params. The prover will then stream the output back to the verifier.
> *CAVEAT:* Every Interval (360 blocks) there will be a random amount of inference samples by the verifier. The verifier will then compare the outputs to the ground truth outputs. The cosine similarity of the outputs will be used to determine the reward for the prover. Failing to do an inference request will result in a 5x penalty.


# How to Run TARGON

## Run a Prover
To get started running a prover, you will need to run the docker containers for the requirements of the prover. To do this, start with a template by runig the following command:
```bash
cp neurons/prover/docker-compose.example.yml neurons/prover/docker-compose.yml
```

This by default includes the following containers:
- TGI Inference Node
- Prover (optional)
- ~~Subtensor (experimental)~~

**experimental** Experimentally, you can uncomment the subtensor service in the template. The subtensor could also be set up locally on the host machine, external from docker. Otherwise, for running an external subtensor instance, whether locally on the machine or remote, you will want to make sure the prover starts up with the flag `--subtensor.chain_endpoint ws://the.subtensor.ip.addr:9944` to connect to the chain.

### GPU Sharding
You can optionally shard the model across multiple GPUs. To do this, you will need to modify the docker template you copied above and include these flags at the end of the command within the service.

NOTE: Scroll horizontally to see the full command if this readme is truncated by the viewport.
```docker
version: '3.8'
services:
  text-generation-service:
    image: ghcr.io/huggingface/text-generation-inference:1.3
    command: --model-id mlabonne/NeuralDaredevil-7B --max-input-length 3072 --max-total-tokens 4096 --sharded --num-shard 2
    volumes:
      - ./models:/data
    ports:
      - "127.0.0.1:8080:80"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ["0","1"]
    shm_size: '1g'
```
NOTE: Sharding is not set by default in the template, so you will need to modify it accordingly prior to starting the container.


### Docker

<details>
<summary>Run prover with Docker</summary>

By default, the docker template you copied above for TGI contains the prover. This is optional.

**optionally**, you can edit the docker-compose.yml file to comment out the proving container, leaving the TGI service to run alone, but then you will need to run the prover with PM2.

If you will use the prover in the docker, here's an example of the prover service section of the docker template, which is also what exposes the axon port.

```docker
  prover:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8091:8091"
    command: ./entrypoint.sh
    volumes:
      - ~/.bittensor/wallets:/root/.bittensor/wallets
```
NOTE: This prover should already exist in your template file.



In the entrypoint.sh, please replace the wallet name and hotkey with your own. If you prefer, you can also change the subtensor chain endpoint to your own chain endpoint.

```bash
python3 app.py --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOTKEY --logging.trace --subtensor.chain_endpoint ws://0.0.0.0:9944
```
NOTE: Trace logging is very verbose. You can use `--logging.debug` instead for less log bloat.

</details>


### PM2

<details>
<summary> Run prover with PM2</summary>

If you will go this route, remember you will need to have the TGI instance running (docker covers that above), and be sure to comment out the prover service in the docker template

Please replace the wallet name and hotkey with your own. If you prefer, you can also change the subtensor chain endpoint to your own chain endpoint.

```bash
cd neurons/prover
pm2 start app.py --name prover --interpreter python3 -- --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOTKEY --logging.trace --subtensor.chain_endpoint 0.0.0.0:9944
```
NOTE: Trace logging is very verbose. You can use `--logging.debug` instead for less log bloat.

</details>


### Options
The add_prover_args function in the targon/utils/config.py file is used to add command-line arguments specific to the prover. Here are the options it provides:

1. --neuron.name: This is a string argument that specifies the name of the neuron. The default value is 'prover'.

2. --blacklist.force_verifier_permit: This is a boolean argument that, if set, forces incoming requests to have a permit. The default value is False.

3. --blacklist.allow_non_registered: This is a boolean argument that, if set, allows provers to accept queries from non-registered entities. This is considered dangerous and its default value is False.

4. --neuron.tgi_endpoint: This is a string argument that specifies the endpoint to use for the TGI client. The default value is "http://0.0.0.0:8080".



## Run a Verifier
To get started running a verifier, you will need to be running the docker containers for the requirements of the verifier. To do this, run the following command:
```bash
cp neurons/verifier/docker-compose.example.yml neurons/verifier/docker-compose.yml
./scripts/generate_redis_password.sh
nano neurons/verifier/docker-compose.yml # replace YOUR_PASSWORD_HERE with the password generated by the script
docker compose -f neurons/verifier/docker-compose.yml up -d
```

this includes the following containers:
- Redis
- TGI Inference Node
- Subtensor
- Verifier (optional)

> **IMPORTANT:** You will need to edit the docker-compose.yml file with your new password. You can do this by:
```bash
./scripts/generate_redis_password.sh
```
this will output a secure password for you to use. You will then need to edit the docker-compose.yml file and replace the password with your new password.

first, edit the docker-compose.yml file:
```bash
nano neurons/verifier/docker-compose.yml
```
then replace the password with your new password:

```docker
  redis:
    image: redis:latest
    command: redis-server --requirepass YOUR_PASSWORD_HERE
    ports:
      - "6379:6379"
```

**optionally**, you can edit the docker-compose.yml file to include the verifier container, but you will need to edit the docker-compose.yml file and uncomment the verifier container. Otherwise you can run the verifier with PM2, which is experimental.


### Docker

<details>
<summary>Run with Docker</summary>

```docker
  verifier:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8091:8091"
    command: ./entrypoint.sh
    volumes:
      - ~/.bittensor/wallets:/root/.bittensor/wallets

```
and then edit the entrypoint.sh file to include the args specific for your prover.

```bash
python3 app.py --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOTKEY --logging.debug --logging.trace --subtensor.chain_endpoint 0.0.0.0:9944 --database.password YOUR_PASSWORD_HERE
```

replace the wallet name, wallet hotkey, and db pass with your wallet name, wallet hotkey and pass. You can also change the subtensor chain endpoint to your own chain endpoint if you perfer.

</details>

### PM2

<details>
<summary>Run with PM2</summary

Run the following command to start the verifier with PM2:

```bash
cd neurons/verifier

pm2 start app.py --name verifier --interpreter python3 -- --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOTKEY --logging.debug --logging.trace --subtensor.chain_endpoint
ws://0.0.0.0:9944 --database.password YOUR_PASSWORD_HERE
```

### Options
The add_verifier_args function in the targon/utils/config.py file is used to add command-line arguments specific to the verifier. Here are the options it provides:

1. --neuron.sample_size: The number of provers to query in a single step. Default is 10.

2. --neuron.disable_set_weights: A flag that disables setting weights. Default is False.

3. --neuron.moving_average_alpha: Moving average alpha parameter, how much to add of the new observation. Default is 0.05.

4. --neuron.axon_off or --axon_off: A flag to not attempt to serve an Axon. Default is False.

5. --neuron.vpermit_tao_limit: The maximum number of TAO allowed to query a verifier with a vpermit. Default is 4096.

6. --neuron.tgi_endpoint: The endpoint to use for the TGI client. Default is "http://0.0.0.0:8080".

7. --database.host: The path to write debug logs to. Default is "127.0.0.1".

8. --database.port: The path to write debug logs to. Default is 6379.

9. --database.index: The path to write debug logs to. Default is 1.

10. --database.password: The password to use for the redis database. Default is None.

11. --neuron.compute_stats_interval: The interval at which to compute statistics. Default is 360.

These options can be used to customize the behavior of the verifier when it is run.

Refer to the code here:


replace the wallet name and wallet hotkey with your wallet name and wallet hotkey. You can also change the subtensor chain endpoint to your own chain endpoint if you perfer.

</details>


# How to Contribute

## Code Review
Project maintainers reserve the right to weigh the opinions of peer reviewers using common sense judgement and may also weigh based on merit. Reviewers that have demonstrated a deeper commitment and understanding of the project over time or who have clear domain expertise may naturally have more weight, as one would expect in all walks of life.

Where a patch set affects consensus-critical code, the bar will be much higher in terms of discussion and peer review requirements, keeping in mind that mistakes could be very costly to the wider community. This includes refactoring of consensus-critical code.

Where a patch set proposes to change the TARGON subnet, it must have been discussed extensively on the discord server and other channels, be accompanied by a widely discussed BIP and have a generally widely perceived technical consensus of being a worthwhile change based on the judgement of the maintainers.

## Finding Reviewers
As most reviewers are themselves developers with their own projects, the review process can be quite lengthy, and some amount of patience is required. If you find that you've been waiting for a pull request to be given attention for several months, there may be a number of reasons for this, some of which you can do something about:
