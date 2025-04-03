# Targon: A Deterministic Verification of Large Language Models

Targon (Bittensor Subnet 4) is a deterministic verification mechanism that is
used to incentivize miners to run openai compliant endpoints and serve synthetic
and organic queries.

NOTICE: Using this software, you must agree to the Terms and Agreements provided
in the terms and conditions document. By downloading and running this software,
you implicitly agree to these terms and conditions.

# Targon v6

Targon v6 and v5 run in parallel. 70% of rewards go to v6 (once miners start
passing verification) and 30% to v5.

## Running v6

v6 requires a Confidential Compute compatible node. We reccomend eith h100s or
h200s. To install v6, simple run `./tvm/install` and pass the following flags

@ahmed TODO

This will walk you though the install and download process, and boot the CVM.
After that, update the v5 miner.py file to report all the ip addresses of each
CVM you have running. That should be all you need to do.

# Table of Contents (v5)

1. [Compute Requirements](#recommended-compute-requirements)
1. [Installation](#installation)
   - [Install PM2](#install-pm2)
   - [Install Targon](#install-targon-on-your-machine)
1. [How to Run Targon](#how-to-run-targon)
   - [Running VLLM](#vllm)
   - [Running a Miner](#running-a-miner)
   - [Running a Validator](#running-a-validator)
1. [What is a Deterministic Verification Network?](#what-is-a-deterministic-verification-network)
   - [Role of a Miner](#role-of-a-miner)
   - [Role of a Validator](#role-of-a-validator)
1. [Features of Targon](#features-of-targon)
   - [Full OpenAI Compliance](#full-openai-compliance)
   - [Targon-Hub](#targon-hub)
1. [How to Contribute](#how-to-contribute)

# Recommended Compute Requirements

For validators we recommend a 8xA100, although a 1xA100 could also be used. We
plan on focusing on bringing these costs down in the coming updates.

For miners, A100 or H100s are common choices. Benchmarking is up to the miner to
determine what GPU works best for their optimizations.

#### Minimum Viable Compute Recommendations

- **VRAM:** 80 GB
- **Storage:** 200 GB
- **RAM:** 16 GB
- **CPU**: 4 core

# Installation

## Overview

In order to run Targon, you will need to install PM2 and the Targon package. The
following instructions apply only to Ubuntu OSes. For your specific OS, please
refer to the official documentation.

### Install PM2 on your machine

#### Download NVM

To install or update nvm, you should run the install script. To do that, you may
either download and run the script manually, or use the following cURL or Wget
command:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

#### Add NVM to bash profile

Running either of the above commands downloads a script and runs it. The script
clones the nvm repository to ~/.nvm, and attempts to add the source lines from
the snippet below to the correct profile file (~/.bash_profile, ~/.zshrc,
~/.profile, or ~/.bashrc).

```bash
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
```

#### Install Node

```bash
nvm install node
```

#### Install PM2

```bash
npm install pm2@latest -g
```

You have now installed PM2.

### Install Targon on your machine

#### Clone the repository

```bash
git clone https://github.com/manifold-inc/targon.git
cd targon
```

#### Install dependencies

```bash
python3 -m pip install -e .
```

You have now installed Targon. You can now run a validator or a miner.

# How to Run Targon

## Running a Miner

Before starting or registering your miner in Targon, first you will want to run
VLLM serving different images validators are requesting. You can find a list at
https://stats.sybil.com/stats/validator under the live tab. The more models you
run, the higher your incentive.

You will also need to include
`--enable-auto-tool-choice --tool-call-parser llama3_json` and
`--enable-auto-tool-choice --tool-call-parser hermes` for your Llama and Hermes
vLLM instances to support tool calling.

Once you have one (or multiple) models running, modify the default miner code to
proxy to the proper VLLM instance on each request. Verifiers will include the
`X-Targon-Model` header so that the miner node does not need to parse the actual
body.

In the `miner.py` script you will find two functions called `list_models` and
`list_nodes`. To serve multiple models you must:

1. Fill this out to respond to validators with any model you currently have
   available along with an estimate on how many queries per second you can
   handle (below is an example):

```py
async def list_models(self):
    return {
        "ExampleName/Meta-Llama-3.1-8B-Instruct": 4,
        "ExampleName/mythomax-l2-13b": 4,
        "ExampleName/Hermes-3-Llama-3.1-8B": 4,
        "ExampleName/Nxcode-CQ-7B-orpo": 4,
        "ExampleName/deepseek-coder-33b-instruct": 4,
        "ExampleName/Llama-3.1-Nemotron-70B-Instruct-HF": 4,
    }
```

2. Update the `create_chat_completion` and `create_completion` methods in
   neurons/miner.py to route to the appropriate vllm upstream server based on
   the model (which is either in the headers or from the request payload's model
   param)

Once this is complete, you are ready to continue starting your miner node.

### PM2

Running a miner through PM2 will require the vLLM instance to be running.

```bash
pm2 start neurons/miner.py --name miner --interpreter  python3 -- --wallet.name [WALLET_NAME] --netuid 4 --wallet.hotkey [WALLET_HOTKEY] --subtensor.network finney --model-endpoint [MODEL_ENDPOINT] --api_key [API_KEY] --axon.port [AXON PORT] --logging.trace
```

> Please replace the following with your specific configuration:
>
> - \[WALLET_NAME\]
> - \[WALLET_HOTKEY\]
> - \[MODEL_ENDPOINT\]
> - \[API_KEY\]
> - \[AXON_PORT\]

NOTE: Trace logging is very verbose. You can use `--logging.info` instead for
less log bloat.

Additionally:

```bash
--no-force-validator-permit [TRUE/FALSE]

```

is defaulted to false to force incoming requests to have a permit. Set this to
true if you are having trouble getting requests from validators on the 'test'
network.

## Running a Validator

### PM2

Validators are simply run through pm2, enabling auto restarts and auto updates.
A validator should be run on at least an A100, but the larger the better, as
larger clusters can handle more models. The machine should have
[nvidia-smi / cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)
installed along with [docker](https://docs.docker.com/engine/install/ubuntu/).

**No vllm instance needed**

Validator Instance:

```bash
pm2 start neurons/validator.py --name validator --interperter python3 -- --wallet.name [WALLET_NAME]

```

> Please replace the following with your specific configuration:
>
> - \[WALLET_NAME\]

## Explanation of Args

### Shared Args

1. **--netuid** ==> Subnet Netuid. *Defaults to 4*
1. **--epoch-length** ==> Default epoch length (how often we set weights,
   measured in 12 second blocks). *Defaults to 360*
1. **--mock** ==> Mock neuron and all network components. *Defaults to False*

### Miner Args

1. **--neuron.name** ==> Trials for this neuron go in neuron.root/ (wallet-cold
   \- wallet-hot) / neuron.name. *Defaults to miner*
1. **--force_validator.permit** ==> If set, forces incoming requests to have a
   permit. *Defaults to True*
1. **--model-endpoint** ==> Endpoint to use for the OpenAi CompatibleClient.
   *Defaults to "http://127.0.0.1:8000/v1"*
1. **--api-key** ==> API key for OpenAi Compatible API. *Defaults to "12345"*

### Validator Args

1. **--neuron.name** ==> Trials for this neuron go in neuron.root/ (wallet-cold
   \- wallet-hot) / neuron.name. *Defaults to validator*
1. **--timeout** ==> The timeout for each forward call in seconds. *Defaults to
   8*
1. **--vpermit-tao-limit** ==> The maximum number of TAO allowed to query a
   validator with a permit. *Defaults to 4096*
1. **--cache-file** ==> Pickle file to save score cache to. *Defaults to
   cache.pickle*
1. **--database.url** ==> Database URL to save Miner Data to Targon Hub.
1. **--autoupdate-off** ==> Disable automatic updates to Targon on latest
   version on Main if set. *Defaults to True*
1. **--models.mode** ==> Mode to use for determining what models to run. Can be
   one of:`default`, or `config`.
   - `endpoint`: defaults to `https://targon.sybil.com/api/models`. This will
     mimic the manifold validator
   - `default`: only run NousResearch/Meta-Llama-3.1-8B-Instruct
   - `config`: parse a text file named `models.txt` with a list of models
     separated by newlines
1. **--models.endpoint** ==> Only used when models.mode is `endpoint`. Sets the
   api endpoint to ping for list of models. Defaults to targon hub.

> Example model config file `models.txt`
>
> ```
> NousResearch/Meta-Llama-3.1-8B-Instruct
> NousResearch/Meta-Llama-3.1-70B-Instruct
> NousResearch/Meta-Llama-3.1-405B-Instruct
> ```

## Validator .env

Some more robust settings can be applied via a .env file. Eventually, targon
aims to move all settings to a .env file instead of cli arguments. Currently,
the following .env variables are supported with their defaults.

```
AUTO_UPDATE=False # Turn off autoupdate. Overrides cli flag.
IMAGE_TAG=latest # Verifier image tag. Useful for testing new updates.
HEARTBEAT=False # Enable heartbeat. Requires pm2. Set to True to enable heartbeat monitoring.
IS_TESTNET=False # If validator should run in testnet.
```

## Autoupdate

Autoupdate is implemented in targon/utils.py. This is to ensure that your
codebase matches the latest version on Main of the Targon Github Repository.

### Validator Autoupdate

Validator Autoupdate is implemented and defaulted to run once weights have been
set. To **disable**, please add the flag to your command line build:

```bash
pm2 start neurons/validator.py --name validator --interpreter python3 -- --wallet.name [WALLET_NAME] --autoupdate-off
```

### Miner Autoupdate

Miner Autoupdate is **not** implemented. Miners will need to check the Targon
repository and update themselves as new versions are released. If interested in
utilizing the autoupdate feature that Validators use, please follow the steps
below:

*NOTE*: This will not be maintained by the Manifold Labs Team.

1. Import the autoupdate function into your miner script (neurons/miner.py) at
   the top of the file.

```python
from targon.updater import autoupdate
```

3. Call the function at a place of your choosing.

```python
    if self.config.autoupdate:
        autoupdate(branch="main")

```

4. Relaunch your miner with the changes.

# What is A Deterministic Verification Network

Validators send queries to miners that are then scored for speed, and verified
by comparing the logprobs of the responses to a validators own model.

## Role of a Miner

A miner is a node that is responsible for generating a output from a query, both
organic and synthetic.

## Role of a Validator

A validator is a node that is responsible for verifying a miner's output. The
validator will send an openai compliant request to a miner with. The miner will
then send back a response with the output. The validator will then use the log
prob values of the response to verify that each miners response is accurate.
Validators will keep score of each miners response time and use their averages
to assign scores each epoch. Specifically, miner scores are the sum of the
average TPS per model.

# Features of Targon

## Full OpenAI Compliance

Validators can query miners directly using any openai package, and Epistula
headers. Below is boilerplate for querying a miner in python.

```py
miner = openai.AsyncOpenAI(
    base_url=f"http://{axon.ip}:{axon.port}/v1",
    api_key="sn4",
    max_retries=0,
    timeout=Timeout(12, connect=5, read=5),
    http_client=openai.DefaultAsyncHttpxClient(
        event_hooks={
            "request": [
                # This injects Epistula headers right before the request is sent.
                # wallet.hotkey is the public / private keypair
                #
                # You can find this function in the `epistula.py` file in 
                # the targon repo
                create_header_hook(wallet.hotkey, axon.hotkey_ss58)
            ]
        }
    ),
)
```

# How to Contribute

## Code Review

Project maintainers reserve the right to weigh the opinions of peer reviewers
using common sense judgement and may also weigh based on merit. Reviewers that
have demonstrated a deeper commitment and understanding of the project over time
or who have clear domain expertise may naturally have more weight, as one would
expect in all walks of life. Where a patch set affects consensus-critical code,
the bar will be much higher in terms of discussion and peer review requirements,
keeping in mind that mistakes could be very costly to the wider community. This
includes refactoring of consensus-critical code. Where a patch set proposes to
change the Targon subnet, it must have been discussed extensively on the discord
server and other channels, be accompanied by a widely discussed BIP and have a
generally widely perceived technical consensus of being a worthwhile change
based on the judgement of the maintainers. That being said, Manifold welcomes
all PR's for the betterment of the subnet and Bittensor as a whole. We are
striving for improvement at every interval and believe through open
communication and sharing of ideas will success be attainable.
