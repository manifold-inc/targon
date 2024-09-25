# Targon: A Deterministic Verification of Large Language Models

Targon (Bittensor Subnet 4) is a deterministic verification mechanism that is
used to incentivize miners to run openai compliant endpoints and serve synthetic
and organic queries.

NOTICE: Using this software, you must agree to the Terms and Agreements provided
in the terms and conditions document. By downloading and running this software,
you implicitly agree to these terms and conditions.

# Table of Contents

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

## Running manifolds VLLM for Miners

### VLLM

#### Docker

In order to run Targon, you must have the manifold vLLM fork instance that
includes powv tokens up and running. The easiest way is to run the vllm fork via
docker. It is recommended to add an environment variable for the api key.

> **NOTE**: The api key passed to vllm is your made up api key that you will
> also pass to your miner/validator. This ensures that no one else can access
> your vllm instance

> **NOTE** Some features like --tensor-parallel-size might not work without
> manual changes to vllm in coordination with powv. We suggest validators run
> the fork with minimal to no extra paramaters.

Install [docker](https://docs.docker.com/install/) and run

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "VLLM_API_KEY=<make up your own>" \
    -p 8000:8000 \
    --ipc=host \
    manifoldlabs/vllm-openai:powv \
    --model NousResearch/Meta-Llama-3.1-8B-Instruct
```

https://hub.docker.com/r/manifoldlabs/vllm-openai

#### PM2

Running without docker currently requires a dev installation of the vllm fork.

```bash
git clone https://github.com/manifold-inc/vllm-powv.git
cd vllm-powv
pip install jsonschema

# If this fails, you may need to set the path for cuda. This may be different per
# system. 
#
# CUDACXX=/usr/local/cuda-12/bin/nvcc pip install -e .
pip install -e .
```

Now you are ready to server your VLLM instance to PM2

```bash
pm2 start vllm --name vllm-serve --interpreter python3 -- serve NousResearch/Meta-Llama-3.1-8B-Instruct --dtype auto --api-key [some-secret-you-also-pass-to-validator] --port 8000
```

The `--neuron.model_endpoint` for miner / vali using this vllm instance on the
same machine would be `http://localhost:8000/v1`. **Make sure** to include the
`/v1` to the end of the URL.

Also note *again* that `--api-key` is defined by you. Set it to something hard
to guess, and probably random. Whatever you decide, pass it in to both
`--api-key` in vllm, and `--neuron.api_key` on the miner / vali

## Running a Miner

### PM2

Running a miner through PM2 will require the vLLM instance to be running.

```bash
pm2 start neurons/miner.py --name miner --interperter python3 -- --wallet.name [WALLET_NAME] --netuid 4 --wallet.hotkey [WALLET_HOTKEY] --subtensor.network finney --model-endpoint [MODEL_ENDPOINT] --api_key [API_KEY] --axon.port [AXON PORT] --logging.trace
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
--force-validator-permit [TRUE/FALSE]

```

is defaulted to true to force incoming requests to have a permit.

## Running a Validator

### PM2

Validators are simply run through pm2, enabling auto restarts and auto updates.

**No vllm instance needed**

Validator Instance:

```bash
pm2 start neurons/validator.py --name validator --interperter python3 -- --wallet.name [WALLET_NAME]

```

> Please replace the following with your specific configuration:
>
> - \[WALLET_NAME\]

### Targon Hub (WIP)

The goal of the hub is to give validators a simple way to directly generate
revenue off of their bittensor bandwidth. This is designed as a template for
validators to take and create their own branded hubs with, however pull requests
are still encouraged.

If you are interested in running your own instance of Targon Hub, you will need
to add an additional flag to save the records of miners' responses to a
PostgreSQL DB.

**NOTE**: No flag means no database!

```bash
--database.url [DB_CONNECTION_STRING]

```

> Please replace the following with your specific connection URL:
>
> - \[DB_CONNECTION_STRING\]

Below are steps to create a Supabase connection string to utilze this feature:

1. Either create an account or log in to
   [Supabase](https://supabase.com/dashboard/sign-in%5D)
1. You might be asked to create an organization. In which case, choose the
   options best suited for your use case.
1. Once completed, create a new project with a secure password and location of
   your choosing. Save your password, you will need it later. Your project will
   then take a few minutes to be provisioned.
1. Once the project has been created, click on the green `Connect` button near
   the top right of the screen
1. A modal should open up. Click on connection string, URI, and change the mode
   from `transaction` to `session` in the dropdown
1. Copy the connection string shown and insert your password
1. Clone [Targon Hub](https://github.com/manifold-inc/targon-hub) and follow its
   setup instructions
1. Launch the validator with new `--database.url` flag and connection string

Please reach out to the SN4 team for help setting up targon hub in sn4 chat or
[our discord](https://discord.gg/manifold)

```bash
pm2 start neurons/validator.py --name validator --interperter python3 -- --wallet.name [WALLET_NAME] --database.url [DB_CONNECTION_STRING]
```

As your validator runs, you will start seeing records being added into your
Supabase database. This will be directly what your Targon Hub will query.

## Autoupdate

Autoupdate is implemented in targon/utils.py. This is to ensure that your
codebase matches the latest version on Main of the Targon Github Repository.

### Validator Autoupdate

Validator Autoupdate is implemented and defaulted to run once weights have been
set. To **disable**, please add the flag to your command line build:

```bash
pm2 start neurons/validator.py --name validator --interperter python3 -- --wallet.name [WALLET_NAME] --autoupdate-off
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

# What is A Deterministic Verification Network

Targon uses a novel, in-house proof-of-work value to verify model responses.
These are generated just like log-probs, and can be used to verify with 99.99%
accuracy that a response was generated with the model requested. Verified
responses are scored by speed and consistency of verification.

## Role of a Miner

A miner is a node that is responsible for generating a output from a query, both
organic and synthetic.

## Role of a Validator

A validator is a node that is responsible for verifying a miner's output. The
validator will send an openai compliant request to a miner with. The miner will
then send back a response with the output. The validator will then use the proof
of work values of the response to verify that each miners response is accurate.
Validators will keep score of each miners response time and use their averages
to assign scores each epoch.

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
