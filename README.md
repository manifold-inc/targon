# Targon: A Redundant Deterministic Verification of Large Language Models

Targon (Bittensor Subnet 4) is a redundant deterministic verification mechanism
that can be used to interpret and analyze ground truth sources and a query.

NOTICE: Using this software, you must agree to the Terms and Agreements provided
in the terms and conditions document. By downloading and running this software,
you implicitly agree to these terms and conditions.

# Table of Contents

1. [Compute Requirements](#recommended-compute-requirements)
1. [Installation](#installation)
   - [Install PM2](#install-pm2)
   - [Install Targon](#install-targon-on-your-machine)
1. [How to Run Targon](#how-to-run-targon)
   - [Running a VLLM](#running-a-vllm)
   - [Running a Miner](#running-a-miner)
   - [Running a Validator](#running-a-validator)
1. [What is a Redundant Deterministic Verification Network?](#what-is-a-redundant-deterministic-verification-network)
   - [Role of a Miner](#role-of-a-miner)
   - [Role of a Validator](#role-of-a-validator)
1. [Features of Targon](#features-of-targon)
   - [Infenrence Request](#inference-request)
   - [Jaro-Winkler Similarity](#jaro-winkler-similarity)
   - [Targon-Hub](#targon-hub)
1. [How to Contribute](#how-to-contribute)

# Recommended Compute Requirements

The following table shows the suggested compute providers for running a
validator or miner.

| Provider   | Cost   | Location | Machine Type    | Rating |
| ---------- | ------ | -------- | --------------- | ------ |
| TensorDock | Low    | Global   | VM & Container  | 4/5    |
| Latitude   | Medium | Global   | Bare Metal      | 5/5    |
| Paperspace | High   | Global   | VM & Bare Metal | 4.5/5  |
| GCP        | High   | Global   | VM & Bare Metal | 3/5    |
| Azure      | High   | Global   | VM & Bare Metal | 3/5    |
| AWS        | High   | Global   | VM & Bare Metal | 3/5    |
| Runpod     | Low    | Global   | VM & Container  | 5/5    |

#### Minimum Viable Compute Requirements

- **VRAM:** 80 GB
- **Storage:** 200 GB
- **RAM:** 16 GB
- **CPU**: 4

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

## Running a VLLM

### VLLM

In order to run Targon, you must have a VLLM instance up and running.

First, install VLLM and JSON Schema on your machine

```bash
pip install vllm jsonschema
```

Now you are ready to server your VLLM instance to PM2

```bash
pm2 start vllm --name vllm-serve --interpreter python3 -- serve mlabonne/NeuralDaredevil-7B --dtype auto --api-key [some-secret-you-also-pass-to-validator] --port 8000
```

The `--neuron.model_endpoint` for miner / vali using this vllm instance on the
same machine would be `http://localhost:8000/v1`. **Make sure** to include the
`/v1` to the end of the URL.

Also note that `--api-key` is defined by you. Set it to something hard to guess,
and probably random. Whatever you decide, pass it in to both `--api-key` in
vllm, and `--neuron.api_key` on the miner / vali

## Running a Miner

### PM2

Running a miner through PM2 will require the LLM instance of your choice to be
running.

```bash
pm2 start neurons/miner.py --name miner --interperter python3 -- --wallet.name [WALLET_NAME] --netuid 4 --wallet.hotkey [WALLET_HOTKEY] --subtensor.network finney --neuron.model_endpoint [MODEL_ENDPOINT] --nueron.api_key [NEURON_API_KEY] --axon.port [AXON_PORT] --logging.trace
```

> Please replace the following with your specific configuration:
>
> - \[WALLET_NAME\]
> - \[WALLET_HOTKEY\]
> - \[MODEL_ENDPOINT\]
> - \[NEURON_API_KEY\]
> - \[AXON_PORT\]

NOTE: Trace logging is very verbose. You can use `--logging.info` instead for
less log bloat.

Additionally:

```bash
--blacklist.force_validator_permit [TRUE/FALSE]

```

is defaulted to true to force incoming requests to have a permit.

## Running a Validator

### PM2

Running a validator through PM2 will require the LLM instance of your choice to
be running.

```bash
pm2 start neurons/validator.py --name validator --interperter python3 -- --wallet.name [WALLET_NAME] --netuid 4 --subtensor.network finney --neuron.model_endpoint [MODEL_ENDPOINT] --neuron.api_key [NEURON_API_KEY]

```

> Please replace the following with your specific configuration:
>
> - \[WALLET_NAME\]
> - \[MODEL_ENDPOINT\]
> - \[NEURON_API_KEY\]


### Targon Hub

The goal of the hub is to give validators a simple way to directly generate revenue off of their bittensor bandwidth. 
This is designed as a template for validators to take and create their own branded hubs with, however pull requests are still encouraged.

This is also the place where miners can view their individual performace in real-time. Miners will be able to see:
- Jaro Score
- Time to First Token
- Time for All Tokens
- Total Time
- Tokens Per Second

If you are interested in running your own instance of Targon Hub, you will need to add an additonal flag to save the records of miners' responses to a PostgreSQL DB.

**NOTE**: No flag means no database!

```bash
--database.url [DB_CONNECTION_STRING]

```
> Please replace the following with your specific connection URL:
>
> - \[DB_CONNECTION_STRING\]

Below are steps to create a Supabase connection string to utilze this feature:

1. Either create an account or log in to [Supabase](https://supabase.com/dashboard/sign-in])
2. You might be asked to create an organization. In which case, choose the options best suited for your use case.
3. Once completed, create a new project with a secure password and location of your choosing. 
   Save your password, you will need it later. Your project will then take a few minutes to be provisioned.
4. Once the project has been created, click on the green ```Connect``` button near the top right of the screen
5. A modal should open up. Click on connection string, URI, and change the mode from ```transaction``` to ```session```
   in the dropdown
6. Copy the connection string shown and insert your password
7. Clone [Targon Hub](https://github.com/manifold-inc/targon-hub) and follow its setup instructions
8. Launch the validator with new `--database.url` flag and connection string 

Please reach out to the SN4 team for help setting up targon hub in sn4 chat or [our discord](https://discord.gg/manifold)


```bash
pm2 start neurons/validator.py --name validator --interperter python3 -- --wallet.name [WALLET_NAME] --netuid 4 --subtensor.network finney --neuron.model_endpoint [MODEL_ENDPOINT] --neuron.api_key [NEURON_API_KEY] --database.url [DB_CONNECTION_STRING]
```

As your validator runs, you will start seeing records being added into your Supabase database. This will be directly what your Targon Hub will query.

## Autoupdate

Autoupdate is implemented in targon/utils.py. This is to ensure that your codebase matches the latest version on Main of the Targon Github Repository. 

### Validator Autoupdate

Validator Autoupdate is implemented and defaulted to run once weights have been set. To **disable**, please add the flag to your command line build:

```bash
pm2 start neurons/validator.py --name validator --interperter python3 -- --wallet.name [WALLET_NAME] --netuid 4 --subtensor.network finney --neuron.model_endpoint [MODEL_ENDPOINT] --neuron.api_key [NEURON_API_KEY] --autoupdate false
```

### Miner Autoupdate

Miner Autoupdate is **not** implemented. Miners will need to check the Targon repository and update themselves as new versions are released.
If interested in utilizing the autoupdate feature that Validators use, please follow the steps below:

*NOTE*: This will not be maintained by the Manifold Labs Team.

1. Import the autoupdate function into your miner script (neurons/miner.py) at the top of the file.

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
1. **--neuron.epoch_length** ==> Default epoch length (how often we set weights,
   measured in 12 second blocks). *Defaults to 360*
1. **--mock** ==> Mock neuron and all network components. *Defaults to False*
1. **--neuron.model_endpoint** ==> Endpoint to use for the OpenAi
   CompatibleClient. *Defaults to "http://127.0.0.1:8000/v1"*
1. **--neuron.model_name** ==> Name of the model used for completion. *Defaults
   to "mlabonne/NeuralDaredevil-7B"*
1. **--neuron.api_key** ==> API key for OpenAi Compatible API. *Defaults to
   "12345"*

### Miner Args

1. **--neuron.name** ==> Trials for this neuron go in neuron.root/ (wallet-cold
   \- wallet-hot) / neuron.name. *Defaults to miner*
1. **--blacklist.force_validator.permit** ==> If set, forces incoming requests
   to have a permit. *Defaults to True*

### Validator Args

1. **--neuron.name** ==> Trials for this neuron go in neuron.root/ (wallet-cold
   \- wallet-hot) / neuron.name. *Defaults to validator*
1. **--neuron.timeout** ==> The timeout for each forward call in seconds.
   *Defaults to 12*
1. **--neuron.sample_size** ==> The number of miners to query in a single step.
   *Defaults to 48*
1. **--nueron.vpermit_tao_limit** ==> The maximum number of TAO allowed to query
   a validator with a permit. *Defaults to 4096*
1. **--nueron.cache_file** ==> Pickle file to save score cache to. *Defaults to
  cache.pickle*
1. **--database.url** ==> Database URL to save Miner Data to Targon Hub.
1. **--autoupdate_off** ==> Disable automatic updates to Targon on latest version on Main if set. *Defaults to True* 

# What is a Redundant Deterministic Verification Network?

The query and a deterministic seed are used to generate a ground truth output
with the specified model, which can be run by the validator or as a light
client. The validator then sends requests to miners with the query and
deterministic seed. The miner's responses are scored based on Tokens Per Second
only if they pass a Jaro-Winkler Similarity test given a predefined threshold.

## Role of a Miner

A miner is a node that is responsible for generating a output from a query and a
deterministic sampling params.

## Role of a Validator

A validator is a node that is responsible for verifying a miner's output. The
validator will send a request to a miner with a query and deterministic sampling
params. The miner will then send back a response with the output. The validator
will then compare the output to the ground truth output. If the outputs are
*similar enough* in respect to Jaro Winkler Similarity, then the miner has
completed the challenge.

# Features of Targon

## Inference Request

An inference request is a request sent by a validator to a miner. The inference
request contains a query and inference sampling params. The miner will then
generate an output from the query and deterministic sampling params that is then
streamed back to the validator.

> *CAVEAT:* Every Interval (360 blocks) there will be a random amount of
> inference samples by the validator. The validator will then compare the
> outputs to the ground truth outputs. The Jaro-Winkler Simalarity of the
> outputs will be used to determine the reward for the miner if the miner is
> running an appropriate model. If the miner fails verification, their score
> will be 0.

## Jaro-Winkler Similarity

The Jaro-Winkler Similarity is a string metric measuring edit distance between
two strings. Jaro-Winkler Similarity is defined as:

```math
S_w = S_j + P * L * (1 - S_j)
```

```bash
where: 
S_w = Jaro-Winkler Similarity
S_j = Jaro Similarity
P = Scaling Factor
L = Length of the matching prefix up to a max of 4 characters

```

Jaro Similarity is the measure of similarity between two strings. The value of
Jaro distance ranged from 0 to 1 where 0 means no similarity and 1 means very
similar.

The Jaro Similarity is calculated using the following formula:

```math
S_j = 1/3((m/abs(s1)) + (m/abs(s2)) + (m-t)/m))

```

```bash
where:
S_j = Jaro Similarity
m = number of matching characters
t = half of the number of transpositions
s1 = length of first string
s2 = length of second string

```

In the context of Targon, the scaling factor for our Jaro-Winkler calculation
was set to 0.25.

> *Note:*
> [Jaro-Winkler Similarity](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance%5D)
> is not a mathematical metric, as it fails the Triangle Inequality.

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
