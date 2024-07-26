# Targon: A Redundant Deterministic Verification of Large Language Models

Targon (Bittensor Subnet 4) is a redundant deterministic verification mechanism
that can be used to interpret and analyze ground truth sources and a query.

NOTICE: Using this software, you must agree to the Terms and Agreements provided
in the terms and conditions document. By downloading and running this software,
you implicitly agree to these terms and conditions.

# Table of Contents
1. [Compute Requirements](#recommended-compute-requirements)
2. [Installation](#installation)
    - [Install PM2](#install-pm2)
    - [Install Targon](#install-targon)
3. [What is a Redundant Deterministic Verification Network?](#what-is-a-redundent-determinsitic-verfification-network)
    - [Role of a Miner](#role-of-a-miner)
    - [Role of a Validator](#role-of-a-validator)
4. [Features of Targon](#features-of-targon)
    - [Infenrence Request](#inference-request)
    - [Jaro-Winkler Scoring](#jaro-winkler-scoring)
5. [How to Run Targon](#how-to-run-targon)
    - [How to run a Miner](#how-to-run-a-miner)
    - [How to run a Validator](#how-to-run-a-validator)
6. [How to Contribute](#how-to-contribute)


# Recommended Compute Requirements
The following table shows the suggested compute providers for running a validator or miner.

| Provider   | Cost  | Location |   Machine Type   | Rating |
|------------|-------|----------|--------------|--------|
| TensorDock | Low   | Global   | VM & Container   | 4/5    |
| Latitude   | Medium| Global   | Bare Metal       | 5/5    |
| Paperspace | High  | Global   | VM & Bare Metal  | 4.5/5  |
| GCP        | High  | Global   | VM & Bare Metal  | 3/5    |
| Azure      | High  | Global   | VM & Bare Metal  | 3/5    |
| AWS        | High  | Global   | VM & Bare Metal  | 3/5    |
| Runpod     | Low   | Global   | VM & Container   | 5/5    |

# Installation

## Overview
In order to run Targon, you will need to install PM2 and the Targon package. The following instructions apply only to Ubuntu OSes. For your specific OS, please refer to the official documentation.

### Install PM2 on your machine

#### Download NVM
To install or update nvm, you should run the install script. To do that, you may either download and run the script manually, or use the following cURL or Wget command:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```
#### Add NVM to bash profile

Running either of the above commands downloads a script and runs it. The script clones the nvm repository to \~/.nvm, and attempts to add the source lines from the snippet below to the correct
profile file (~/.bash_profile, ~/.zshrc, ~/.profile, or ~/.bashrc).

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
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

You have now installed Targon. You can now run a validator or a miner.

# What is a Redundant Deterministic Verification Network?

The query and a deterministic seed are used to generate a ground truth output with the specified model, which can be run by the validator
or as a light client. The validator then sends requests to miners with the query and deterministic seed. The miner output
are compared to the ground truth output. If the tokens are equal in reference to Jaro-Winkler Similarity, the miner has successfully completed the challenge.

## Role of a Miner

A miner is a node that is responsible for generating a output from a query and a deterministic sampling params. 

## Role of a Validator

A validator is a node that is responsible for verifying a miner's output. The validator will send a request to a miner with a query and 
deterministic sampling params. The miner will then send back a response with the output. The validator will then compare the output to the ground truth
output. If the outputs are equal in reference to Jaro Winkley Similarity, then the miner has completed the challenge.

# Features of Targon

## Inference Request

An inference request is a request sent by a validator to a miner. The inference request contains a query and inference sampling params. 
The miner will then generate an output from the query and deterministic sampling params. The miner will then stream the output back to
the validator.

> *CAVEAT:* Every Interval (360 blocks) there will be a random amount of inference samples by the validator. The validator will then compare the outputs to
the ground truth outputs. The Jaro-Winkler Simalrity of the outputs will be used to determine the reward for the miner. If the miner fails verification, their score will be 0.

## Jaro-Winkler Similarity

The Jaro-Winkler Similarity is a string metric measuring edit distance between two strings. Jaro-Winkler Similarity is defined as:
```math
S_w = S_j + P * L * (1 - S_j)
```
```Bash
where: 
S_w = Jaro-Winkler Similarity
S_j = Jaro Similarity
P = Scaling Factor
L = Length of the matching prefix up to a max of 4 characters

```

Jaro similarity is the measure of similarity between two strings. The value of Jaro distance ranged from 0 to 1 where 0 means no similarity and 1 means equality.

The Jaro similarity is calculated using the following formula 
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

In the context of Targon, the scaling factor for our Jaro-Winkler calculation was set to 0.25.

# How to Run Targon

## Running a Miner

### PM2
Running a miner through PM2 will require the LLM instance of your choice to be running.


```bash
pm2 start neurons/miner.py --name miner --interperter python3 -- --wallet.name [WALLET_NAME] --netuid 4 --wallet.hotkey [WALLET_HOTKEY] --subtensor.network finney --neuron.model_endpoint [MODEL_ENDPOINT] --nueron.api_key [NEURON_API_KEY] --axon.port [AXON_PORT] --logging.trace
```

> Please replace the following with your specific configuration:
>  - [WALLET_NAME]
>  - [WALLET_HOTKEY]
>  - [MODEL_ENDPOINT]
>  - [NEURON_API_KEY]
>  - [AXON_PORT]

NOTE: Trace logging is very verbose. You can use `--logging.debug` instead for less log bloat.

Additionally:
```bash
--blacklist.force_validator_permit [TRUE/FALSE]

```
is defaulted to true to force incoming requests to have a permit.

## Running a Validator

### PM2
Running a validator through PM2 will require the LLM instance of your choice to be running. 

```bash
pm2 start neurons/validator.py --name validator --interperter python3 -- --wallet.name [WALLET_NAME] --netuid 4 --subtensor.network finney --neuron.model_endpoint [MODEL_ENDPOINT] --neuron.proxy.port [PROXY_PORT] --neuron.api_key [NEURON_API_KEY] --neuron.epoch_length 360

```


> Please replace the following with your specific configuration:
>  - [WALLET_NAME]
>  - [MODEL_ENDPOINT]
>  - [PROXY_PORT]
>  - [NEURON_API_KEY]

## Explanation of Args

### Shared Args
1. **--netuid** ==> Subnet Netuid. *Defaults to 4*
2. **--neuron.epoch_length** ==> Default epoch length (how often we set weights, measured in 12 second blocks). *Defaults to 360*
3. **--mock** ==> Mock neuron and all network components. *Defaults to False*
4. **--neuron.model_endpoint** ==> Endpoint to use for the OpenAi CompatibleClient. *Defaults to "http://127.0.0.1:8080"*
5. **--neuron.model_name** ==> Name of the model used for completion. *Defaults to "mlabonne/NeuralDaredevil-7B"*
6. **--neuront.api_key** ==> API key for OpenAi Compatible API. *Defaults to "12345"*

### Miner Args
1. **--neuron.name** ==> Trials for this neuron go in neuron.root/ (wallet-cold - wallet-hot) / neuron.name. *Defaults to miner*
2. **--blacklist.force_validator.permit** ==> If set, forces incoming requests to have a permit. *Defaults to True*

### Validator Args
1. **--neuron.name** ==> Trials for this neuron go in neuron.root/ (wallet-cold - wallet-hot) / neuron.name. *Defaults to validator*
2. **--neuron.timeout** ==> The timeout for each forward call in seconds. *Defaults to 12*
3. **--neuron.sample_size** ==> The number of miners to query in a single step. *Defaults to 48*
4. **--nueron.vpermit_tao_limit** ==> The maximum number of TAO allowed to query a validator with a permit. *Defaults to 4096*

# How to Contribute

## Code Review
Project maintainers reserve the right to weigh the opinions of peer reviewers using common sense judgement and may also weigh based on merit. 
Reviewers that have demonstrated a deeper commitment and understanding of the project over time or who have clear domain expertise may naturally
have more weight, as one would expect in all walks of life. Where a patch set affects consensus-critical code, the bar will be much higher in terms
of discussion and peer review requirements, keeping in mind that mistakes could be very costly to the wider community. This includes refactoring of consensus-critical code.
Where a patch set proposes to change the Targon subnet, it must have been discussed extensively on the discord server and other channels, be accompanied by
a widely discussed BIP and have a generally widely perceived technical consensus of being a worthwhile change based on the judgement of the maintainers. That being said,
Manifold welcomes all PR's for the betterment of the subnet and Bittensor as a whole. We are striving for improvement at every interval and believe through open communication
and sharing of ideas will success be attainable.
