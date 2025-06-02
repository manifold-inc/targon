# Targon: The Confidential Decentralized AI Cloud

Targon is a next-generation AI infrastructure platform that leverages
Confidential Compute (CC) and Protected pcie (PPCIE) technology to secure the
entire stack. By providing a secure execution environment from hardware to
application layers, Targon enables verifiable and trustworthy operations across
the entire infrastructure in a decentralized fashion.

NOTICE: Using this software, you must agree to the Terms and Agreements provided
in the terms and conditions document. By downloading and running this software,
you implicitly agree to these terms and conditions.

## Table of Contents

1. [Overview](#overview)
   - [Core Security Features](#core-security-features)
   - [AI Infrastructure Capabilities](#ai-infrastructure-capabilities)
   - [Current Implementation](#current-implementation)
   - [Future Roadmap](#future-roadmap)
1. [Running a Validator](#running-a-validator)
   - [Validator Prerequisites](#validator-prerequisites)
     - [Docker Installation](#docker-installation)
     - [Docker Compose Installation](#docker-compose-installation)
     - [Other Prerequisites](#other-prerequisites)
   - [Validator Installation](#validator-installation)
     - [Clone Repository](#1-clone-repository)
     - [Environment Setup](#2-environment-setup)
     - [Build Services](#3-build-services)
   - [Validator Service Architecture](#validator-service-architecture)
     - [NVIDIA Attest Service](#nvidia-attest-service)
     - [Targon Service](#targon-service)
   - [Validator Monitoring and Maintenance](#validator-monitoring-and-maintenance)
     - [Logs](#logs)
     - [Updates](#updates)
     - [Troubleshooting](#troubleshooting)
1. [Running a Miner](#running-a-miner)
   - [Miner Prerequisites](#miner-prerequisites)
   - [Launching TVM](#launching-tvm)
     - [TVM Configuration](#tvm-configuration)
     - [TVM Installation](#tvm-installation)
     - [Updating Miner Config](#updating-miner-configuration)
1. [Contribution Guidelines](#contribution-guidelines)

## Overview

Targon provides a comprehensive secure AI infrastructure powered by CC and PPCIE
technology:

### Core Security Features

- Hardware-enforced memory encryption and protection
- Secure boot with hardware root of trust
- GPU TEE (Trusted Execution Environment) for isolated execution
- Remote attestation for verifiable computation
- Secure key management and cryptographic operations
- Protected execution environment with memory isolation

### AI Infrastructure Capabilities

- End-to-end secure model inference pipeline
- Hardware-level attestation and verification
- Protected model execution with CC or PPCIE isolation
- Verifiable computation through remote attestation
- Secure memory management for AI workloads
- Isolated execution environment for sensitive operations

### Current Implementation

- NVIDIA Confidential Compute integration OR
- NVIDIA PPCIE
- Hardware-level security guarantees
- Protected inference execution
- Remote attestation capabilities
- Secure memory encryption
- Isolated compute resources

### Future Roadmap

- Secure model training with Confidential Compute or PPCIE
- Protected data processing and storage
- Bare metal access for secure AI workloads
- Comprehensive AI development platform
- Developer-friendly tools and interfaces
- Automated scaling and resource management
- Multi-vendor Confidential Compute / PPCIE support:
  - AMD SEV-SNP integration
  - Additional hardware security technologies

## Running a Validator

### Validator Prerequisites

#### Docker Installation

1. **Set up Docker's apt repository**

   ```bash
   # Add Docker's official GPG key
   sudo apt-get update
   sudo apt-get install ca-certificates curl gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg

   # Add the repository to Apt sources
   echo \
     "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

1. **Install Docker Engine**

   ```bash
   # Update the package index
   sudo apt-get update

   # Install Docker Engine, containerd, and Docker Compose
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

1. **Verify Installation**

   ```bash
   # Verify Docker Engine installation
   sudo docker run hello-world
   ```

1. **Post-installation Steps**

   ```bash
   # Create the docker group
   sudo groupadd docker

   # Add your user to the docker group
   sudo usermod -aG docker $USER

   # Activate the changes to groups
   newgrp docker
   ```

#### Docker Compose Installation

Docker Compose is included in the Docker Engine installation above. Verify it's
installed:

```bash
docker compose version
```

#### Other Prerequisites

- Basic compute resources (any device capable of running a Go binary and Python)

### Validator Installation

#### 1. Clone Repository

```bash
git clone https://github.com/manifold-inc/targon.git
cd targon
```

#### 2. Environment Setup

Create a `.env` file in the project directory:

```bash
### Required
HOTKEY_PHRASE="your hotkey phrase"

### Optional (dont set unless needed)

## Attest endpoint, if running go code outside docker compose
# NVIDIA_ATTEST_ENDPOINT=http://nvidia-attest:3344

## Netuid validator is running on. Useful for testnet
# NETUID=4

## Chain to connect to
# CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443

## Sends discord notifications for things like setting weights
# DISCORD_URL=
```

#### 3. Run Targon

```bash
# Build NVIDIA Attest Service
docker compose up -d --force-recreate
```

### Validator Service Architecture

#### NVIDIA Attest Service

- Handles GPU attestation
- Required for validator operations
- Provides hardware-level security verification

#### Targon Service

- Main application service
- Handles validator operations
- Communicates with Bittensor network
- Built from Go source code

### Validator Monitoring and Maintenance

#### Logs

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f nvidia-attest
```

#### Updates

```bash
# Pull latest code
git pull

# Rebuild and restart services
docker compose up -d --build --force-recreate
```

#### Troubleshooting

```bash
# Check service status
docker compose ps

# View detailed logs
docker compose logs -f --tail=100

# Restart services
docker compose up -d --build --force-recreate
```

## Running a Miner

### Miner Prerequisites

Targon currently supprots Intel TDX Enabled Machines. Please refer to the
documentation attached to set up your hardware before launching TVM.

[Nvidia Deployment Guide for Intel TDX and KVM](https://docs.nvidia.com/cc-deployment-guide-tdx.pdf)

Please ensure that your GPUs are configured to PPCIE Mode and you have more than
1.5 TB of storage.

### Launching TVM

After completing all the prerequisite steps above, you are ready to run TVM.
This process will:

1. Attest your hardware configuration
1. Execute the launch script for TVM
1. Verify the environment Integrity

If you encounter any errors during this process, please review and correct your
hardware configuration according to the guidelines in the previous sections.

#### TVM Configuration

Before installing TVM, you'll need to gather the following information:

1. **Required Arguments**

   - `--miner-hot-key`: Your miner SS58 address that is **REGISTERED** on SN4
   - `--private-key`: The corresponding private key for your hotkey (without 0x
     prefix)
   - `--public-key`: Your corresponding public key (without 0x prefix)
   - `--validator-hot-key`: Validator hotkey for Epistula headers (always the
     Manifold Validator hotkey)

1. **Accessing Your Keys**

   ```bash
   cd bittensor
   cd wallets
   cd default  # or your wallet name
   cd hotkeys
   cat <your_hotkey_name>
   ```

   This will display your hotkey information:

   ```json
   {
     "accountId": "0x...",
     "publicKey": "0x...",
     "privateKey": "...",
     "secretPhrase": "wool ...",
     "secretSeed": "0x...",
     "ss58Address": "5..."
   }
   ```

#### TVM Installation

1. **Clone Repository**

   ```bash
   # Clone the repository
   git clone https://github.com/manifold-inc/targon.git
   cd targon
   ```

1. **Run TVM Installer**

   ```bash
   # Run the TVM installer with network submission
   ./tvm/install --submit --service-url http://tvm.targon.com:8080 --miner-hot-key MINER_HOT_KEY --private-key PRIVATE_KEY --public-key PUBLIC_KEY --validator-hot-key 5Hp18g9P8hLGKp9W3ZDr4bvJwba6b6bY3P2u3VdYf8yMR8FM
   ```

   > **Note**: To test without submitting to the network, remove the
   > `--submit --service-url` flags.

If you encounter any issues during verification, ensure that:

- All prerequisite steps were completed successfully
- Your GPU is properly configured for Confidential Compute / PPCIE
- Your keys are correctly formatted and valid

At this point, all setup on your TVM nodes is complete.

> WARNING: You must wait approximately 1 hour after this step before updating
> your configuration file for CVM Nodes. Otherwise, you will fail attestation.

### Updating/Running Miner

After setting up your TVM nodes, you need to update your miner configuration to
report the IP addresses of each CVM you are running.

1. **Update/Create the Configuration File** Edit `config.json` file to include
   the IP addresses of your TVM nodes. Add them to the list of endpoints that
   your miner reports to the network, along with any other desired paramaters.

   Example `config.json`

   ```json
    {
    // ONLY include pure IP address of each node
    "nodes": ["0.0.0.0", "1.1.1.1"],
    "hotkey_phrase": "one one one one one three one one one one one two",
    // External ip of your miner, used to register axon on bittensor
    "ip": "160.202.129.179",
    // Port for miner to use
    "port": 7777,
    // Chain endpoint for your miner to connect to
    "chain_endpoint": "wss://test.finney.opentensor.ai:443",
    // Netuid to use, only change when running testnet miner on 337
    "netuid": 4
    }
   ```

> **Note**: Make sure to keep your TVM node IPs up to date. If you add or remove
> TVM nodes, update this configuration accordingly.

1. **Start Miner** Run
   `docker compose -f docker-compose.miner.yml up -d --build`
