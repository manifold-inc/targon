## Running a Validator

### Validator Prerequisites

- Docker installed (see docs/docker.md)
- Basic compute resources (any cpu server)

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
MONGO_USERNAME=
MONGO_PASSWORD=
VALIDATOR_IP="ipv4 address of the vali"

### Optional

# Sets a multiplier on all http connection timeouts
TIMEOUT_MULT=1

## Sends discord notifications for things like setting weights
# DISCORD_URL=

### Dev Variables (dont set unless needed)

## Attest endpoint, if running go code outside docker compose
# NVIDIA_ATTEST_ENDPOINT=http://nvidia-attest:3344

## Netuid validator is running on. Useful for testnet
# NETUID=4

## Chain to connect to
# CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443


```

> **Note** both mongo environment keys are keys **you define** and can be
> whatever you want. We suggest using the outpout of
> `tr -dc A-Za-z0-9 </dev/urandom | head -c 24; echo`. Run twice, and use one as
> the password and one as the username. These **do not** need to be double
> quoted in the .env file

#### 3. Build and Run Services

```bash
docker compose up -d --force-recreate --build
```

### Validator Service Architecture

#### NVIDIA Attest Service

- Handles GPU attestation
- Runs on port 3344
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
docker compose logs -f targon
```

#### Updates

```bash
# Pull latest code
git pull

# Rebuild and restart services
docker compose pull
docker compose up -d --build --force-recreate
```

#### Troubleshooting

```bash
# Check service status
docker compose ps

# View detailed logs
docker compose logs -f --tail=100

# Restart services
docker compose restart
```
