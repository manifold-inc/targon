# SN4 Mining Setup

Mining on Targon runs on **TargonOS** — a signed, purpose-built operating system that provisions the machine end to end. You install it once; on every boot it attests to the Targon network, brings up its encrypted storage, and starts the Targon services on its own.

Steps to install TargonOS can be found in the [TargonOS installation guide](installing-targonos.md).

> TargonOS replaces the previous provisioning flow (manual host OS preparation + the TVM installer). If you have nodes set up the old way, reprovision them by installing TargonOS.

## Emissions

Miner emissions are paid per card based on each auction's target price and the amount of compute live on the network: while an auction is below its card target, every live card earns up to the payment cap; as more cards come online, the pool is spread across them and the per-card rate settles toward the target price. To view the current emission targets, query the auctions API:

```bash
curl -X GET https://tower.targon.com/api/v2/auctions | jq
```

or visit [stats.targon.com/targets](https://stats.targon.com/targets) to see the compute emission targets.

**Example response** (truncated):

```json
{
  "auctions": {
    "TDX-VM-NVIDIA-H200": {
      "target_cards": 96,
      "target_price": 290,
      "max_price": 350,
      "min_cluster_size": 8
    },
    "TDX-VM-NVIDIA-B200": {
      "target_cards": 16,
      "target_price": 420,
      "max_price": 650,
      "min_cluster_size": 8
    }
  },
  "tao_price": 215.13482275
}
```

| Field | Meaning |
| --- | --- |
| `target_cards` | Maximum number of cards targeted for the auction. |
| `target_price` | Targeted price per card. |
| `max_price` | Payment cap per card — never paid more than this. |
| `min_cluster_size` | Minimum cluster size required to participate. |

All prices are in USD per hour per card, reported in cents — e.g. a `max_price` of `350` means $3.50/hour.

> ⚠️ **Important:** Any emissions not allocated to auctions are burned.

## Supported Hardware Configurations

TargonOS only installs on machines that match one of the hardware profiles below. During install, the hardware verification step checks the node against `tower.targon.com` and refuses to proceed if the machine does not meet a profile's minimums.

| Profile | CPU Vendor | GPU | GPU Count | Min RAM | Min CPU Threads | Min Combined Storage |
| --- | --- | --- | --- | --- | --- | --- |
| `h100-8x` | Intel | H100 SXM | 8 | 1.5 TiB (1,649,267,441,664 bytes) | 128 | 3 TB (3,000,000,000,000 bytes) |
| `h200-8x` | Intel | H200 | 8 | 1.9 TiB (2,089,072,092,774 bytes) | 192 | 3 TB (3,000,000,000,000 bytes) |
| `b200-8x` | Intel | B200 | 8 | 1.9 TiB (2,089,072,092,774 bytes) | 192 | 3 TB (3,000,000,000,000 bytes) |
| `b300-8x` | Intel | B300 | 8 | 1.9 TiB (2,089,072,092,774 bytes) | 192 | 3 TB (3,000,000,000,000 bytes) |
| `rtx-pro-6000-blackwell-8x` | Intel | RTX PRO 6000 Blackwell | 8 | 1,000 GiB (1,073,741,824,000 bytes) | 192 | 3 TB (3,000,000,000,000 bytes) |

Notes:

- **CPU vendor:** Intel TDX is required, since TargonOS relies on Intel TDX for the TEE (see [BIOS configuration](installing-targonos.md#bios-configuration)).
- **GPU count:** Each profile requires exactly 8 GPUs of the listed SKU.
- **RAM / CPU threads:** These are minimums; machines with more RAM or threads are accepted.
- **Encrypted storage:** The installer wipes every eligible disk and builds encrypted (LUKS) storage on top of them. The combined eligible disk capacity must meet the profile's minimum.

## Installing TargonOS

Follow the [TargonOS installation guide](installing-targonos.md). It covers the BIOS settings the installer requires, and both install paths:

- **iPXE netboot (unattended)** — best for fleets; pass your hotkey on the kernel command line and the install runs with no console interaction.
- **ISO (interactive)** — best for one-off installs; boot from USB or BMC virtual media and the installer walks you through network setup, disk selection, hardware verification, and hotkey entry.

> ⚠️ **Warning:** The installer **wipes every eligible disk** on the machine. Only run it on hardware you intend to dedicate to TargonOS.

After the install, the machine reboots into TargonOS and takes care of itself — attestation, encrypted storage, and the Targon services all come up automatically on every boot. No further setup is required on the node.

## Updating or Running the Miner

Once your TargonOS nodes are up, update your miner configuration to report the IP address of each node you are running.

1.  **Update or create the configuration file**

Edit `config.json` to include the IP addresses of your TargonOS nodes. Add them to the list of endpoints your miner reports to the network, along with any other desired parameters.

Example `config.json` (comments are for reference — remove them from the actual file, as JSON does not support comments):

```json
{
  // Only include the pure IP address of each node
  "nodes": [
    { "ip": "0.0.0.0" },
    { "ip": "0.0.1.1" }
  ],
  "hotkey_phrase": "one one one one one three one one one one one two",
  // External IP of your miner, used to register the axon on Bittensor
  "ip": "160.202.129.179",
  // Port for the miner to use
  "port": 7777,
  // Chain endpoint for your miner to connect to
  "chain_endpoint": "wss://test.finney.opentensor.ai:443",
  // Netuid to use — only change when running a testnet miner on 337
  "netuid": 4,
  // Minimum stake (in alpha) required for validators to get nodes
  "min_stake": 1000
}
```

> **Note:** Keep your node IPs up to date. If you add or remove TargonOS nodes, update this configuration accordingly.

2.  **Start the miner**

```bash
docker compose -f deploy/docker-compose-miner.yml up -d --build
```

or:

```bash
make up-miner
```