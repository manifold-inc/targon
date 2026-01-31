# Confidential GPU Mining Setup
### Miner Prerequisites
> ⚠️ **IMPORTANT** ⚠️  
> It is **HIGHLY RECOMMENDED** that you have BIOS
> access to your machines. You will be adjusting the configurations throughout
> the process and delays from providers are expected. Without BIOS access, you
> may face significant delays or be unable to complete the setup process.


This doc provides setup scripts and instructions for running miners in **confidential computing environments** with both **CPU (AMD SEV-SNP)** and **GPU passthrough (NVIDIA Hopper series)**.

## Table of Contents

* [Auctions](#auctions)
* [Supported Platforms](#supported-platforms)
* [AMD EPYC 4th Gen 9xx4 Series SEV-SNP Setup](amd-cpus.md)

    1. [Hardware Requirements (AMD)](amd-cpus.md#hardware-requirements-amd)
    2. [Software Requirements (AMD)](amd-cpus.md#software-requirements-amd)
    3. [BIOS Configuration (AMD SEV-SNP)](amd-cpus.md#bios-configuration-amd-sev-snp)
    4. [Host OS Preparation (AMD)](amd-cpus.md#host-os-preparation-amd)

* [Intel TDX + Hopper GPU Setup](intel-hopper-gpus.md)
    1. [Hardware Requirements (Intel)](intel-hopper-gpus#hardware-requirements-intel)
    2. [Software Requirements (Intel)](intel-hopper-gpus#software-requirements-intel)
    3. [BIOS Configuration (Intel TDX)](intel-hopper-gpus#bios-configuration-intel-tdx)
    4. [Host OS Preparation (Intel)](intel-hopper-gpus#host-os-preparation-intel)

* [Intel TDX + Blackwell GPU Setup](intel-blackwell-gpus.md)
    1. [Hardware Requirements (Intel)](intel-blackwell-gpus#hardware-requirements-intel)
    2. [Software Requirements (Intel)](intel-blackwell-gpus#software-requirements-intel)
    3. [BIOS Configuration (Intel TDX)](intel-blackwell-gpus#bios-configuration-intel-tdx)
    4. [Host OS Preparation (Intel)](intel-blackwell-gpus#host-os-preparation-intel)


* [Launching TVM](#launching-tvm)
    1. [TVM Configuration](#tvm-configuration)
    2. [TVM Installation](#tvm-installation)

* [Updating or Running Miner](#updating-or-running-miner)

# Auctions

Miner emissions are currently split proportionally to public demand. To view the current split of miner emissions, query the auctions API:

```bash
curl -X GET https://tower.targon.com/api/v1/auctions | jq
```

**Example Response:**

```json
{
  "auctions": {
    "SEV-CPU-AMD-EPYC-V4": {
      "max_bid": 100,
      "emission": 5,
      "min_cluster_size": 0,
      "node_type": ""
    },
    "TDX-NVCC-NVIDIA-H200": {
      "max_bid": 300,
      "emission": 60,
      "min_cluster_size": 8,
      "node_type": ""
    }
  },
  "tao_price": 272.31050874
}
```

> ⚠️ **Important:** The remainder of emissions not allocated to auctions is burned.

# Supported Platforms

### 1. AMD EPYC™ v4 (4th Gen 9xx4 Series: Genoa/Bergamo) with SEV-SNP

>Supports **CPU confidential workloads** inside SEV-SNP encrypted VMs.

**Hardware Requirements**

* **Processor:** AMD EPYC™ 9xx4 Series (Genoa/Bergamo) with SEV-SNP support
* **Storage:** 1 TB minimum

**Software Requirements**

* **Host OS:** Ubuntu 25.04 Server
* **HGX Firmware Bundle:** Version 1.7.0 or higher (a.k.a. Vulcan 1.7)


### 2. Intel TDX + NVIDIA Hopper GPUs (H100, H200)

>Supports **GPU passthrough** inside TDX confidential VMs.

**Hardware Requirements**

* **CPU:** 5th Gen Intel® Xeon® Scalable Processors (Intel® Xeon® 6 Processors)
* **GPU:** NVIDIA H100 or H200 with Confidential Compute support
* **Storage:** 3 TB minimum

**Software Requirements**

* **Host OS:** Ubuntu 22.04 LTS or later
* **HGX Firmware Bundle:** Version 1.7 (a.k.a. Vulcan 1.7)

### 3. Intel TDX + NVIDIA Blackwell GPUs (B200)

>Supports **GPU passthrough** inside TDX confidential VMs.

**Hardware Requirements**

* **CPU:** 5th Gen Intel® Xeon® Scalable Processors (Intel® Xeon® 6 Processors)
* **GPU:** NVIDIA B200 (multi-GPU configuration)
* **Storage:** 3 TB minimum

**Software Requirements**

* **Host OS:** Ubuntu 25.10
* **Guest OS:** Ubuntu 24.04 LTS

---

>⚠️ **Important:** Ensure firmware and drivers are up-to-date to leverage SEV-SNP or TDX capabilities and confirm system compatibility before provisioning confidential workloads.




# Setup Instructions by Platform

Choose your platform to view detailed setup instructions:

* [AMD EPYC 4th Gen SEV-SNP Setup](amd-cpus.md)
  Includes hardware, software, BIOS configuration, and host OS preparation for AMD CPUs.

* [Intel TDX + Hopper GPU Setup](intel-hopper-gpus)
  Includes hardware, software, BIOS configuration, and host OS preparation for Intel TDX with NVIDIA GPUs.

* [Intel TDX + Blackwell GPU Setup](intel-blackwell-gpus)
  Includes hardware, software, BIOS configuration, and host OS preparation for Intel TDX with NVIDIA Blackwell GPUs.


# Launching TVM
After completing all the prerequisite steps above, you are ready to run TVM.
This process will:

1. Attest your hardware configuration
1. Execute the launch script for TVM
1. Verify the Confidential Compute environment

If you encounter any errors during this process, please review and correct your
hardware configuration according to the guidelines in the previous sections.

## TVM Configuration

1. **Required Arguments**

   - `--hotkey-phrase`: Your miner Hotkey Phrase
   - `--node-type`: cpu or nvcc
   - `--submit`: Whether to actually submit and download image
   - `--service-url`: http://tvm.targon.com
   - `--vm-download-dir`: location you want to download vm
   - `--host-machine-storage`: maximum storage available on host-machine in TB (integer only, no decimals - use floor function, e.g., if storage is 5.9 TB, specify 5)

> **⚠️ Important Storage Warning**: If you specify the wrong storage size for `--host-machine-storage` and qcow2 cannot grow until the mentioned size, the machine will stop earning emission. Ensure you provide the correct amount of storage available on your host machine in TB. Also, check the minimum storage requirement for storage based on your platform type.

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


## TVM Installation

1. **Clone Repository**

   ```bash
   git clone https://github.com/manifold-inc/targon.git
   cd targon
   ```

2. **Run TVM Installer**

   **For AMD CPU Servers:**

   ```bash
   sudo ./tvm/install --service-url http://tvm.targon.com \
                      --vm-download-dir ./ \
                      --submit \
                      --hotkey-phrase "your phrase" \
                      --node-type cpu \
                      --host-machine-storage 21TB \
                      --launch-vm
   ```

   **For Intel TDX + Hopper GPUs:**

   ```bash
   sudo ./tvm/install --service-url http://tvm.targon.com \
                      --vm-download-dir ./ \
                      --submit \
                      --hotkey-phrase "your phrase" \
                      --node-type nvcc \
                      --host-machine-storage 21TB \
                      --launch-vm
   ```
    **For Intel TDX + Blackwell GPUs:**

   ```bash
   sudo ./tvm/install --service-url http://tvm.targon.com \
                      --vm-download-dir ./ \
                      --submit \
                      --hotkey-phrase "your phrase" \
                      --node-type blackwell \
                      --host-machine-storage 21TB \
                      --launch-vm
   ```


   > ✅ **Tip:** To test without submitting to the network, remove the `--submit --service-url` flags.


### Additional TVM Installer Flags

The installer supports several additional flags that allow more control over VM setup, reporting, and debugging:

| Flag                       | Description                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--json`                   | Generate a JSON report of the attestation checks.                                                                         |
| `--output <file>`          | Specify the output file for the JSON report (default is auto-generated).                                                  |
| `--version`                | Show version information and exit.                                                                                        |
| `--compact`                | Display compact output (only failed checks and summaries).                                                                |
| `--submit`                 | Submit the report to the attestation service.                                                                             |
| `--fix-warnings`           | Attempt to fix common warnings by installing packages, loading modules, and suggesting kernel parameters (non-intrusive). |
| `--service-url <url>`      | URL of the attestation service.                                                                                           |
| `--hotkey-phrase <phrase>` | Miner Hotkey Phrase for epistula.                                                                                         |
| `--vm-download-dir <path>` | Directory to download the VM to (default: `~/manifold-vms`).                                                              |
| `--node-type <type>`       | Node type; must be one of the supported types [`cpu`, `nvcc`].                                                      |
| `--host-machine-storage <storage-tb>` | Amount of storage in TB available on the host machine for VM operations. |
| `--launch-vm`              | Automatically extract the downloaded VM and launch it after installation.                                                 |

---

If you encounter any issues during verification, ensure that:

- All prerequisite steps were completed successfully
- Your GPU is properly configured for Confidential Compute
- Your keys are correctly formatted and valid

At this point, all setup on your TVM nodes is complete.



# Updating or Running Miner

After setting up your TVM nodes, you need to update your miner configuration to
report the IP addresses of each CVM you are running.

1. **Update/Create the Configuration File** Edit `config.json` file to include
   the IP addresses of your TVM nodes. Add them to the list of endpoints that
   your miner reports to the network, along with any other desired parameters.

   Example `config.json`

   ```json
    {
    // ONLY include pure IP address of each node and its bid (in cents per hour per gpu)
    "nodes": [{"ip":"0.0.0.0", "price": 120}, {"ip":"0.0.1.1", "price": 220}],
    "hotkey_phrase": "one one one one one three one one one one one two",
    // External ip of your miner, used to register axon on bittensor
    "ip": "160.202.129.179",
    // Port for miner to use
    "port": 7777,
    // Chain endpoint for your miner to connect to
    "chain_endpoint": "wss://test.finney.opentensor.ai:443",
    // Netuid to use, only change when running testnet miner on 337
    "netuid": 4,
    // Min stake in alpha required for validators to get nodes
    "min_stake": 1000
    }
   ```

> **Note**: Make sure to keep your TVM node IPs up to date. If you add or remove
> TVM nodes, update this configuration accordingly.

1. **Start Miner** Run
   `docker compose -f docker-compose.miner.yml up -d --build`

