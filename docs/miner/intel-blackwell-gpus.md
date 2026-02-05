# Intel TDX + Blackwell GPU Setup (Multi-GPU B200)

### Hardware Requirements (Intel)

- NVIDIA B200 GPUs (multi-GPU configuration)
- 3 TB Storage
- Intel CPU Requirements:
  - 5th Gen Intel® Xeon® Scalable Processor
  - Intel® Xeon® 6 Processors

### Software Requirements (Intel)

- Ubuntu 25.10 (Host OS)
- Ubuntu 24.04 LTS (Guest OS)

### BIOS Configuration (Intel TDX)

Enter your system **BIOS/UEFI** and configure the following settings:

```markdown
# CPU Configuration → Processor Configuration
Limit CPU PA to 46 Bits → Disable  

# Intel TME, Intel TME-MT, Intel TDX 
Total Memory Encryption (Intel TME) → Enable
Total Memory Encryption (Intel TME) Bypass → Auto
Total Memory Encryption Multi-Tenant (Intel TME-MT) → Enable
Memory Integrity → Disable
Intel TDX → Enable
TDX Secure Arbitration Mode Loader (SEAM) → Enabled
Disable excluding Mem below 1MB in CMR → Auto
Intel TDX Key Split → <Non-zero value>

# SGX
Software Guard Extension → Enabled  
SGX Factory Reset → Enabled  
```

## Preparing the Host

Ensure your system is up to date:

```bash
sudo apt update
sudo apt upgrade
sudo reboot # if required
```

## Download Required Packages (Host)

```bash
sudo apt update
sudo apt install qemu-system-x86 \
    ovmf \
    libvirt-daemon-system \
    libvirt-clients \
    infiniband-diags

# Install NVLSM package
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nvlsm_2025.06.10-1_amd64.deb
sudo apt install ./nvlsm_2025.06.10-1_amd64.deb
```

## Configuring the Host

Add nohibernate to grub in ``/etc/default/grub``
```bash
GRUB_CMDLINE_LINUX="nohibernate kvm_intel.tdx=1"
```

```bash
sudo update-grub
sudo grub-install --no-nvram

# Add user to kvm group:
LOG_USER=$(logname)
if [ -n "$LOG_USER" ] && [ "$LOG_USER" != "root" ]; then
  sudo usermod -aG kvm $LOG_USER
fi

sudo reboot
```

## Validating the Host Detects TDX

To check that your kernel is the new TDX-aware version, and that your configuration
options were correctly applied, run the following commands. Note that your TDX Module
version may be different.

```bash
sudo dmesg | grep -i tdx
```

Example output:

```
[ 10.162072] virt/tdx: BIOS enabled: private KeyID range [64, 128)
[ 10.162074] virt/tdx: Disable ACPI S3. Turn off TDX in the BIOS to use ACPI S3.
[ 21.678799] virt/tdx: TDX module 1.5.06.00, build number 744, build date 0134d817
[ 26.540654] virt/tdx: 8405028 KB allocated for PAMT
[ 26.540658] virt/tdx: module initialized
```

To update the TDX firmware, run the following sequence of commands:

```bash
cd /shared

# For Emerald Rapids only:
wget https://github.com/intel/tdx-module/releases/download/TDX_1.5.16/intel_tdx_module.tar.gz

# For Granite Rapids only:
wget https://github.com/intel/confidential-computing.tdx.tdx-module/releases/download/TDX_2.0.08/intel_tdx_module.tar.gz

tar -xvzf intel_tdx_module.tar.gz
sudo mkdir -p /boot/efi/EFI/TDX/
sudo cp TDX-Module/intel_tdx_module.so /boot/efi/EFI/TDX/TDX-SEAM.so
sudo cp TDX-Module/intel_tdx_module.so.sigstruct /boot/efi/EFI/TDX/TDX-SEAM.so.sigstruct
sudo reboot
```

> **Note**: The Intel TDX Module is the firmware code that should be kept up to date.
> Version 1.x should be used with Emerald Rapids, and version 2.x should be used with
> Granite Rapids.

## Autoload VFIO & IB UMAD

Linux Virtual Function I/O (VFIO) is a passthrough driver meant to bind the GPU on the
host to a guest virtual machine. IB UMAD is the module used to control NVSwitches for
multi-GPU Blackwell deployments. Creating the file below ensures the driver is ready to
be bound to the NVIDIA GPUs and/or NVLink Switch interconnects in future steps.

Create and open a new file: 
```bash
vim /etc/modules-load.d/vfio.conf
```
```bash
vfio
vfio_pci
ib_umad
```

## Prevent NVIDIA Drivers from Loading on the Host
Create and open a new file:
```bash
vim /etc/modprobe.d/
```
```bash
blacklist nvidia
blacklist nvidia_drm
blacklist nvidia_modeset
blacklist nvidia_uvm
```

## Installing Fabric Manager on Host (Blackwell Multi-GPU Only)

NVIDIA Fabric Manager (FM) is required to be installed and running for proper operation of
multi-GPU Blackwell CC. While FM may be installed within maintenance VMs or within the
guest VM itself, these instructions install it on the host.

```bash
sudo apt install nvidia-fabricmanager-590
```

Warning: Starting with branch 590, the Ubuntu packages have been renamed by removing the
branch designation from the package name. Switching branches, installing specific driver
versions, and upgrade or downgrade requirements will be supported through version
locking (pinning) packages. Refer to the Ubuntu 590 and later packages section of the
recent updates for more information.

You might need to enable the -proposed repository from Ubuntu:

```bash
sudo add-apt-repository "deb http://archive.ubuntu.com/ubuntu noble-proposed main restricted universe multiverse"
```

Fabric Manager requires a setting change in CC modes for Blackwell. Open
`/usr/share/nvidia/nvswitch/fabricmanager.cfg` and change `PARTITION_RAIL_POLICY=greedy`
to `PARTITION_RAIL_POLICY=symmetric`.

```bash
# Start FM
sudo systemctl enable nvidia-fabricmanager
sudo systemctl restart nvidia-fabricmanager
```

## Host OS Preparation (Intel)

For TD Quote Generation and TD Quote Verification, collateral is needed. Intel provides
the necessary collateral through the Intel® Provisioning Certification Service for ECDSA
Attestation (PCS).

### Provisioning Certificate Caching Service (PCCS)

To set up the PCCS in the next step, you need a subscription key for the Intel PCS. You
can obtain this from the [Intel Provisioning Certification Service](https://api.portal.trustedservices.intel.com/provisioning-certification).

1. If you did not request such a subscription key before, [subscribe to Intel PCS](https://api.portal.trustedservices.intel.com/products#product=liv-intel-software-guard-extensions-provisioning-certification-service), which requires to log in (or to create an account). Two subscription keys are generated (for key rotation) and both can be used for the following steps. Click on Subscribe at the bottom of the page. Then click on show for Primary Key and use that key.
<div style="display: flex; overflow-x: auto; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/05daf49a-fe47-4f73-82ba-5ef1bf724f28" width="250"/>
  <img src="https://github.com/user-attachments/assets/175d34f1-5dac-44cc-8a74-64ef29052124" width="250"/>
  <img src="https://github.com/user-attachments/assets/a55eb886-2a82-480b-ae42-5643a2795e90" width="250"/>
  <img src="https://github.com/user-attachments/assets/9296be95-c3d1-43bc-9750-c1ad0bd593a5" width="250"/>
  <img src="https://github.com/user-attachments/assets/ae8706df-2e20-46ed-92c5-28d6a0db5474" width="250"/>
  <img src="https://github.com/user-attachments/assets/0b48d18a-daeb-4215-8ed2-a5b1f1d7d1b0" width="250"/>
  <img src="https://github.com/user-attachments/assets/05351a14-4c08-4cda-9ddd-59346fc8023a" width="250"/>
</div>

2. If you did request such a subscription key before, [retrieve one of your keys](https://api.portal.trustedservices.intel.com/manage-subscriptions), which requires to log in. You have two subscription keys (for key rotation), and both can be used for the following steps.

If not done during another component installation, set up the appropriate Intel SGX
package repository for your distribution of choice:

```bash
echo 'deb [signed-by=/etc/apt/keyrings/intel-sgx-keyring.asc arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu noble main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list
wget https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key
sudo mkdir -p /etc/apt/keyrings
cat intel-sgx-deb.key | sudo tee /etc/apt/keyrings/intel-sgx-keyring.asc > /dev/null
sudo apt-get update
```

Install PCCS with following commands. The installer will prompt you for the following
configs. Answer the remaining questions according to your needs, e.g., your proxy
settings, a desired user password, and an admin password. The configuration step will
also allow you to create a self-signed SSL certificate for the PCCS.
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -yq --no-install-recommends nodejs=20.11.1-1nodesource1
sudo apt-get install -y cracklib-runtime
sudo apt install -y --no-install-recommends sgx-dcap-pccs
```

⚠️ **ONLY Execute If PCCS Installation Fails**:
Known Issue PCCS Installation Fails Due to npm audit
On Ubuntu 25.10.


**First, attempt the standard installation above.** If PCCS installation succeeds, **skip this entire section** and proceed to the configuration prompts below.

**If you encounter installation errors such as:**
- ``npm audit`` reporting high severity vulnerabilities
- ``dpkg: error processing package sgx-dcap-pccs (--configure)``
- ``post-installation script subprocess returned error exit status 1``

**Then apply this workaround:**

**Root Cause:** The PCCS installer runs npm audit during installation. With newer npm versions (common on Ubuntu 25.10), npm audit returns a non-zero exit code, which incorrectly causes the PCCS installation to fail, even though the vulnerabilities are in install-time tooling and do not affect PCCS runtime security.

**Fix Steps (Only If Installation Failed):**

**Edit the PCCS install script:**

```bash
sudo nano /opt/intel/sgx-dcap-pccs/install.sh
```

Find the line that runs:

```bash
npm audit
```

**Change it to:**

```bash
npm audit || true
```

This prevents npm audit from aborting the installation.

**Re-run package configuration:**

```bash
sudo dpkg --configure -a
```

PCCS should now configure successfully. After this, continue with the configuration prompts below.
| Prompt                                 | Example / Notes                                                   |
| -------------------------------------- | ----------------------------------------------------------------- |
| **Do you want to configure PCCS now?** | `Y`                                                               |
| **Set HTTPS listening port**           | `8081` (default)                                                  |
| **Accept local connections only?**     | `Y` (recommended if you don’t need remote access)                 |
| **Intel PCS API key**                  | Paste the key you generated on the Intel portal                   |
| **Caching fill method**                | `LAZY` is fine for most                                           |
| **Administrator password**             | Must meet complexity (uppercase, lowercase, number, special char) |
| **Server user password**               | Also needs complexity                                             |
| **Generate insecure HTTPS key/cert**   | `Y` if you want a self-signed certificate                         |


#### How to check successful PCCS setup?

You can verify PCCS is active and can reach the PCS with the command below:

```bash
curl -k -G "https://localhost:8081/sgx/certification/v4/rootcacrl"
```

If successful, the HEX-encoded DER representation of the Intel Root CA CRL will be
displayed:

```
308201213081c8020101300a06082a8648ce3d0403023068311a301806035504030c11496e74656c2053475820526f6f74204341311a3018060355040a0c11496e74656c20436f72706f726174696f6e3114301206035504070c0b53616e746120436c617261310b300906035504080c024341310b3009060355040613025553170d3233303430333130323235315a170d3234303430323130323235315aa02f302d300a0603551d140403020101301f0603551d2304183016801422650cd65a9d3489f383b49552bf501b392706ac300a06082a8648ce3d0403020348003045022051577d47d9fba157b65f1eb5f4657bbc5e56ccaf735a03f1b963d704805ab118022100939015ec1636e7eafa5f426c1e402647c673132b6850cabd68cef6bad7682a03
```

#### How to check service log of the PCCS?

You can check the service log of the PCCS with the following command:

```bash
sudo journalctl -u pccs
```

The PCCS should be running. Example output after PCCS start:

```
date time localhost systemd[1]: Started pccs.service - Provisioning Certificate Caching Service (PCCS).
date time localhost node[3305]: date time [info]: HTTPS Server is running on: https://localhost:8081
```

#### How to change the configuration of the PCCS?

If you need to make changes to the PCCS setup after installation, the default location of
the PCCS configuration file is `/opt/intel/sgx-dcap-pccs/config/default.json`. If changes
are made to the PCCS configuration file, you will need to restart the PCCS service using
the following command:

```bash
sudo systemctl restart pccs
```

### Platform Registration

On the host OS of platform to register, retrieve the PCKCIDRT:

From the package repository of your distribution of choice:

Set up the appropriate Intel SGX package repository:

```bash
echo 'deb [signed-by=/etc/apt/keyrings/intel-sgx-keyring.asc arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu noble main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list
wget https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key
sudo mkdir -p /etc/apt/keyrings
cat intel-sgx-deb.key | sudo tee /etc/apt/keyrings/intel-sgx-keyring.asc > /dev/null
sudo apt-get update
```

Install PCKCIDRT:

```bash
sudo apt install -y sgx-pck-id-retrieval-tool
```

#### Execute the PCKCIDRT

On the host OS of platform to register, execute the PCKCIDRT. This step depends on the
method used for PCKCIDRT retrieval in step 1:

If retrieved from a package repository:

```bash
cd /opt/intel/sgx-pck-id-retrieval-tool
sudo ./PCKIDRetrievalTool -f host_$(hostnamectl --static).csv
```

On successful execution of the PCKCIDRT, you'll see output similar to the following:

```
Intel(R) Software Guard Extensions PCK Cert ID Retrieval Tool Version 1.23.100.0

Registration status has been set to completed status.
<hostname>.csv has been generated successfully!
```

#### Extract the Platform Manifest

On the host OS of platform to register, use the following commands to extract the PM
from the `<hostname>.csv` and store the result in the file `platformmanifest.bin`:

```bash
sudo apt-get install -y csvtool
sudo bash -c "csvtool col 6 host_$(hostnamectl --static).csv | xxd -r -p > host_$(hostnamectl --static)_pm.bin"
```

#### Register with Intel Registration Service

On the Registration Platform, send the PM to the registration REST API endpoint of the
IRS. As shown in the linked API documentation, this can be done with a simple curl
command (after adjusting the hostname placeholder):

```bash
curl -i \
--data-binary @<hostname>-pm.bin \
-X POST "https://api.trustedservices.intel.com/sgx/registration/v1/platform" \
-H "Content-Type: application/octet-stream"
```

If the registration is successful, the IRS will return a "HTTP/1.1 201 Created" reply,
with the PPID of the registered platform as content. Sample response:

```
HTTP/1.1 201 Created
Content-Length: 32
Content-Type: text/plain
Request-ID: <request id>
Date: <date>

<PPID>
```

> **Note**: Platform registration can be done in other ways as well. For more details on
> alternative registration methods including Direct Registration and Indirect
> Registration, see the [Intel TDX Enabling Guide - Platform Registration](https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/02/infrastructure_setup/#platform-registration).

### Quote Generation Service (QGS)

The Quote Generation Service (QGS) is a service that runs in the host OS (or inside a
dedicated VM) to host the TD Quoting Enclave. Note that the QGS cannot run on another
machine, because the verification of the TD Report requires that the corresponding TD
and the TD Quoting Enclave run on the same machine.

#### Install QGS

If not done during another component installation, set up the appropriate Intel SGX
package repository for your distribution of choice:

```bash
echo 'deb [signed-by=/etc/apt/keyrings/intel-sgx-keyring.asc arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu noble main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list
wget https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key
sudo mkdir -p /etc/apt/keyrings
cat intel-sgx-deb.key | sudo tee /etc/apt/keyrings/intel-sgx-keyring.asc > /dev/null
sudo apt-get update
```

Install the QGS with the following command, which will also install the necessary
prerequisites (the Quote Provider Library (QPL) and the Quoting Library (QL)).

```bash
sudo apt install -y \
    tdx-qgs \
    libsgx-dcap-default-qpl \
    libsgx-dcap-ql
```

More detailed information about these instructions can be found in our Intel® SGX
Software Installation Guide For Linux* OS.

#### How to check service log of the QGS?

You can check the service log of the QGS with the following command:

```bash
sudo journalctl -u qgsd -f
```

#### Configure QCNL

On start, the QGS reads the configuration file `/etc/sgx_default_qcnl.conf`, and uses the
contained settings for TD Quote Generation. This file contains various settings that
might be important in your environment.

Selected highlights regarding this configuration file:

- If the QGS should accept insecure HTTPS certificates from the PCCS (as configured in
  previous step), set the JSON-key `use_secure_cert` in the configuration file to
  `false`.

See the comments of the configuration file `/etc/sgx_default_qcnl.conf` for more
information on other settings.

After changing settings in the file `/etc/sgx_default_qcnl.conf`, you have to restart
the QGS:

```bash
sudo systemctl restart qgsd.service
```
