# Intel TDX + Hopper GPU Setup

### Hardware Requirements (Intel)

- NVIDIA H100 or H200 GPU with Confidential Compute support
- 3 TB Storage
- Intel CPU Requirements:
  - 5th Gen Intel® Xeon® Scalable Processor
  - Intel® Xeon® 6 Processors

### Software Requirements (Intel)

- Ubuntu 22.04 LTS or later
- HGX FW Bundle 1.7 (Known as Vulcan 1.7)

## BIOS Configuration (Intel TDX)

The following BIOS settings must be configured correctly for Confidential
Compute to work:

1. **CPU Configuration**

   - Navigate to: `CPU Configuration` → `Processor Configuration`
   - Set `Limit CPU PA to 46 Bits` to `Disable`

1. **Intel TME, Intel TME-MT, Intel TDX Settings**

   - Navigate to: `Intel TME, Intel TME-MT, Intel TDX`
   - Configure the following:
     - `Total Memory Encryption (Intel TME)` → `Enable`
     - `Total Memory Encryption (Intel TME) Bypass` → `Auto`
     - `Total Memory Encryption Multi-Tenant (Intel TME-MT)` → `Enable`
     - `Memory Integrity` → `Disable`
     - `Intel TDX` → `Disable`
     - `TDX Secure Arbitration Mode Loader (SEAM)` → `Enabled`
     - `Disable excluding Mem below 1MB in CMR` → `Auto`
     - `Intel TDX Key Split` → Set to a non-zero value

1. **Software Guard Extension**

   - Navigate to: `Software Guard Extension`
   - Set to `Enable`

1. **SGX Factory Reset**

  - Navidate to: `SGX Factory Reset`
  - Set to `Enable`

> ⚠️ **Note**: These BIOS settings are critical for Confidential Compute
> functionality. Incorrect settings may prevent the system from booting or cause
> security features to fail.

## Host OS Preparation (Intel)
For TD Quote Generation and TD Quote Verification, collateral is needed. Intel provides the necessary collateral through the Intel® Provisioning Certification Service for ECDSA Attestation (PCS). 

**Provisioning Certificate Caching Service (PCCS)**

To setup the PCCS in the next step, you need a subscription key for the Intel PCS. You can obtain this from the [Intel Provisioning Certification Service](https://api.portal.trustedservices.intel.com/provisioning-certification).

1. If you did not request such a subscription key before, [subscribe to Intel PCS](https://api.portal.trustedservices.intel.com/products#product=liv-intel-software-guard-extensions-provisioning-certification-service), which requires to log in (or to create an account). Two subscription keys are generated (for key rotation) and both can be used for the following steps. Click on Subscribe at botton of the page. Then click on show for Primary Key and use that key.
<img width="3456" height="1984" alt="image" src="https://github.com/user-attachments/assets/2311bc63-ceab-49c7-9ccf-784e096f29a5" />

3. If you did request such a subscription key before, [retrieve one of your keys](https://api.portal.trustedservices.intel.com/manage-subscriptions), which requires to log in. You have two subscription keys (for key rotation), and both can be used for the following steps.

If not done during another component installation, set up the appropriate Intel SGX package repository for your distribution of choice:

```bash
echo 'deb [signed-by=/etc/apt/keyrings/intel-sgx-keyring.asc arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu noble main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list
wget https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key
sudo mkdir -p /etc/apt/keyrings
cat intel-sgx-deb.key | sudo tee /etc/apt/keyrings/intel-sgx-keyring.asc > /dev/null
sudo apt-get update
```

Install PCCS with following commands, the installer will prompt you for the following configs. Answer the remaining questions according to your needs, e.g., your proxy settings, a desired user password, and an admin password. The configuration step will also allow you to create a self-signed SSL certificate for the PCCS.

| Prompt                                 | Example / Notes                                                   |
| -------------------------------------- | ----------------------------------------------------------------- |
| **Do you want to configure PCCS now?** | `Y`                                                               |
| **Set HTTPS listening port**           | `8081` (default) |
| **Accept local connections only?**     | `Y` (recommended if you don’t need remote access)                 |
| **Intel PCS API key**                  | Paste the key you generated on the Intel portal                   |
| **Caching fill method**                | `LAZY` is fine for most                                           |
| **Administrator password**             | Must meet complexity (uppercase, lowercase, number, special char) |
| **Server user password**               | Also needs complexity                                             |
| **Generate insecure HTTPS key/cert**   | `Y` if you want a self-signed certificate                         |


```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -yq --no-install-recommends nodejs=20.11.1-1nodesource1
sudo apt-get install -y cracklib-runtime
sudo apt install -y --no-install-recommends sgx-dcap-pccs
```

**How to check successful PCCS setup?**

You can verify PCCS is active and can reach the PCS with the command below:

```bash
curl -k -G "https://localhost:8081/sgx/certification/v4/rootcacrl"
```

If successful, the HEX-encoded DER representation of the Intel Root CA CRL will be displayed:

```
308201213081c8020101300a06082a8648ce3d0403023068311a301806035504030c11496e74656c2053475820526f6f74204341311a3018060355040a0c11496e74656c20436f72706f726174696f6e3114301206035504070c0b53616e746120436c617261310b300906035504080c024341310b3009060355040613025553170d3233303430333130323235315a170d3234303430323130323235315aa02f302d300a0603551d140403020101301f0603551d2304183016801422650cd65a9d3489f383b49552bf501b392706ac300a06082a8648ce3d0403020348003045022051577d47d9fba157b65f1eb5f4657bbc5e56ccaf735a03f1b963d704805ab118022100939015ec1636e7eafa5f426c1e402647c673132b6850cabd68cef6bad7682a03
```

**How to check service log of the PCCS?**

You can check the service log of the PCCS with the following command:

```bash
sudo journalctl -u pccs
```

The PCCS should be running. Example output after PCCS start:

```
date time localhost systemd[1]: Started pccs.service - Provisioning Certificate Caching Service (PCCS).
date time localhost node[3305]: date time [info]: HTTPS Server is running on: https://localhost:8081
```

**How to change the configuration of the PCCS?**

If you need to make changes to the PCCS setup after installation, the default location of the PCCS configuration file is `/opt/intel/sgx-dcap-pccs/config/default.json`. If changes are made to the PCCS configuration file, you will need to restart the PCCS service using the following command:


```bash
sudo systemctl restart pccs
```

**Platform Registration**


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

On the host OS of platform to register, execute the PCKCIDRT. This step depends on the method used for PCKCIDRT retrieval in step 1:

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

On the host OS of platform to register, use the following commands to extract the PM from the `<hostname>.csv` and store the result in the file `platformmanifest.bin`:

```bash
sudo apt-get install -y csvtool
sudo bash -c "csvtool col 6 host_$(hostnamectl --static).csv | xxd -r -p > host_$(hostnamectl --static)_pm.bin"
```

#### Register with Intel Registration Service

On the Registration Platform, send the PM to the registration REST API endpoint of the IRS. As shown in the linked API documentation, this can be done with a simple curl command (after adjusting the hostname placeholder):

```bash
curl -i \
--data-binary @<hostname>-pm.bin \
-X POST "https://api.trustedservices.intel.com/sgx/registration/v1/platform" \
-H "Content-Type: application/octet-stream"
```

If the registration is successful, the IRS will return a "HTTP/1.1 201 Created" reply, with the PPID of the registered platform as content. Sample response:

```
HTTP/1.1 201 Created
Content-Length: 32
Content-Type: text/plain
Request-ID: <request id>
Date: <date>

<PPID>
```

> **Note**: Platform registration can be done in other ways as well. For more details on alternative registration methods including Direct Registration and Indirect Registration, see the [Intel TDX Enabling Guide - Platform Registration](https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/02/infrastructure_setup/#platform-registration).

**Quote Generation Service (QGS)**

The Quote Generation Service (QGS) is a service that runs in the host OS (or inside a dedicated VM) to host the TD Quoting Enclave. Note that the QGS cannot run on another machine, because the verification of the TD Report requires that the corresponding TD and the TD Quoting Enclave run on the same machine.

#### Install QGS

If not done during another component installation, set up the appropriate Intel SGX package repository for your distribution of choice:

```bash
echo 'deb [signed-by=/etc/apt/keyrings/intel-sgx-keyring.asc arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu noble main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list
wget https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key
sudo mkdir -p /etc/apt/keyrings
cat intel-sgx-deb.key | sudo tee /etc/apt/keyrings/intel-sgx-keyring.asc > /dev/null
sudo apt-get update
```

Install the QGS with the following command, which will also install the necessary prerequisites (the Quote Provider Library (QPL) and the Quoting Library (QL)).

```bash
sudo apt install -y \
    tdx-qgs \
    libsgx-dcap-default-qpl \
    libsgx-dcap-ql
```

More detailed information about these instructions can be found in our Intel® SGX Software Installation Guide For Linux* OS.

#### How to check service log of the QGS?

You can check the service log of the QGS with the following command:

```bash
sudo journalctl -u qgsd -f
```

#### Configure QCNL

On start, the QGS reads the configuration file `/etc/sgx_default_qcnl.conf`, and uses the contained settings for TD Quote Generation. This file contains various settings that might be important in your environment.

Selected highlights regarding this configuration file:

- If the QGS should accept insecure HTTPS certificates from the PCCS (as configured in previous step), set the JSON-key `use_secure_cert` in the configuration file to `false`.

See the comments of the configuration file `/etc/sgx_default_qcnl.conf` for more information on other settings.

After changing settings in the file `/etc/sgx_default_qcnl.conf`, you have to restart the QGS:

```bash
sudo systemctl restart qgsd.service
```

**Install Prerequisite Packages**

```bash
# Update package lists
sudo apt update

# Install required packages
sudo apt install build-essential libncurses-dev bison flex libssl-dev libelf-dev \
debhelper-compat=12 meson ninja-build libglib2.0-dev python3-pip nasm iasl
```

**What these packages do:**

- `build-essential`: Basic build tools and libraries
- `libncurses-dev`: Terminal handling library
- `bison` & `flex`: Parser generators
- `libssl-dev`: SSL/TLS development files
- `libelf-dev`: ELF file format library
- `debhelper-compat`: Debian packaging helper
- `meson` & `ninja-build`: Build system tools
- `libglib2.0-dev`: Core library
- `python3-pip`: Python package manager
- `nasm` & `iasl`: Assembly and ACPI tools

**Verification:**

```bash
# Verify all packages are installed
dpkg -l | grep -E 'build-essential|libncurses-dev|bison|flex|libssl-dev|libelf-dev|debhelper-compat|meson|ninja-build|libglib2.0-dev|python3-pip|nasm|iasl'
```

**Setup Working Directory**

```bash
# Create and set permissions for shared directory
sudo mkdir /shared
cd /shared/
sudo chmod -R 777 /shared
```

**What this does:**

- Creates a central directory for all Confidential Compute related files
- Sets full read/write/execute permissions for all users
- Provides a consistent working environment

**Verification:**

```bash
# Verify directory exists and has correct permissions
ls -la /shared
# Should show: drwxrwxrwx
```

**Download and Patch GitHub Packages**

```bash
# Clone required repositories
git clone https://github.com/NVIDIA/nvtrust.git
git clone https://github.com/intel/tdx-linux.git

# Setup Intel's patches
cd tdx-linux
git checkout -b device-passthrough 1323f7b1ddf81076e3fcda6385c0c0dcf506258c

# Clone specific Linux Kernel branch
git clone -b kvm-coco-queue-20240512 https://git.kernel.org/pub/scm/linux/kernel/git/vishal/kvm.git

# Setup QEMU
git clone https://gitlab.com/qemu-project/qemu
cd qemu
git checkout -b hcc-h100 ff6d8490e33acf44ed8afd549e203a42d6f813b5
cd ..

# Clone OVMF
git clone -b edk2-stable202408.01 https://github.com/tianocore/edk2

# Patch the kernel
cd /shared/tdx-linux/kvm
cp ../tdx-kvm/tdx_kvm_baseline_698ca1e40357.mbox .
git am --empty=drop tdx_kvm_baseline_698ca1e40357.mbox

# Patch QEMU
cd /shared/tdx-linux/qemu
cp ../tdx-qemu/tdx_qemu_baseline_900536d3e9.mbox .
git am --empty=drop tdx_qemu_baseline_900536d3e9.mbox
```

**What each repository does:**

- `nvtrust`: NVIDIA's Trusted Computing framework
- `tdx-linux`: Intel's TDX patches for Linux
- `kvm`: Kernel-based Virtual Machine with Confidential Compute support
- `qemu`: Virtual machine emulator with H100 support
- `edk2`: UEFI firmware implementation

**Verification:**

```bash
# Verify all repositories are cloned
ls -la /shared

# Verify kernel patches
cd /shared/tdx-linux/kvm
git log --oneline | head -n 5

# Verify QEMU patches
cd /shared/tdx-linux/qemu
git log --oneline | head -n 5
```

**Build the Kernel**

```bash
# Navigate to kernel directory
cd /shared/tdx-linux/kvm

# Copy current kernel config
cp /boot/config-$(uname -r) .config

# Disable unnecessary features
scripts/config -d KEXEC \
-d KEXEC_FILE \
-d SYSTEM_TRUSTED_KEYS \
-d SYSTEM_REVOCATION_KEYS

# Enable required features
scripts/config -e KVM \
-e KVM_INTEL \
-e KVM_TDX_GUEST_DRIVER \
-e HYPERV \
-e INTEL_TDX_HOST \
-e CRYPTO_ECC \
-e CRYPTO_ECDH \
-e CRYPTO_ECDSA \
-e CRYPTO_ECRDSA

# Configure kernel
make oldconfig
# Press and hold "enter" when prompted for new features

# Build kernel and modules
make -j$(nproc)
make modules -j$(nproc)
```

**What each feature does:**

- Disabled features:
  - `KEXEC`: Kernel execution mechanism (not needed for CC)
  - `SYSTEM_TRUSTED_KEYS`: System key management (handled by CC)
- Enabled features:
  - `KVM` & `KVM_INTEL`: Virtualization support
  - `KVM_TDX_GUEST_DRIVER`: TDX guest support
  - `HYPERV`: Hyper-V compatibility
  - `INTEL_TDX_HOST`: TDX host support
  - `CRYPTO_*`: Required cryptographic features

**Verification:**

```bash
# Verify kernel configuration
grep -E "KVM|TDX|CRYPTO" .config

# Verify build artifacts
ls -la arch/x86/boot/bzImage
ls -la modules.builtin
```

**Install and Configure Host OS**

```bash
# Install kernel modules
sudo make modules_install
sudo make install

# Configure TDX module
sudo sh -c "echo options kvm_intel tdx=on > /etc/modprobe.d/tdx.conf"

# Configure GRUB
# Edit /etc/default/grub and modify GRUB_CMDLINE_LINUX_DEFAULT:
sudo vim /etc/default/grub
# Modify GRUB_CMDLINE_LINUX_DEFAULT TO:
GRUB_CMDLINE_LINUX_DEFAULT="nohibernate kvm_intel.tdx=on intel_iommu=on iommu=pt"

# Update GRUB configuration
sudo update-grub
```

**What this does:**

- Installs the newly built kernel modules
- Enables TDX support in the kernel module
- Configures GRUB with optimal settings for Confidential Compute
- Disables hibernation for better stability
- Enables IOMMU for proper device isolation

**Verification:**

```bash
# Verify module installation
ls -la /lib/modules/$(uname -r)/kernel/drivers/kvm/

# Verify TDX configuration
cat /etc/modprobe.d/tdx.conf

# Verify GRUB configuration
grep "GRUB_CMDLINE_LINUX_DEFAULT" /etc/default/grub
```

**Build QEMU**

```bash
# Navigate to QEMU directory
cd /shared/tdx-linux/qemu

# Install libslirp for network support
git clone -b v4.8.0 https://gitlab.freedesktop.org/slirp/libslirp.git
cd libslirp
meson build
sudo ninja -C build install
cd ..

# Ensure libslirp is in the ldconfig path
sudo ln -s /usr/local/lib/x86_64-linux-gnu/libslirp.so.0 /lib/x86_64-linux-gnu/

# Build and install QEMU
./configure --enable-slirp --enable-kvm --target-list=x86_64-softmmu
make -j$(nproc)
sudo make install
```

**What this does:**

- Installs libslirp for network support in CVMs
- Configures QEMU with necessary features
- Builds QEMU with KVM and network support
- Installs QEMU system-wide

**Verification:**

```bash
# Verify libslirp installation
ls -l /lib/x86_64-linux-gnu/libslirp.so.0

# Verify QEMU installation
qemu-system-x86_64 --version
```

**Build OVMF**

```bash
# Navigate to EDK2 directory
cd /shared/tdx-linux/edk2

# Initialize submodules
git submodule update --init

# Create build script
cat << 'EOF' > build_ovmf.sh
#!/bin/bash
rm -rf Build
make -C BaseTools
. edksetup.sh
cat <<-EOF > Conf/target.txt
ACTIVE_PLATFORM = OvmfPkg/OvmfPkgX64.dsc
TARGET = DEBUG
TARGET_ARCH = X64
TOOL_CHAIN_CONF = Conf/tools_def.txt
TOOL_CHAIN_TAG = GCC5
BUILD_RULE_CONF = Conf/build_rule.txt
MAX_CONCURRENT_THREAD_NUMBER = $(nproc)
EOF
build clean
build
if [ ! -f Build/OvmfX64/DEBUG_GCC5/FV/OVMF.fd ]; then
  echo "Build failed, OVMF.fd not found"
  exit 1
fi
cp Build/OvmfX64/DEBUG_GCC5/FV/OVMF.fd ./OVMF.fd
EOF

# Make script executable
chmod +x build_ovmf.sh

# Execute the build script
./build_ovmf.sh

# Reboot the host to apply all changes
sudo reboot
```

> ⚠️ **IMPORTANT WARNING** ⚠️
>
> Ubuntu 24.04's kernel may not boot if TDX is pre-enabled in the BIOS/UEFI.
> After rebooting, verify you're running kernel 6.9.0-rc7+:
>
> ```bash
> uname -r
> 6.9.0-rc7+
> ```

**Enable TDX in BIOS**

At this point, you need to adjust the BIOS settings for Intel TDX. **Intel TME,
Intel TME-MT, Intel TDX Settings**

- Navigate to: `Intel TME, Intel TME-MT, Intel TDX`
- Configure the following:
  - `Intel TDX` → `Enable`
- Check that the following are correct:
  - `Total Memory Encryption (Intel TME)` → `Enable`
  - `Total Memory Encryption (Intel TME) Bypass` → `Auto`
  - `Total Memory Encryption Multi-Tenant (Intel TME-MT)` → `Enable`
  - `Memory Integrity` → `Disable`
  - `TDX Secure Arbitration Mode Loader (SEAM)` → `Enabled`
  - `Disable excluding Mem below 1MB in CMR` → `Auto`
  - `Intel TDX Key Split` → Set to a non-zero value

After configuring BIOS settings:

```bash
# Reboot the system
sudo reboot

# After reboot, verify kernel version
uname -r
# Should show: 6.9.0

# Verify TDX is enabled
sudo dmesg | grep -i tdx
# You MUST see this line for TDX to be properly enabled:
# [ 21.364890] virt/tdx: module initialized

# If you don't see the "module initialized" message, TDX is not properly enabled.
# Double-check your BIOS settings and try again.
```

> **Note**: If you see errors like
> `SEAMCALL (0x0000000000000022) failed: 0xc0000c0000000000`, these may be
> ignored for this release. This error occurs if you do not have the latest
> TDX-Module installed. To update the TDX-Firmware, run the following commands:
>
> ```bash
> # Download and extract TDX-Module
> wget https://github.com/intel/tdx-module/releases/latest/download/intel_tdx_module.tar.gz -O intel_tdx_module.tar.gz
> tar -xvzf intel_tdx_module.tar.gz
>
> # Install TDX-Module
> sudo mkdir -p /boot/efi/EFI/TDX/
> sudo cp TDX-Module/intel_tdx_module.so /boot/efi/EFI/TDX/TDX-SEAM.so
> sudo cp TDX-Module/intel_tdx_module.so.sigstruct /boot/efi/EFI/TDX/TDX-SEAM.so.sigstruct
>
> # Reboot to apply changes
> sudo reboot
> ```

**Configure GPU for Confidential Compute**

The NVIDIA H100 can be toggled into and out of CC modes only with a privileged
call from the host. Here are the main flags:

- `--query-cc-settings`: Prints the current mode that the GPU is operating in
- `--set-cc-mode mode`: Where mode is one of the following:
  - `on`
  - `off`
  - `devtools`

To configure the GPUs:

```bash
# Setup NVIDIA Trust Tools
cd /shared/nvtrust
git submodule update --init
cd host_tools/python

# Note: The following errors can be safely ignored:
# 2025-02-26,22:12:11.043 ERROR Configuring CC not supported on NvSwitch 0000:07:00.0
# NVSwitch_gen3 0x22a3 BAR0 0xa6000000

# Set GPUs into Protected PCIe (Multi-GPU) Mode
# First, ensure all NVIDIA devices are not in CC mode
for i in $(seq 0 $(($(lspci -nn | grep -c "10de") - 1))); do 
  sudo python3 ./nvidia_gpu_tools.py --set-cc-mode=off --reset-after-cc-mode-switch --gpu=$i
done

# Then, set the GPUs and NVSwitches into Protected PCIe mode
for i in $(seq 0 $(($(lspci -nn | grep -c "10de") - 1))); do 
  sudo python3 ./nvidia_gpu_tools.py --set-ppcie-mode=on --reset-after-ppcie-mode-switch --gpu=$i
done
```

> **Warning**: You must complete the GPU configuration step for every GPU and
> switch you pass into the CVM. This configuration is persistent across reboots
> and power cycles. To revert these changes, run the previous commands again
> with `--set-<mode>-mode=off`.

