## Running a Miner

### Miner Prerequisites

> ⚠️ **IMPORTANT WARNING** ⚠️ It is **HIGHLY RECOMMENDED** that you have BIOS
> access to your machines. You will be adjusting the configurations throughout
> the process and delays from providers are expected. Without BIOS access, you
> may face significant delays or be unable to complete the setup process.

#### Hardware Requirements

- NVIDIA H100 or H200 GPU with Confidential Compute support
- 3 TB Storage
- Intel CPU Requirements:
  - 5th Gen Intel® Xeon® Scalable Processor
  - Intel® Xeon® 6 Processors

#### Software Requirements

- Ubuntu 22.04 LTS or later
- HGX FW Bundle 1.7 (Known as Vulcan 1.7)

#### BIOS Configuration

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

> ⚠️ **Note**: These BIOS settings are critical for Confidential Compute
> functionality. Incorrect settings may prevent the system from booting or cause
> security features to fail.

### Preparing the Host

#### Install Prerequisite Packages

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

#### Setup Working Directory

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

#### Download and Patch GitHub Packages

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

#### Build the Kernel

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

#### Install and Configure Host OS

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

#### Build QEMU

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

#### Build OVMF

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
> After rebooting, verify you're running kernel 6.0.9:
>
> ```bash
> uname -r
> 6.0.9
> ```

#### Enable TDX in BIOS

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
# Should show: 6.0.9

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

#### Configure GPU for Confidential Compute

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

### Launching TVM

After completing all the prerequisite steps above, you are ready to run TVM.
This process will:

1. Attest your hardware configuration
1. Execute the launch script for TVM
1. Verify the Confidential Compute environment

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
     Manifold Validator hotkey 5Hp18g9P8hLGKp9W3ZDr4bvJwba6b6bY3P2u3VdYf8yMR8FM)

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
- Your GPU is properly configured for Confidential Compute
- Your keys are correctly formatted and valid

At this point, all setup on your TVM nodes is complete.

### Updating/Running Miner

After setting up your TVM nodes, you need to update your miner configuration to
report the IP addresses of each CVM you are running.

1. **Update/Create the Configuration File** Edit `config.json` file to include
   the IP addresses of your TVM nodes. Add them to the list of endpoints that
   your miner reports to the network, along with any other desired paramaters.

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
    "netuid": 4
    }
   ```

> **Note**: Make sure to keep your TVM node IPs up to date. If you add or remove
> TVM nodes, update this configuration accordingly.

1. **Start Miner** Run
   `docker compose -f docker-compose.miner.yml up -d --build`
