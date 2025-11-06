# AMD EPYC 4th Gen 9xx4 series SEV-SNP Setup

> ⚠️ **Important:** AMD currently supports **SecureAI solutions** out of the box starting with **Ubuntu 25.04 Server**. Ensure your system is installed and fully updated.

### Hardware Requirements (AMD)

* **Processor:** AMD EPYC™ 9xx4 Series (Genoa, Bergamo) with **SEV-SNP** support
* **Storage:** 1 TB

Verify your CPU details using the following command:

```bash
lscpu | grep -E 'Model name|Architecture|Vendor|Flags'
```
Key Things to Check

**1. Vendor ID:** Should display: `AuthenticAMD`

**2. Model Name:** Must indicate a 9xx4 series EPYC processor, for example: `EPYC 9354`, `EPYC 9454`, `EPYC 9654`, `EPYC 9754`.

**3. Flags:** Must include the following for SEV-SNP support:
  * `sev` → Secure Encrypted Virtualization
  * `sev_es` → Encrypted State
  * `sev_snp` → Secure Nested Paging (**required for SNP**)

---
### Software Requirements (AMD)
- **Host OS:** Ubuntu 25.04 Server
- **HGX Firmware Bundle:** Version 1.7.0 or higher (also known as Vulcan 1.7)


---

### BIOS Configuration (AMD SEV-SNP)

Enter your system **BIOS/UEFI** and configure the following settings:

```markdown
# Advanced → CPU Configuration
SMEE → Enabled  
SEV ASID Count → 509 ASIDs  
SEV-ES ASID Space Limit Control → Manual  
SEV-ES ASID Space Limit → 100  
SNP Memory Coverage → Enabled  

# Advanced → NB Configuration
IOMMU → Enabled  
SEV-SNP support → Enabled  
```
---

### Host OS Preparation (AMD)

**1. Check your Ubuntu version**

```bash
lsb_release -a
```

Verify that you have the correct version (**Ubuntu 25.04 “plucky”**):

```
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 25.04
Release:        25.04
Codename:       plucky
```

> If you're currently on Ubuntu 24.04 LTS, upgrade to 25.04 with:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y update-manager-core
sudo sed -i 's/^Prompt=.*/Prompt=normal/' /etc/update-manager/release-upgrades
sudo do-release-upgrade
# Reboot when prompted
```

After the upgrade, verify you're on 25.04:

```bash
lsb_release -a
```

**2. Update package lists and upgrade the system**
<!-- For the host, this will update packages and install QEMU along with required virtualization components. -->

Update package lists and upgrade installed packages
```bash
sudo apt update
sudo apt upgrade -y
```
Install QEMU along with required virtualization components
```bash
sudo apt install -y libvirt-daemon-system libvirt-clients libvirt-daemon
```
Verify libvirt installation
```bash
libvirtd --version
```
Reboot if required
```bash
sudo reboot
```

**3. Validating the Host Detects SEV-SNP**

**3.1. After the host reboots, check that your kernel is **SNP-aware** and the configuration options were correctly applied**

Check kernel version
```bash
uname -a
```
Example output:
```
Linux ubuntu-server 6.14.0-28-generic #28-Ubuntu SMP PREEMPT_DYNAMIC Wed Jul 23 12:05:14 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
```
> **⚠️ Important:**  
> Dates and hashes may vary. The key is to ensure your kernel is **6.14+**.

**3.2. Validate the kernel was configured with the proper Confidential Compute (CC) crypto options**


```bash
grep CONFIG_CRYPTO_EC /boot/config-$(uname -r)
```
Example output:
```
CONFIG_CRYPTO_ECC=y
CONFIG_CRYPTO_ECDH=y
CONFIG_CRYPTO_ECDSA=m
CONFIG_CRYPTO_ECRDSA=m
CONFIG_CRYPTO_ECB=y
CONFIG_CRYPTO_ECHAINIV=m
```

**3.3. Verifying SEV-SNP Detection***

Ensure that the kernel actually detects the **SEV-SNP processor**.  

> ⚠️ **Important:**  
> If you do not see the correct output below, please review the **Bios Configuration** section above, to verify the BIOS and hardware configuration.

Check SEV-SNP detection in kernel messages
```bash
sudo dmesg | grep -i -e rmp -e sev
```

**Expected output example:**
```
[ 0.000000] SEV-SNP: RMP table physical range [0x0000000088900000 - 0x00000000a8efffff]
[ 6.072556] ccp 0000:45:00.1: sev enabled
[ 6.195348] ccp 0000:45:00.1: SEV firmware updated from 1.49.3 to 1.55.21
[ 7.793012] ccp 0000:45:00.1: SEV API:1.55 build:21
[ 7.793024] ccp 0000:45:00.1: SEV-SNP API:1.55 build:21
[ 7.806923] kvm_amd: SEV enabled (ASIDs 100 - 509)
[ 7.806926] kvm_amd: SEV-ES enabled (ASIDs 1 - 99)
[ 7.806929] kvm_amd: SEV-SNP enabled (ASIDs 1 - 99)
```

> ✅ **Tip:**  
>Look for lines mentioning **SEV-SNP enabled** and the correct ASID ranges. This confirms your AMD EPYC v4 processor is correctly detected and SEV-SNP is active.

