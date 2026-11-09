# Installing TargonOS

Releases live at [https://releases.targon.com](https://releases.targon.com), organized by version
(e.g. `/13.1.0/`). Current release: **13.1.0**.

> ⚠️ **Warning:** the installer **wipes every eligible disk** on the machine. Only run it on
> hardware you intend to dedicate to TargonOS.

## What you need

- A UEFI machine matching one of the [supported hardware configurations](miner.md#supported-hardware-configurations), with the TEE (Intel TDX or AMD SEV-SNP) enabled in BIOS (see [BIOS configuration](#bios-configuration)) and a TPM 2.0. The installer's hardware verification step rejects anything that doesn't match.
- Network connectivity with outbound HTTPS — DHCP by default, or a static address, bond, or VLAN via [`targon.net.*`](#network-configuration-targonnet). The installer fetches the signed OS payload from `releases.targon.com` and validates hardware against `tower.targon.com`; these endpoints are pinned inside the signed installer and cannot be pointed elsewhere.
- Your Bittensor miner hotkey as a BIP39 seed phrase (12 or 24 words, as emitted by `btcli`).
Only the derived SS58 public address is ever written to disk — the phrase itself is not stored.

## BIOS configuration

Set these before booting the installer — it refuses to install with the TEE disabled. Menu
names vary slightly between vendors; the settings below follow the common Supermicro/AMI layout.

### Intel TDX

```
CPU Configuration → Processor Configuration
  Limit CPU PA to 46 Bits                          → Disable

Intel TME, Intel TME-MT, Intel TDX
  Total Memory Encryption (Intel TME)              → Enable
  Total Memory Encryption (Intel TME) Bypass       → Auto
  Total Memory Encryption Multi-Tenant (TME-MT)    → Enable
  Memory Integrity                                 → Disable
  Intel TDX                                        → Enable
  TDX Secure Arbitration Mode Loader (SEAM)        → Enabled
  Disable excluding Mem below 1MB in CMR           → Auto
  Intel TDX Key Split                              → <non-zero value>

SGX
  Software Guard Extension                         → Enabled
  SGX Factory Reset                                → Enabled
```

### AMD SEV-SNP

```
Advanced → CPU Configuration
  SMEE                                             → Enabled
  SEV ASID Count                                   → 509 ASIDs
  SEV-ES ASID Space Limit Control                  → Manual
  SEV-ES ASID Space Limit                          → 100
  SNP Memory Coverage                              → Enabled

Advanced → NB Configuration
  IOMMU                                            → Enabled
  SEV-SNP Support                                  → Enabled
```

## Option 1 — iPXE netboot (unattended)

Best for fleets. Chainload the installer UKI and pass your hotkey on the kernel command line;
the install runs to completion with no console interaction.

Start from the reference script at
[`/13.1.0/netboot/targonos.ipxe.example`](https://releases.targon.com/13.1.0/netboot/targonos.ipxe.example):

```
#!ipxe
dhcp
chain http://releases.targon.com/13.1.0/netboot/targonos-installer.efi targon.install.unattended=1 targon.install.hotkey=word1-word2-...-word12
```

Replace `targon.install.hotkey` with your miner hotkey seed phrase, words separated by dashes
instead of spaces. Recognized arguments:


| Argument                            | Meaning                                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `targon.install.unattended=1`       | Run headlessly: take every eligible disk, skip all prompts. Network comes from `targon.net.*` if given, otherwise DHCP on the first NIC with internet. |
| `targon.install.hotkey=<w1-w2-...>` | Dash-separated BIP39 seed phrase the installer derives the hotkey from. Required when unattended.                                                 |
| `targon.net.ip=...`                 | Uplink device and IPv4 addressing (static or DHCP). See [below](#network-configuration-targonnet).                                                |
| `targon.net.bond=...`               | Optional bond over two or more NICs. Repeatable.                                                                                                  |
| `targon.net.vlan=...`               | Optional 802.1Q VLAN on a NIC or bond. Repeatable.                                                                                                |


The unattended install halts (rather than guessing) if the network probe fails, the TEE is
disabled in BIOS, the seed phrase is missing or invalid, or a `targon.net.*` config is invalid
or cannot be brought up.

### Network configuration (`targon.net.*`)

By default the installer takes DHCP on whichever NIC reaches the internet. If you need a static
address, a bond, or a VLAN, describe it with `targon.net.*` arguments. The configuration is
persisted to the installed system and re-applied on every boot. The syntax follows dracut /
`systemd-network-generator`:

```
targon.net.ip=<addr>/<prefix>::<gateway>:::<dev>:static:<dns1>[:<dns2>]
targon.net.ip=<dev>:dhcp
targon.net.bond=<name>:<mac>,<mac>[:mode=...,lacp_rate=...,xmit_hash_policy=...,miimon=...]
targon.net.vlan=<name>:<parent>[:<id>]
```

`<dev>` and `<parent>` are either a bond/VLAN name defined in another argument, or the **MAC
address** of a physical NIC (colon or dashed form). Interface names like `eth0` or `ens16f0np0`
are not accepted — the installer runs without udev, so kernel names don't match what the installed
OS or your inventory calls the port. Find MACs in the BMC's Redfish `EthernetInterfaces`, the
switch's LAG member table, `ip -br link` on the previous OS, or in iPXE via `ifstat` or
`${net0/mac}`, `${net1/mac}`, ...

Bond defaults are `mode=802.3ad lacp_rate=fast xmit_hash_policy=layer3+4 miimon=100`. The VLAN
id defaults to the trailing digits of its name (`vlan100` → 100).

Examples:

```
# Static IPv4 on one NIC
targon.net.ip=192.0.2.10/24::192.0.2.1:::02:00:00:aa:bb:01:static:1.1.1.1:8.8.8.8

# DHCP pinned to one NIC
targon.net.ip=02:00:00:aa:bb:01:dhcp

# LACP bond over two NICs, VLAN 100 on the bond, static address on the VLAN
targon.net.bond=bond0:02:00:00:aa:bb:01,02:00:00:aa:bb:02
targon.net.vlan=vlan100:bond0
targon.net.ip=198.51.100.20/28::198.51.100.17:::vlan100:static:1.1.1.1:8.8.8.8
```

In a fleet-wide iPXE script, let iPXE supply the MACs (indices follow PCI enumeration order and
are stable per hardware model):

```
chain http://releases.targon.com/13.1.0/netboot/targonos-installer.efi targon.install.unattended=1 targon.install.hotkey=... targon.net.bond=bond0:${net2/mac},${net3/mac} targon.net.vlan=vlan100:bond0 targon.net.ip=198.51.100.20/28::198.51.100.17:::vlan100:static:1.1.1.1:8.8.8.8
```

In an interactive (ISO) install these arguments pre-configure the network screen; if they fail,
the screen reports why and you can continue by hand.

### UEFI HTTP Boot without iPXE

If your DHCP server steers UEFI HTTP Boot clients directly, you can skip iPXE: the release also
ships an operator bundle (`targonos-13.1.0.netboot.tar.gz`) with the installer UKI, an example
dnsmasq config, and a README. Your boot server only serves the UKI — the OS payload always comes
from the pinned release server.

## Option 2 — ISO (interactive)

Best for one-off installs. Download
[`/13.1.0/targonos-13.1.0-amd64.iso`](https://releases.targon.com/13.1.0/targonos-13.1.0-amd64.iso),
write it to a USB stick or attach it via your BMC (below), and boot it in UEFI mode.

The installer walks you through network setup, disk selection, hardware verification, and
hotkey entry (typed as a 12/24-word grid; the derived SS58 address is shown before you
confirm). A final confirmation gates the point of no return.

### Attaching the ISO via BMC virtual media

Most BMCs (Supermicro, ASRock Rack, iDRAC, etc.) can mount an ISO straight from an HTTP server.
`releases.targon.com` is reachable by IP for BMCs without DNS:

- Server: `http://172.232.15.41`
- Image path: `/13.1.0/targonos-13.1.0-amd64.iso`

Mount it as virtual CD/DVD media, set the one-time boot device to the virtual CD (UEFI), and
reboot into the installer.

## Verifying artifacts

Each release directory ships a `SHA256SUMS` covering every file:

```
sha256sum -c SHA256SUMS
```

That is an integrity check for humans; the actual trust root is the per-artifact minisign
signature (`.minisig`), which the installer verifies against its baked-in release key before
writing anything to disk.

## After the install

The machine reboots into TargonOS. On every boot it attests to the Targon network, brings up
its encrypted storage, and starts the Targon services on its own. No further setup is required
on the host — it needs the same outbound HTTPS reachability as the install.