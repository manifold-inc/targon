# Installing TargonOS

Releases live at [https://releases.targon.com](https://releases.targon.com), organized by version
(e.g. `/0.1.0/`). Current release: **0.1.0**.

> ⚠️ **Warning:** the installer **wipes every eligible disk** on the machine. Only run it on
> hardware you intend to dedicate to TargonOS.

## What you need

- A UEFI machine with the TEE (Intel TDX) enabled in BIOS (see
[BIOS configuration](#bios-configuration)), a TPM 2.0, and GPUs. The machine must match one of
the [supported hardware configurations](miner.md#supported-hardware-configurations) — the
installer's hardware verification step rejects anything that doesn't.
- Network with DHCP and outbound HTTPS — the installer fetches the signed OS payload from
`releases.targon.com` and validates hardware against `tower.targon.com`. These endpoints are
pinned inside the signed installer and cannot be pointed elsewhere.
- Your Bittensor miner hotkey as a BIP39 seed phrase (12 or 24 words, as emitted by `btcli`).
Only the derived SS58 public address is ever stored.

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

## Option 1 — iPXE netboot (unattended)

Best for fleets. Chainload the installer UKI and pass your hotkey on the kernel command line;
the install runs to completion with no console interaction.

Start from the reference script at
`[/0.1.0/netboot/targonos.ipxe.example](https://releases.targon.com/0.1.0/netboot/targonos.ipxe.example)`:

```
#!ipxe
dhcp
chain http://releases.targon.com/0.1.0/netboot/targonos-installer.efi targon.install.unattended=1 targon.install.hotkey=word1-word2-...-word12
```

Replace `targon.install.hotkey` with your miner hotkey seed phrase, words separated by dashes
instead of spaces. These are the only two recognized arguments:


| Argument                            | Meaning                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| `targon.install.unattended=1`       | Run headlessly: auto-configure network via DHCP, take every eligible disk, skip all prompts.      |
| `targon.install.hotkey=<w1-w2-...>` | Dash-separated BIP39 seed phrase the installer derives the hotkey from. Required when unattended. |


The unattended install halts (rather than guessing) if the network probe fails, the TEE is
disabled in BIOS, or the seed phrase is missing or invalid.

If your DHCP server steers UEFI HTTP Boot clients directly, you can skip iPXE entirely: the
release also ships an operator bundle (`targonos-0.1.0.netboot.tar.gz`) containing the installer
UKI, an example dnsmasq config, and a README. Your boot server only serves the UKI — the OS
payload always comes from the pinned release server.

## Option 2 — ISO (interactive)

Best for one-off installs. Attach the ISO
(`[/0.1.0/targonos-0.1.0-amd64.iso](https://releases.targon.com/0.1.0/targonos-0.1.0-amd64.iso)`)
via your BMC's virtual media and boot it in UEFI mode.

### Attaching the ISO via BMC virtual media

Most BMCs (Supermicro, ASRock Rack, iDRAC, etc.) can mount an ISO straight from an HTTP file
server — no download needed. `releases.targon.com` is reachable by IP for BMCs without DNS:

- Server: `http://172.232.15.41`
- Image path: `/0.1.0/targonos-0.1.0-amd64.iso`

Mount it as virtual CD/DVD media, set the one-time boot device to the virtual CD (UEFI), and
reboot into the installer.

### Running the installer

The installer walks you through network setup, disk selection, hardware verification, and
hotkey entry (typed as a 12/24-word grid; the derived SS58 address is shown before you
confirm). A final confirmation gates the point of no return.

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
on the host — it needs the same network conditions as the install: DHCP and outbound HTTPS.