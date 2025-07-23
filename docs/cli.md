## How to Install

Run the command `just install-cli`:

```bash
just install-cli
```

This will install the CLI binary to your Go bin directory (typically `~/go/bin/`). Make sure this directory is in your PATH.

## How to Use

The CLI tool is accessed via the `targon-cli` command. On first run, it will prompt you to create a configuration file with your hotkey phrase.

```bash
targon-cli [command] [flags]
```

## Quick Start

1. **Install the CLI:**

   ```bash
   just install-cli
   ```

2. **First-time setup (creates config file):**

   ```bash
   targon-cli --help
   # This will prompt for hotkey phrase and create ~/.config/.targon.json
   ```

## Commands

### Main Commands

#### `targon-cli attest`

Manually attest a miner or IP address for GPU verification.

**Usage:**

```bash
targon-cli attest [flags]
```

**Flags:**

- `--ip string` - Specific IP address for off-chain testing
- `--uid int` - Specific UID to grab GPU info for

**Examples:**

```bash
# Attest a specific UID
targon-cli attest --uid 123

# Attest a specific IP address
targon-cli attest --ip http://192.168.1.100:8080

```

#### `targon-cli config`

Update configuration settings.

**Usage:**

```bash
targon-cli config [flags]
```

**Flags:**

- `--hotkey string` - Hotkey phrase to update to

**Examples:**

```bash
# Update hotkey phrase
targon-cli config --hotkey "your-hotkey-phrase-here"
```

#### `targon-cli get`

Fetch data from MongoDB or the blockchain and display it in various formats.

**Usage:**

```bash
targon-cli get [command]
```

**Subcommands:**

##### `targon-cli get errors`

Get attestation errors for a specific UID.

**Usage:**

```bash
targon-cli get errors [flags]
```

**Flags:**

- `--uid int` - Specific UID to grab GPU info for

**Examples:**

```bash
targon-cli get errors --uid 123
```

## Configuration

The CLI tool uses a JSON configuration file located at `~/.config/.targon.json`. This file stores your hotkey phrase and other settings.

**Configuration file structure:**

```json
{
  "HOTKEY_PHRASE": "your-hotkey-phrase-here"
}
```
