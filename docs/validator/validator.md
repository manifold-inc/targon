# Running a Validator

**Validator Prerequisites**

- Docker installed (see docs/docker.md)
- Basic compute resources (any cpu server)

## Validator Installation

**1. Setup Validator CPU Server**

Follow the steps in amd-cpus.md to setup your hardware for the validator virtual
machine. We reccomend [latitude](https://www.latitude.sh/) m4.metal.small boxes
that you can easily upgrade from 24.04 to 25.04 ubuntu

**2. Environment Setup**

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

## Netuid validator is running on. Useful for testnet
# NETUID=4

## Chain to connect to
# CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
```

> **Note** both mongo environment keys are keys **you define** and can be
> whatever you want. We suggest using the output of
> `tr -dc A-Za-z0-9 </dev/urandom | head -c 24; echo`. Run twice, and use one as
> the password and one as the username. These **do not** need to be double
> quoted in the .env file

**3. Pull validator VM**

Use `tvm/install` binary on the CPU server with the following arguments

- `--hotkey-phrase`: Your validator Hotkey Phrase
- `--node-type`: vali-cpu
- `--submit`: Actually submit and download image
- `--service-url`: http://tvm.targon.com
- `--vm-download-dir`: location you want to download vm (i.e /vms/manifold)

**4. Start the VM**

Go to the vm-download-dir, and into the unzipped folder that is newly created.
Run `sudo launch.sh` to start the vm.

To stop the vim, use `ps -aux | grep qemu` followed by `sudo kill -9 [puid]`,
where puid is the second column of the output row prefixed by `root`.

**5. Start the validator**

To initalize the validator, first install targon-cli (see cli docs for
installation) **on the server hosting the vm** and run
`targon-cli vali init [path to .env file]`. This will prompt the user to confirm
the passed environment file. This must be done on the server where the VM is
running, or you must pass --ip with the ip of the server. This will start the
validator with the environment variables passed.

## Validator Monitoring and Maintenance

**Logs**

To get logs, first you must get a list of the running containers using
`targon-cli vali containers`. Then, you can view the logs of any of the
containers using `targon-cli vali logs --container [container name]`.

**Updates**

To update the validator running in the VM, either repull and re-init the VM, or
run `targon-cli vali update`.
