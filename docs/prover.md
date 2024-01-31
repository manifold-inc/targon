# Run a Prover

## Role of a Prover
A prover is a node that is responsible for generating a output from a query, private input, and a deterministic sampling params. 

## Get Started
To get started running a prover, you will need to be running the docker containers for the requirements of the prover. To do this, run the following command:
```bash
cp neurons/prover/docker-compose.example.yml neurons/prover/docker-compose.yml
docker compose -f neurons/prover/docker-compose.yml up -d
```

this includes the following containers:
- TGI Inference Node
- Subtensor
- Prover (optional)


**experimental** optionally, you can edit the docker-compose.yml file to include the proving container, but you will need to edit the docker-compose.yml file and uncomment out the prover container. Otherwise you can run the prover with PM2.


#### Docker

<details>
<summary>Run with Docker</summary>

```docker
  prover:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8091:8091"
    command: ./entrypoint.sh
    volumes:
      - ~/.bittensor/wallets:/root/.bittensor/wallets

```
and then edit the entrypoint.sh file to include the args specific for your prover.

```bash
python app.py --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOTKEY --logging.debug --logging.trace --subtensor.chain_endpoint 0.0.0.0:9944
```

replace the wallet name and wallet hotkey with your wallet name and wallet hotkey. You can also change the subtensor chain endpoint to your own chain endpoint if you perfer.

</details>

#### PM2

<details>
<summary> Run with PM2</summary>
```bash
cd neurons/prover
pm2 start app.py --name prover -- --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOTKEY --logging.debug --logging.trace --subtensor.chain_endpoint 0.0.0.0:9944
```

replace the wallet name and wallet hotkey with your wallet name and wallet hotkey. You can also change the subtensor chain endpoint to your own chain endpoint if you perfer.

</details>

#### Options
The add_prover_args function in the targon/utils/config.py file is used to add command-line arguments specific to the prover. Here are the options it provides:

1. --neuron.name: This is a string argument that specifies the name of the neuron. The default value is 'prover'.

2. --blacklist.force_verifier_permit: This is a boolean argument that, if set, forces incoming requests to have a permit. The default value is False.

3. --blacklist.allow_non_registered: This is a boolean argument that, if set, allows provers to accept queries from non-registered entities. This is considered dangerous and its default value is False.

4. --neuron.tgi_endpoint: This is a string argument that specifies the endpoint to use for the TGI client. The default value is "http://0.0.0.0:8080".

