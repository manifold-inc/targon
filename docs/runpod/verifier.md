# SN4 RunPod Guide

## Step 1: Create TGI Instance

- **Manifold Template**: Use the provided template to run TGI on Runpod. [Template Link](#)
  - After login, create a new instance using the template.
  - **GPU Type**: Select A100 and click on "Deploy".
  - **Instance Page**: Monitor the status of your instance.

## Step 2: Set up Redis and Verifier

- **Pods Setup**: Navigate to Pods in the sidebar and select `+ CPU Pod`.
  - Customize the pod to include the verifier axon port.
  - Deploy your setup.

## Step 3: Ubuntu Pod Connection

1. Connect to the Ubuntu pod and execute the following commands:
```bash
apt update
apt install git
git clone https://github.com/manifold-inc/targon.git
cd targon/
apt install python3 python3-venv
python3 -m venv venv
source venv/bin/activate
python -m pip install -e .
apt install nano redis
./scripts/generate_redis_password.sh
```

2. **Redis Configuration**:
- Edit `/etc/redis/redis.conf`, find the line `# requirepass foobared`.
- Uncomment and set your password.
- Restart Redis server: `/etc/init.d/redis-server stop` then `/etc/init.d/redis-server start`.

3. **Node Version Manager and PM2 Setup**:
```bash
nvm install --lts && npm install pm2 -g
cd neurons/verifier/
pm2 start app.py --name verifier --wallet.name default --wallet.hotkey default --logging.debug --logging.trace --subtensor.chain_endpoint ws://xx.xx.xx.xx:9944 --database.password xxxxxxxx --neuron.tgi_endpoint https://xxxxxxx-80.proxy.runpod.net/
```

4. **TGI Endpoint Configuration**:
- Connect to the TARGON TGI - SN4 hearth.
- Use the "Connect to HTTP Service (Port 80)" option to get the URL for `--neuron.tgi_endpoint`.
