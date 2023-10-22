git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .  # This may take 5-10 minutes.

pm2 start model_endpoint.sh