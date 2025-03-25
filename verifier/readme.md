to setup remote nodes, copy and install targon verifier on the server;

1. Clone https://github.com/manifold-inc/targon

1. `cd targon/verifier && pip install -r requirements.txt`

1. install sglang specific deps:
   `pip install "sglang[all]>=0.4.3.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python`

feel free to do this in a venv, you may need to modify the run_verifier script
to properly use the right venv

1. download the huggingface model, example for deepseek r1:
   `huggingface-cli download deepseek-ai/DeepSeek-R1 --max-workers 64`

1. install pm2 see targon readme for instructions

1. run the verifier script via pm2 with the model arg

`pm2 start run_verifier.sh -- deepseek-ai/DeepSeek-R1`

1. add this node to your config.json file where your targon validator is
   running. An example config file with two nodes, one for r1 and one for v3 is
   below

```json
{
  "verification_ports": {
    "deepseek-ai/DeepSeek-R1": {
      "port": 8000,
      "url": "http://0.0.0.0",
      "max_model_len": 12800,
      "endpoints": [
        "CHAT",
        "COMPLETION"
      ]
    },
    "deepseek-ai/DeepSeek-V3": {
      "port": 8000,
      "url": "http://0.0.0.0",
      "max_model_len": 128000,
      "endpoints": [
        "CHAT",
        "COMPLETION"
      ]
    }
  }
}
```

1. restart your targon validator. you should see it pick up these nodes on
   launch, and every ~10 blocks see logs saying they have validated responses
   from r1/v3
