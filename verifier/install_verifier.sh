#!/bin/bash
git clone https://github.com/manifold-inc/targon.git
cd targon
pip install uv
cd verifier
uv pip install -r requirements.txt
uv pip install  "sglang[all]>=0.4.3.post4 sgl-kernel"
npm i -g pm2
pm2 start run_verifier.sh
