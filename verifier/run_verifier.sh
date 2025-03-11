#!/bin/bash
TENSOR_PARALLEL=8 MODEL=$1 CONTEXT_LENGTH=128000 uvicorn verifier_sglang:app --port 8000 --host 0.0.0.0 
