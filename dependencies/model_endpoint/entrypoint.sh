volume=$PWD/models
model=mlabonne/NeuralDaredevil-7B

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.3 --model-id $model --max-input-length 3072 --max-total-tokens 4096