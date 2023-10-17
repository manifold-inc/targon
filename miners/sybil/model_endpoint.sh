model=teknium/OpenHermes-2-Mistral-7B 
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --net=host --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id $model
