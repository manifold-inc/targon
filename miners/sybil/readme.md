# Miners

to get started mining, first start your model endpoint. 

if you are using runpod, please create the model instance first.

you can do this by using a container image 'ghcr.io/huggingface/text-generation-inference:1.1.0' and then running the following command in the 'docker command': --port 8080 --model-id YOUR_HF_MODEL

for your port, specify 8080

when it is done spinning up, navigate to 'pods' tab and click 'connect'. You will see a card that has your port. click this, and that url is what you will use in --sybil.api_url


if you are not using runpod, then install pm2 and run the following command:

```bash
cd miners/sybil/tgi
pm2 start model_endpoint.sh
```
    
    then once this is running, start the miner. it will automatically connect at 0.0.0.0:8080



there is no more need to specify /generate in the api_url! this is done automatically.



