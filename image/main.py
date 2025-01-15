import time
from enum import Enum
from fastapi import FastAPI
import base64
from io import BytesIO
import os
from huggingface_hub import HfApi
import diffusers
import torch

from pydantic import BaseModel

MODEL_NAME = os.getenv("MODEL", None)
if MODEL_NAME is None:
    print("Missing model name")
    exit()

print("loading api")
api = HfApi()
model = api.model_info(MODEL_NAME)
if model is None or model.config is None:
    "Cant find model"
    exit()


diffuser_class = model.config["diffusers"]["_class_name"]
diffuser = getattr(diffusers, diffuser_class)
print("loading from pretrained")
model = diffuser.from_pretrained(MODEL_NAME)
print("moving to cuda")
model.to("cuda")


app = FastAPI()


class Sizes(Enum):
    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"
    EXTRA_WIDE = "1792x1024"
    EXTRA_TALL = "1024x1792"


class ImageRequest(BaseModel):
    prompt: str
    model: str
    size: Sizes


@app.post("/v1/images/generations")
async def generate_question(req: ImageRequest):
    generator = torch.Generator(device="cuda").manual_seed(4)
    width, height = req.size.value.split("x")
    image = model(
        prompt=req.prompt, height=int(height), width=int(width), generator=generator
    )
    print(image)
    image = image.images[0]
    buffered = BytesIO()
    image.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    return {"created": time.time(), "data": [{"b64_json": img_str}]}


@app.get("/")
def ping():
    return "", 200


print("Starting fastapi")
