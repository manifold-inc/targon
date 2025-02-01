import os
import asyncio
from fastapi import FastAPI
from fastapi.routing import APIRouter
from vllm import LLM
from huggingface_hub import HfApi
import importlib
from diffusers import DiffusionPipeline
from xgboost import XGBClassifier

from image import generate_image_functions
from llm import get_llm_functions

# Load the model.
XGB_MODEL_PATH = os.getenv("XGB_MODEL_PATH", 'xgb_model.json')
MODEL_NAME = os.getenv("MODEL_NAME", None)
if MODEL_NAME is None:
    exit()

api = HfApi()
model = api.model_info(MODEL_NAME)

if model is None or model.config is None:
    exit()

# Lock to ensure atomicity.
LOCK = asyncio.Lock()
LOCK_GENERATE = asyncio.Lock()

ENDPOINTS = []

app = FastAPI()
router = APIRouter()

match model.pipeline_tag:
    case "text-to-image":
        ENDPOINTS.append("image")
        # diffuser_class = model.config["diffusers"]["_class_name"]
        # diffuser = importlib.import_module(f"diffusers.{diffuser_class}")
        MODEL_WRAPPER = DiffusionPipeline.from_pretrained(MODEL_NAME)
        MODEL_WRAPPER.to("cuda")

        # create xgb wrapper
        xgb_model = XGBClassifier()
        xgb_model.load_model(XGB_MODEL_PATH)
        
        verify = generate_image_functions(MODEL_WRAPPER, MODEL_NAME, ENDPOINTS, xgb_model)
        
        router.add_api_route("/image/verify", verify, methods=["POST"])
    case "text-generation":
        TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL", 1))
        ENDPOINTS.append("completion")
        MODEL_WRAPPER = LLM(
            model=MODEL_NAME,
            enforce_eager=True,
            gpu_memory_utilization=1,
            tensor_parallel_size=TENSOR_PARALLEL,
        )
        TOKENIZER = MODEL_WRAPPER.get_tokenizer()
        if TOKENIZER.chat_template is not None:
            ENDPOINTS.append("chat")
        verify, generate_question = get_llm_functions(
            MODEL_WRAPPER, TOKENIZER, LOCK, LOCK_GENERATE, ENDPOINTS
        )
        router.add_api_route("/llm/generate", generate_question, methods=["POST"])
        router.add_api_route("/llm/verify", verify, methods=["POST"])
    case _:
        print(f"Unknown pipeline {model.pipeline_tag}")
        exit()


app.include_router(router)


@app.get("/endpoints")
def endpoints():
    return ENDPOINTS


@app.get("/")
def ping():
    return "", 200
