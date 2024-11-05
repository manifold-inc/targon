from dotenv import load_dotenv
import os

load_dotenv()

NO_AUTO_UPDATE = not not os.getenv("NO_AUTO_UPDATE", False)
VERIFIER_IMAGE_TAG = os.getenv("IMAGE_TAG", "latest")
GPUS = os.getenv("GPUS", None)
if GPUS is not None:
    GPUS = GPUS.split(",")
