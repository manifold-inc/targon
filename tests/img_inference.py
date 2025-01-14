import uvicorn
from OmniGen import OmniGenPipeline
from fastapi import FastAPI
from pydantic import BaseModel
import io
from fastapi.responses import StreamingResponse

app = FastAPI()

# Create Pydantic model for request body
class ImageRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 2.5
    seed: int = 0

# Initialize the pipeline globally
pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/generate")
async def generate_image(request: ImageRequest):
    # Generate image using the pipeline
    images = pipe(
        prompt=request.prompt,
        height=request.height,
        width=request.width,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
    )
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    images[0].save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Return the image as a streaming response
    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


