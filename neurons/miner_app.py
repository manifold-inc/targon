from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/inference")
async def inference(req: Request):
    # TODO implement epistula
    return {"Hello": "World"}
