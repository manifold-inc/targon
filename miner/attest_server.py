from fastapi import FastAPI


app = FastAPI()


@app.post("/attest")
async def attest():
    return
