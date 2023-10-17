import asyncio
import uvicorn
import fastapi
import json
import time
from http import HTTPStatus
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import bittensor as bt
from targon.protocol import TargonStreaming, TargonDendrite
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

from server_protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    ModelCard, ModelList, ModelPermission, UsageInfo, random_uuid
)
from packaging import version

from conversation import conv_templates, SeparatorStyle

app = fastapi.FastAPI()
dendrite = None
axons = None

async def get_gen_prompt(request) -> str:
    # Check if fastchat is available

    # Fetch the conversation template based on the requested model
    conv_template = conv_templates.get(request.model, conv_templates['default'])
    conv = conv_template.copy()  # Create a copy of the template to avoid modifying the original

    # Populate the conversation based on the provided messages
    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system = message["content"]
            # elif msg_role in conv.roles:
            #     conv.append_message(msg_role, message["content"])
            else:
                conv.append_message(message['role'], message['content'])

        # Get the prompt from the conversation instance
        prompt = conv.get_prompt()

    return prompt

@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    # This will likely need to be adjusted to reflect the models available in your bittensor setup.
    model_cards = [
        ModelCard(id="sybil-v0", root="sybil-root", permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    global dendrite, axons
    # TODO: Perform checks similar to original server
    #       (e.g., validate model, check input length, etc.)
    print(request)
    prompt = await get_gen_prompt(request)  # You might need to adapt this function

    synapse = TargonStreaming(roles=['user'], messages=[prompt])
    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # Initial setup, sending the role of the assistant for each completion
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                choices=[choice_data],
                                                model=model_name)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        # Assuming dendrite() returns a generator of responses
        responses = await dendrite(axons=axons, synapse=synapse, timeout=60, streaming=True)

        # Convert bittensor responses into OpenAI-compatible format
        async for token in responses:
            # Convert response to text (assuming it's a string; adjust as needed)
            text = token  # or response.decode() or some other conversion

            # Construct the OpenAI-compatible response
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=text),  # The main content from bittensor
                finish_reason=None,  # or some condition to determine if the completion is finished
            )
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                choices=[choice_data],
                                                model=model_name)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        # Indicate end of stream
        yield "data: [DONE]\n\n"


    if request.stream:
        return StreamingResponse(completion_stream_generator(), media_type="text/event-stream")
    else:
        # Non-streaming response
        # TODO: Collect all results and format them into a standard response
        return JSONResponse(content={}, status_code=HTTPStatus.OK)
    # TODO: Add more code for handling other features and scenarios

if __name__ == "__main__":
    # This section will need to be modified to initialize bittensor components
    # and potentially other setup tasks.
    dendrite = TargonDendrite(wallet=bt.wallet(name="lilith", hotkey="A4"))
    subtensor = bt.subtensor(network='finney')
    metagraph = subtensor.metagraph(netuid=4)
    axons = [axon for axon in metagraph.axons if axon.ip == '160.202.128.179']

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
