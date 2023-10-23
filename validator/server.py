# server.py

import json
import uuid
import uvicorn
import pprint
import asyncio
import fastapi
import argparse
import random
import bittensor as bt
from fastapi import Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse, Response
from serpapi import GoogleSearch
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
from sse_starlette.sse import EventSourceResponse
from targon.protocol import TargonQA, TargonSearchResultStream, TargonDendrite


# Initialize FastAPI app
app = fastapi.FastAPI()
RETRY_TIMEOUT = 15000  # milisecond

# Placeholder for the TargonDendrite and axons, as they are not defined in the example
dendrite = None  # Replace with actual TargonDendrite initialization
axons = []  # Replace with actual axons list

# Placeholder for SERP API key, should be added as a command-line argument or in the config file
serp_api_key = None  # Replace with actual SERP API key

def random_uuid() -> str:
    """Generate a random UUID."""
    return str(uuid.uuid4())


@app.post("/v1/threads/search")
async def search(request: Request) -> Response:
    """Generate completion for the request."""
    global dendrite, axons
    request_dict = await request.json()
    query = request_dict.pop("query")
    files = request_dict.pop("files", [])  # Added files to accept in payload
    stream = request_dict.pop("stream", True)

    # Fetch sources from Google using google-search-results package
    if config.serp_api_key is None:
        raise ValueError("SERP API key not set. Please set it in the config file.")
    params = {
        "q": query,
        "api_key": config.serp_api_key,
    }
    search_results = GoogleSearch(params).get_json()
    pprint.pprint(search_results)
    organic_results = search_results['organic_results']
    def get_icon(result):
        if result['favicon']:
            return result['favicon']
        else:
            return 'https://www.micreate.eu/wp-content/img/default-img.png'
    search_sources = [{"type": "url", "url": result['link'], "snippet": result['snippet'], "title": result['title'], "icon": get_icon()} for result in organic_results]
    sources = [{"type": "url", "url": result['link'], "title": result['title'], "icon": result['favicon'] if result['favicon'] else 'https://www.micreate.eu/wp-content/img/default-img.png'} for result in organic_results]

    search_result_synapse = TargonSearchResultStream( query=query, sources=search_sources, stream=True )
    related_synapse = TargonQA( question=f"Q:{query}\nRelated Question:", sources=sources, stream=False )
    uuid = random_uuid()
    if stream:
        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            # Stream sources
            
            # yield json.dumps({"type": "sources", "sources": sources}) + '\n'
            # yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}"
            yield {
                        "event": "new_message",
                        "id": uuid,
                        "retry": RETRY_TIMEOUT,
                        "data": json.dumps({"type": "sources", "sources": sources})
            }
            # yield b"\n"

            # get related
            # axon = random.choice(axons)
            # related_responses = await dendrite(axons=[axon], synapse=related_synapse, timeout=12, streaming=False )
            # answers = [response.answer for response in related_responses]
            # yield {
            #         "event": "new_message",
            #         "id": uuid,
            #         "retry": RETRY_TIMEOUT,
            #         "data": json.dumps( {"type": "related", "related": answers} )
            # }

            # Stream answers
            axon = random.choice(axons)
            result_responses = await dendrite(axons=[axon], synapse=search_result_synapse, timeout=60, streaming=True )
            async for token in result_responses:
                print(token)
                if type(token) == str:
                    yield {
                            "event": "new_message",
                            "id": uuid,
                            "retry": RETRY_TIMEOUT,
                            "data": json.dumps( {"type": "answer", "choices": [ {"delta": {"content": token} } ] } )
                    }
                
            # yield json.dumps({"type": "answer", "choices": answers}).encode("utf-8")
            # yield b"\n"

            # Stream done
            yield json.dumps({"type": "[DONE]"}).encode("utf-8")

        return EventSourceResponse(stream_results(), media_type="text/event-stream")

    # Non-streaming case
    # TODO: Implement logic to return non-streaming response if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Add bittensor wallet arguments and SERP API key argument
    parser.add_argument("--serp-api-key", type=str, help="SERP API Key")
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    # args = parser.parse_args()
    # serp_api_key = args.serp_api_key  # Set SERP API key from command-line arguments

    dendrite = TargonDendrite(wallet=bt.wallet(name=config.wallet.name, hotkey=config.wallet.hotkey))
    subtensor = bt.subtensor(network='finney')
    metagraph = subtensor.metagraph(netuid=4)
    axons = [axon for axon in metagraph.axons]
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)



# import json
# import uvicorn
# import fastapi
# import argparse
# import random
# import bittensor as bt
# from fastapi import Request
# from fastapi.responses import JSONResponse, StreamingResponse, Response

# from targon.protocol import TargonSearchResult, TargonSearchResultStream, TargonDendrite
# from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union


# app = fastapi.FastAPI()

# @app.post("/v1/threads/search")
# async def search(request: Request) -> Response:
#     """Generate completion for the request.

#     The request should be a JSON object with the following fields:
#     - prompt: the prompt to use for the generation.
#     - stream: whether to stream the results or not.
#     - other fields: the sampling parameters (See `SamplingParams` for details).
#     """
#     request_dict = await request.json()
#     query = request_dict.pop("query")
#     sources = request_dict.pop("sources", [])
#     stream = request_dict.pop("stream", False)


#     if stream:
#     # Streaming case
#         synapse = TargonSearchResultStream( query=query, sources=sources, stream=True )
#         async def stream_results() -> AsyncGenerator[bytes, None]:
#             axon = random.choice(axons)
#             responses = await dendrite(axons=[axon], synapse=synapse, timeout=60, streaming=True)
#             async for token in responses:
#                 yield token

#         return StreamingResponse(stream_results())
    

#     # Non-streaming case
#     synapse = TargonSearchResult( query=query, sources=sources )
#     axon = random.choice(axons)
#     responses = await dendrite(axons=[axon], synapse=synapse, timeout=60, streaming=False)
#     answers = [response.answer for response in responses]
#     return {"answers": answers}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#         bt.wallet.add_args(parser)

#     config = bt.config(parser)
#     print(config)
#     # This section will need to be modified to initialize bittensor components
#     # and potentially other setup tasks.
#     dendrite = TargonDendrite(wallet=bt.wallet(name=config.wallet.name, hotkey=config.wallet.hotkey))
#     subtensor = bt.subtensor(network='finney')
#     metagraph = subtensor.metagraph(netuid=4)
#     axons = [axon for axon in metagraph.axons]

#     # Run FastAPI server
#     uvicorn.run(app, host="0.0.0.0", port=8000)