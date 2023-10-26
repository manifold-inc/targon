# server.py

import json
import uuid
import torch
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


def check_favicon(dict, key):
    if key in dict.keys():
        print("Present, ", end =" ")
        print("value =", dict[key])
        return True
    else:
        print("Not present")
        return False


def select_highest_n_peers(n: int, return_all=False):
    """
    Selects the highest incentive peers from the metagraph.

    Parameters:
        n (int): number of top peers to return.

    Returns:
        int: uid of the selected peer from unique highest IPs.
    """
    # Get the top n indices based on incentive
    indices = torch.topk(metagraph.incentive, n).indices

    # Get the corresponding uids
    uids_with_highest_incentives = metagraph.uids[indices].tolist()


    if return_all:
        return uids_with_highest_incentives
    
    # get the axon of the uids
    axons = [metagraph.axons[uid] for uid in uids_with_highest_incentives]

    # get the ip from the axons
    ips = [axon.ip for axon in axons]

    # get the coldkey from the axons
    coldkeys = [axon.coldkey for axon in axons]

    # Filter out the uids and ips whose coldkeys are in the blacklist
    uids_with_highest_incentives, ips = zip(*[(uid, ip) for uid, ip, coldkey in zip(uids_with_highest_incentives, ips, coldkeys)])

    unique_ip_to_uid = {ip: uid for ip, uid in zip(ips, uids_with_highest_incentives)}
    uids = list(unique_ip_to_uid.values())
    return uids_with_highest_incentives

async def is_successful(response: AsyncGenerator) -> bool:
    """Check if the response is successful."""
    error_0 = "Exception: Expecting value: line 1 column 1 (char 0)"
    error_1 = "Exception: 'description'"
    
    async for chunk in response:
        if isinstance(chunk, str) and (error_0 in chunk or error_1 in chunk):
            return False
        return True
        
error_0 = "Exception: Expecting value: line 1 column 1 (char 0)"
error_1 = "Exception: 'description'"
    
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
        if check_favicon(result, 'favicon'):
            return result['favicon']
        else:
            return 'https://www.micreate.eu/wp-content/img/default-img.png'
    search_sources = [{"type": "url", "url": result['link'], "snippet": result['snippet'], "title": result['title'], "icon": get_icon(result)} for result in organic_results]
    sources = [{"type": "url", "url": result['link'], "title": result['title'], "icon": get_icon(result)} for result in organic_results]

    search_result_synapse = TargonSearchResultStream( query=query, sources=search_sources, stream=True, max_new_tokens=1024 )
    related_synapse = TargonQA( question=f"Question:{query}\nWhat is a related question?:", sources=sources, stream=False, max_new_tokens=6 )
    uuid_str = random_uuid()
    if stream:
        async def stream_results() -> AsyncGenerator[Dict[str, str], None]:
            yield {
                "event": "new_message",
                "id": uuid_str,
                "retry": RETRY_TIMEOUT,
                "data": json.dumps({"type": "sources", "sources": sources})
            }

            k = 192 
            # Select the top-k axons based on incentive
            # uids = select_highest_n_peers(k)
            uids = [14]
            top_k_axons = [metagraph.axons[uid] for uid in uids]

            # Create a list to store the results from each axon
            results = [asyncio.create_task(dendrite(axons=[axon], synapse=search_result_synapse, timeout=60, streaming=True)) for axon in top_k_axons]

            # Set a global timeout (for example, 120 seconds)
            global_timeout = 120

            try:
                # Wait for the fastest successful response within the global timeout
                while True:
                    done, pending = await asyncio.wait(results, return_when=asyncio.FIRST_COMPLETED, timeout=global_timeout)

                    for task in done:
                        if task.exception() is not None:
                            # Handle task exception if necessary
                            continue

                        fastest_response = task.result()

                        # Here you should implement your logic to check if the response is successful
                        # if await is_successful(fastest_response):
                        async for token in fastest_response:
                            if (error_1 in token):
                                pass
                            # print(token)
                            if isinstance(token, str):
                                yield {
                                        "event": "new_message",
                                        "id": uuid_str,
                                        "retry": RETRY_TIMEOUT,
                                        "data": json.dumps({"type": "answer", "choices": [{"delta": {"content": token}}]})
                                }
                                print(token, end="", flush=True)
                        return  # End the stream after a successful response
                    if not pending:
                        break  # All tasks are done, exit the loop

            except asyncio.TimeoutError:
                # Handle global timeout, no successful response received
                yield {
                    "event": "error",
                    "id": uuid_str,
                    "retry": RETRY_TIMEOUT,
                    "data": json.dumps({"type": "error", "message": "Request timed out. No successful response received."})
                }
            finally:
                # Cancel all pending tasks
                for task in pending:
                    task.cancel()

            # If no successful response received, you can send a default response or handle it as needed

            # Stream done
            yield json.dumps({"type": "[DONE]"}).encode("utf-8")

        return EventSourceResponse(stream_results(), media_type="text/event-stream")
    # Non-streaming case
    # TODO: Implement logic to return non-streaming response if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Add bittensor wallet arguments and SERP API key argument
    parser.add_argument("--serp-api-key", type=str, help="SERP API Key")
    path = None
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    # args = parser.parse_args()
    # serp_api_key = args.serp_api_key  # Set SERP API key from command-line arguments

    dendrite = TargonDendrite(wallet=bt.wallet(name=config.wallet.name, hotkey=config.wallet.hotkey, path=path if path is not None else '~/.bittensor/wallets'))
    subtensor = bt.subtensor(network='finney')
    metagraph = subtensor.metagraph(netuid=4)
    # axons = [axon for axon in metagraph.axons]

    # axon = bt.AxonInfo(
    #     ip="0.0.0.0",
    #     port=8098,
    #     ip_type=4,
    #     hotkey="0x0",
    #     coldkey="0x0",
    #     protocol=4,
    #     version=1042,
    # )

    # axons = [axon]

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

