import bittensor as bt
from targon.protocol import TargonQA, TargonLinkPrediction, TargonSearchResult, TargonSearchResultStream
from typing import List, Union
import asyncio


class AsyncDendritePool:
    def __init__(self, wallet, metagraph):
        self.metagraph = metagraph
        self.dendrite = bt.dendrite(wallet=wallet)
    
    async def async_forward(
            self,
            uids: List[int],
            synapse: Union[TargonQA, TargonLinkPrediction, TargonSearchResult, TargonSearchResultStream],
            timeout: float = 12.0
    ):

        def call_single_uid(uid):
            synapse_for_generation = synapse.copy()
            return self.dendrite(
                self.metagraph.axons[uid],
                synapse=synapse_for_generation,
                timeout=timeout
            )

        
        async def query_async():
            corutines = [call_single_uid(uid) for uid in uids]
            return await asyncio.gather(*corutines)
        
        return await query_async()