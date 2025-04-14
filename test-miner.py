import asyncio
import json
import aiohttp

from targon.broadcast import cvm_attest, cvm_healthcheck
from neurons.validator import Validator
from targon.jugo import score_cvm_attestations


async def main(uid: int):
    vali = Validator(standalone=True)
    vali.cvm_nodes = {}
    async with aiohttp.ClientSession() as session:
        h, u, nodes = await cvm_healthcheck(
            vali.metagraph, uid, session, vali.wallet.hotkey
        )
        vali.cvm_nodes[u] = (h, nodes)
        print(vali.cvm_nodes)
        hk = vali.metagraph.axons[uid].hotkey
        res = []
        for i in nodes:
            res.append(await cvm_attest(i, uid, session, hk, vali.wallet.hotkey))
        print(res)
        for r in res:
            if r is None:
                continue
            uid, node_url, result = r
            if uid not in vali.cvm_attestations:
                vali.cvm_attestations[uid] = {}
            if node_url not in vali.cvm_attestations[uid]:
                vali.cvm_attestations[uid][node_url] = []

            vali.cvm_attestations[uid][node_url].append(result)
        attestation_stats = await score_cvm_attestations(
            vali.cvm_attestations,
        )
        print(json.dumps(attestation_stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main(int(input("Enter uid: "))))

# python scripts/print_weights.py --wallet.name _
