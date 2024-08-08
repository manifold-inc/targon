from asyncpg.connection import asyncio
from neurons.validator import Validator


MINER_UIDS = []

if __name__ == "__main__":
    validator = Validator()
    asyncio.get_event_loop().run_until_complete(validator.score_organic())
