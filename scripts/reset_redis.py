import asyncio
import argparse
from redis import asyncio as aioredis
from pydantic import BaseModel

class Config(BaseModel):
    host: str
    port: int
    index: int
    password: str


async def reset_redis(config):
    redis = aioredis.Redis(
        host=config.host,
        port=config.port,
        db=config.index,
        password=config.password,
    )

    await redis.flushdb()
    print("Redis database has been reset.")
    await redis.aclose()

def main():
    parser = argparse.ArgumentParser(description="Reset a Redis database.")
    parser.add_argument('--host', type=str, default='localhost', help='Host for the Redis database')
    parser.add_argument('--port', type=int, default=6379, help='Port number for the Redis database')
    parser.add_argument('--db_index', type=int, default=1, help='Database index to use')
    parser.add_argument('--password', type=str, help='Password for the Redis database')

    args = parser.parse_args()

    config = Config(host=args.host, port=args.port, index=args.db_index, password=args.password)

    asyncio.run(reset_redis(config))

if __name__ == "__main__":
    main()
