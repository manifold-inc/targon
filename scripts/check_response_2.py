import asyncio
from enum import Enum
from requests import post


class Endpoints(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


async def check_tokens(
    request, responses, uid, endpoint: Endpoints, port: int, url="http://localhost"
):
    try:
        result = post(
            f"{url}:{port}/verify",
            headers={"Content-Type": "application/json"},
            json={
                "model": request.get("model"),
                "request_type": endpoint.value,
                "request_params": request,
                "output_sequence": responses,
            },
        ).json()
        if err := result.get("error") is not None:
            print(str(err))
            return None
        return result
    except Exception as e:
        print(f"{uid}: " + str(e))
        return None


async def main():
    request = {}
    responses = []
    uid = -1
    endpoint = Endpoints.COMPLETION
    port = 7777
    res = await check_tokens(request, responses, uid, endpoint, port)
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
