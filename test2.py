import bittensor as bt
import requests


endpoint = "https://targon.sybil.com"
wallet = bt.wallet(name="targon_test")

message = requests.get(
    f"{endpoint}/sign-in/bittensor?ss58={wallet.coldkey.ss58_address}"
)
message = message.json()
sig = wallet.coldkey.sign(message.get("message"))
sig = sig.hex()
message = requests.get(
    f"{endpoint}/sign-in/bittensor/callback?ss58={wallet.coldkey.ss58_address}&signature={sig}"
)
message = message.json()

message = requests.get(
    f"{endpoint}/api/credits",
    headers={"Authorization": f"Bearer {message.get('api_key')}"},
)
print(message.json())
