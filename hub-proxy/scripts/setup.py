import argparse
import bittensor as bt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet", type=str, required=True)
    args = parser.parse_args()

    wallet = bt.wallet(name=args.wallet)
    public_key = wallet.hotkey.public_key
    private_key = wallet.hotkey.private_key
    public_key_str = public_key.hex()
    private_key_str = private_key.hex()

    print(f"HOTKEY={wallet.hotkey.ss58_address}")
    print(f"PUBLIC_KEY={public_key_str}")
    print(f"PRIVATE_KEY={private_key_str}")
