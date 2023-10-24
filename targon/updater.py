import os
import sys
import requests
import bittensor as bt
import targon

def autoupdate():
    '''
    Updates the targon codebase if a newer version is available.
    '''
    # try:
    bt.logging.trace("Checking for updates...")
    response = requests.get(
        "https://raw.githubusercontent.com/manifold-inc/targon/main/VERSION",
        headers={'Cache-Control': 'no-cache'}
    )
    response.raise_for_status()
    # try:
    repo_version = response.content.decode()
    
    latest_version = [int(v) for v in repo_version.split(".")]
    local_version = [int(v) for v in targon.__version__.split(".")]
    bt.logging.debug(f"local version: {targon.__version__}")
    bt.logging.debug(f"Latest version: {repo_version}")
    if latest_version > local_version:
        bt.logging.trace("A newer version of Targon is available. Downloading...")
        # download latest version with git pull

        base_path = os.path.abspath(__file__)
        # step backwards in the path until we find targon
        while os.path.basename(base_path) != "targon":
            base_path = os.path.dirname(base_path)
        
        base_path = os.path.dirname(base_path)

        os.system(f"cd {base_path} && git pull")
        # checking local VERSION
        with open(os.path.join(base_path, "VERSION")) as f:
            new__version__ = f.read().strip()
            # convert to list of ints
            new__version__ = [int(v) for v in new__version__.split(".")]
            if new__version__ == latest_version:
                bt.logging.debug("Targon updated successfully.")
                bt.logging.debug("Restarting...")
                bt.logging.debug(f"Running: {sys.executable} {sys.argv}")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                bt.logging.error("Targon git pull failed you will need to manually update and restart for latest code.")
    #     except Exception as e:
    #         bt.logging.error("Failed to convert response to json: {}".format(e))
    #         bt.logging.trace("Response: {}".format(response.text))            
    # except Exception as e:
    #     bt.logging.error("Failed to check for updates: {}".format(e))
