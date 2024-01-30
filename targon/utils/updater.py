# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Manifold Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import sys
import requests
import bittensor as bt
import targon

def autoupdate(branch: str = "main"):
    '''
    Updates the targon codebase if a newer version is available.
    '''
    # try:
    bt.logging.info("Checking for updates...")
    response = requests.get(
        f"https://raw.githubusercontent.com/manifold-inc/targon/{branch}/VERSION",
        headers={'Cache-Control': 'no-cache'}
    )
    response.raise_for_status()
    # try:
    repo_version = response.content.decode()
    
    latest_version = [int(v) for v in repo_version.split(".")]
    local_version = [int(v) for v in targon.__version__.split(".")]
    bt.logging.info(f"local version: {targon.__version__}")
    bt.logging.info(f"Latest version: {repo_version}")
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
                bt.logging.info("Targon updated successfully.")
                bt.logging.info("Restarting...")
                bt.logging.info(f"Running: {sys.executable} {sys.argv}")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                bt.logging.error("Targon git pull failed you will need to manually update and restart for latest code.")
    #     except Exception as e:
    #         bt.logging.error("Failed to convert response to json: {}".format(e))
    #         bt.logging.trace("Response: {}".format(response.text))            
    # except Exception as e:
    #     bt.logging.error("Failed to check for updates: {}".format(e))