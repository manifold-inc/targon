import os 

from . import protocol
from . import prompts
from .updater import autoupdate

from dotenv import load_dotenv

load_dotenv()
from .miner.search import query as search
from .miner.search import QueryParams

from .crawler import Crawler
__version__ = "0.4.6"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)