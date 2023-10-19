from . import protocol
from dotenv import load_env

load_env()
from .miner.search import query as search


__version__ = "0.1.0"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)