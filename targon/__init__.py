from .config import *
from .dataset import *

__version__ = "5.0.1"

version_split = __version__.split(".")
__spec_version__ = (
    (100000 * int(version_split[0]))
    + (1000 * int(version_split[1]))
    + (10 * int(version_split[2]))
)
