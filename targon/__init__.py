from .config import *
from .dataset import *

__version__ = '2.2.5'

version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (100 * int(version_split[1]))
    + (10 * int(version_split[2]))
)
