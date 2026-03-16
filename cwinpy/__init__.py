import io
import warnings
from importlib.metadata import version, PackageNotFoundError


# suppress warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Wswiglal-redir-stdio")
    warnings.filterwarnings("ignore", message="(.*)CONDOR_CONFIG(.*)")

    # register reader/writer
    from . import iostream
    from .data import HeterodynedData, MultiHeterodynedData
    from .info import *
    from .parfile import PulsarParameters

try:
    __version__ = version("cwinpy")
except PackageNotFoundError:
    __version__ = ""
