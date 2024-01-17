import io
import warnings

# suppress warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Wswiglal-redir-stdio")
    warnings.filterwarnings("ignore", message="(.*)CONDOR_CONFIG(.*)")

    # register reader/writer
    from . import iostream
    from .data import HeterodynedData, MultiHeterodynedData
    from .info import *
    from .knope import skyshift
    from .parfile import PulsarParameters

try:
    from ._version import version as __version__
except ModuleNotFoundError:
    __version__ = ""
