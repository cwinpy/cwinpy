### import bilby and bilby_pipe removing the logger version info output ###
import logging
import io

bilby_logger = logging.getLogger("bilby")
handler = logging.StreamHandler(io.StringIO())
bilby_logger.addHandler(handler)
import bilby

bilby_logger.removeHandler(handler)
# reset bilby logging handler to default
from bilby.core.utils import setup_logger as bilby_setup_logger

bilby_setup_logger()
bilby_pipe_logger = logging.getLogger("bilby_pipe")
bilby_pipe_logger.addHandler(handler)
import bilby_pipe

bilby_pipe_logger.removeHandler(handler)
# reset bilby_pipe logging handler to default
from bilby_pipe.utils import setup_logger as bilby_pipe_setup_logger

bilby_pipe_setup_logger()
###

# register reader/writer
from . import iostream

from .data import HeterodynedData, MultiHeterodynedData
from .likelihood import TargetedPulsarLikelihood

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
