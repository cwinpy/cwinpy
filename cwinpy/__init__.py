### import bilby removing the logger version info output ###
import logging
import io
bilby_logger = logging.getLogger('bilby')
handler = logging.StreamHandler(io.StringIO())
bilby_logger.addHandler(handler)
import bilby
bilby_logger.removeHandler(handler)
# reset bilby logging handler to default
from bilby.core.utils import setup_logger
setup_logger()
###

from .data import HeterodynedData, MultiHeterodynedData
from .likelihood import TargetedPulsarLikelihood

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
