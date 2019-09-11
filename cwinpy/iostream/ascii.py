"""
ASCII I/O registrations for cwinpy.HeterodynedData objects
"""

from .readers import register_ascii_series_io
from ..data import HeterodynedData

# -- registration -------------------------------------------------------------

register_ascii_series_io(HeterodynedData, format='txt')
register_ascii_series_io(HeterodynedData, format='txt.gz')
