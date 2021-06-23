"""
ASCII I/O registrations for cwinpy.HeterodynedData objects
"""

from ..data import HeterodynedData
from .readers import register_ascii_series_io

# -- registration -------------------------------------------------------------

register_ascii_series_io(HeterodynedData, format="txt")
register_ascii_series_io(HeterodynedData, format="txt.gz")
register_ascii_series_io(HeterodynedData, format="csv", delimiter=",")
