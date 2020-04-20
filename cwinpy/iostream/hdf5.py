"""
HDF5 I/O registrations for cwinpy.HeterodynedData objects
"""

from ..data import HeterodynedData
from .readers import register_hdf_series_io

# -- registration -------------------------------------------------------------

register_hdf_series_io(HeterodynedData, format="hdf")
register_hdf_series_io(HeterodynedData, format="hdf5")
register_hdf_series_io(HeterodynedData, format="h5")
