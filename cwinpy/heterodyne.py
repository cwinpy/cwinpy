"""
Classes for heterodyning strain data.
"""

import os
import numpy as np
import gwpy


# Things that this class should be able to do:
#  - find requested gravitational-wave data
#  - find requested segment lists
#  - coarse heterodyne a chunk of data for multiple pulsars
#  - fine heterodyne data (potentially reading in, sorting, and combining multiple coarse data chunks)
# Things that a pipeline should be able to do
#  - find missing data, i.e., if an analysis chunk has failed notice this


class Heterodyne(object):
    """
    Heterodyne gravitational-wave strain data based on a source's phase evolution.
    """

    def __init__(self,
                 gpsstart,
                 gpsend,
                 detector=None,
                 frtype=None,
                 channel=None,
                 segments=None,
                 excludesegments=None,
                 heterodyne="coarse",
                 pulsars=None,
                 basedir=None,
                 filterknee=0.25,
                 resamplerate=1.0):
        for gpstime in [gpsstart, gpsend]:
            if not isinstance(gpstime, (int, float)):
                raise TypeError("GPS times must be numbers!")

        if gpsstart < gpsend:
            raise ValueError("GPS start time must be before end time!")

        if not isinstance(detector, str):
            raise TypeError("Detector must be a string giving a detector name")

        if not isinstance(channel, (str, list)):
            raise TypeError("Channel must be a string or a list of strings")

        if not isinstance(basedir, str):
            raise TypeError("basedir must be a string")

        if not isinstance(pulsars, (str, list)):
            raise TypeError