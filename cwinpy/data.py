"""
Functions to deal with heterodyned data.
"""

from __future__ import division, print_function

import numpy as np


def subtract_running_median(data, N=30):
    """

    Parameters
    ----------
    data (array_like): a time series of data points from which to subtract a
        running median
    N (int): the window length of the running median. Defaults to 30 points.
    """

    # create copy of data with buffers prepended and appended
    datacopy = np.hstack((data[:N//2], data, data[-N//2:]))

    running_median = np.zeros(len(data))
    for i in range(len(data)):
        running_median[i] = np.median(datacopy[i:i+N])

    return data - running_median
