"""
Functions to deal with heterodyned data.
"""

from __future__ import division, print_function

import numpy as np


def subtract_running_median(data, N=30):
    """
    Calculate and subtract a running median from a 1D data series. The running
    median will be calculated using a window of samples of a given number.

    Parameters
    ----------
    data: array_like
        a series of data points from which to subtract a running median.
    N: int, 30
        the window length of the running median. Defaults to 30 points.

    Returns
    -------
    newdata: array_like
        an array containing the data with with running median subtracted. 
    """

    datacopy = np.atleast_1d(np.asarray(data))

    if len(datacopy.shape) > 1:
        raise ValueError("Data must be a 1d array")

    if N < 2:
        raise ValueError("The running median window must be greater than 1")

    # create copy of data with buffers prepended and appended
    datacopy = np.hstack((datacopy[:N//2], datacopy, datacopy[-N//2:]))

    running_median = np.zeros(len(data))
    for i in range(len(data)):
        running_median[i] = np.median(datacopy[i:i+N])

    return data - running_median


def compute_variance(data, change_points=None, N=30):
    """
    Compute the (sample) variance of the data within a set of change points.
    The variance will be calculated after subtraction of a running median.
    If the data is complex, we calculate the variance of a vector in which the
    real and imaginary components are concatenated. This is equivalent to a
    two-sided power spectral density.

    Parameters
    ----------
    data: array_like
        A 1d array of (complex) data.
    change_points: array_like, None
        An array of indices of statistical change points within the data
    N: int, 30
        The window size (in terms of data point number) of the running
        median.
    """

    # subtract running median from the data
    datasub = subtract_running_median(data, N=N)

    if change_points is None:
        # return the (sample) variance (hence 'ddof=1')
        if datasub.dtype == np.complex:
            return np.hstack(datasub.real, datasub.imag).var(ddof=1)
        else:
            return datasub.var(ddof=1)
    else:
        cps = np.concatenate(([0], np.asarray(change_points, dtype=np.int),
                              [len(datasub)]))
        variances = np.zeros(len(cps)-1)

        for i in range(len(cps)-1):
            if cps[i+1] < 1 or cps[i+1] > len(datasub)-2:
                raise ValueError("Change point index is out of bounds")

            if cps[i+1] <= cps[i]:
                raise ValueError("Change point order is wrong")

            datachunk = datasub[cps[i]:cps[i+1]]

            # get (sample) variance of chunk
            if datasub.dtype == np.complex:
                variances[i] = np.hstack(datachunk.real, datachunk.imag).var(ddof=1)
            else:
                variances[i] = datachunk.var(ddof=1)

        return variances
