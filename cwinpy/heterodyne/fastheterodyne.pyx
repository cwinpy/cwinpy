import cython
cimport cython

import numpy
cimport numpy

from libc.math cimport M_PI

from gwpy.timeseries import TimeSeries

cdef extern from "complex.h":
    double complex cexp(double complex)

cdef double complex I = 1j

ctypedef numpy.complex128_t COMPLEX_DTYPE_t
ctypedef numpy.float64_t DTYPE_t

def fast_heterodyne(timeseries, phase):
    """
    Given a GWPy TimeSeries and an array of phase values return a heterodyned
    version of the time series.

    Parameters
    ----------
    timeseries: TimeSeries
        A :class:`gwpy.timeseries.TimeSeries` object.
    phase: array_like
        An array of phase values in cycles
    
    Returns
    -------
    het: TimeSeries
        A new time series containing the complex heterodyned data.
    """

    # create new complex time series
    het = TimeSeries(numpy.zeros(len(timeseries), dtype=complex))
    het.__array_finalize__(timeseries)
    het.sample_rate = timeseries.sample_rate

    # do the heterodyning
    do_heterodyne(het.value, timeseries.value, phase)

    return het


cdef void do_heterodyne(
    numpy.ndarray[COMPLEX_DTYPE_t, ndim=1] output,
    numpy.ndarray[DTYPE_t, ndim=1] input,
    numpy.ndarray[DTYPE_t, ndim=1] phase
):
    assert len(input) == len(phase), "Time series input and phase must be the same length"
    assert len(output) == len(phase), "Time series output and phase must be the same length"

    cdef int i = 0
    cdef int max = len(input)
    cdef double complex mtwopii = -2.0 * M_PI * I
    cdef double currentvalue = 0.0
    cdef double currentphase = 0.0

    for i in range(max):
        currentvalue = input[i]
        currentphase = phase[i]
        output[i] = currentvalue * cexp(currentphase * mtwopii)