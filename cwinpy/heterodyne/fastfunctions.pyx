import cython
cimport cython

import numpy
cimport numpy

from libc.math cimport M_PI

import lal
from gwpy.timeseries import TimeSeries

cdef extern from "complex.h":
    double complex cexp(double complex)

cdef double complex I = 1j

ctypedef numpy.complex128_t COMPLEX_DTYPE_t
ctypedef numpy.float64_t DTYPE_t

cdef extern from "filter_core.h":
    double filter_core(double x, int nrecurs, double *recursCoef, int ndirect, double *directCoef, int nhistory, double *history)

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


def fast_filter(pulsar, data, filters, forwardsonly=False):
    """
    Apply the low pass filters to the data for a particular pulsar.

    Parameters
    ----------
    pulsar: str
        The name of the pulsar who's data is being filtered.
    data: array_like
        The array of complex heterodyned data to be filtered.
    forwardswonly: bool
        Set to True to only filter the data in the forwards direction. This
        means that the filter phase lag will still be present.
    """

    # filter data in the forward direction
    do_filter_fowards(data.value, filters[pulsar][0][0], filters[pulsar][1][0], filters[pulsar][2][0], filters[pulsar][0][1], filters[pulsar][1][1], filters[pulsar][2][1])

    if not forwardsonly:
        do_filter_backwards(data.value, filters[pulsar][0][0], filters[pulsar][1][0], filters[pulsar][2][0], filters[pulsar][0][1], filters[pulsar][1][1], filters[pulsar][2][1])
        

cdef void do_filter_fowards(numpy.ndarray[COMPLEX_DTYPE_t, ndim=1] data, freal1, freal2, freal3, fimag1, fimag2, fimag3):
    cdef int i = 0
    cdef int max = len(data)
    cdef double rd = 0.0
    cdef double id = 0.0

    # extract filter data into buffers
    cdef int ndirect = freal1.directCoef.length
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rdirect1_buff = numpy.ascontiguousarray(freal1.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rdirect2_buff = numpy.ascontiguousarray(freal2.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rdirect3_buff = numpy.ascontiguousarray(freal3.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] idirect1_buff = numpy.ascontiguousarray(fimag1.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] idirect2_buff = numpy.ascontiguousarray(fimag2.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] idirect3_buff = numpy.ascontiguousarray(fimag3.directCoef.data, dtype=numpy.float64)

    cdef int nrecurs = freal1.recursCoef.length
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rrecurs1_buff = numpy.ascontiguousarray(freal1.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rrecurs2_buff = numpy.ascontiguousarray(freal2.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rrecurs3_buff = numpy.ascontiguousarray(freal3.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] irecurs1_buff = numpy.ascontiguousarray(fimag1.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] irecurs2_buff = numpy.ascontiguousarray(fimag2.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] irecurs3_buff = numpy.ascontiguousarray(fimag3.recursCoef.data, dtype=numpy.float64)

    cdef int nhistory = freal1.history.length
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rhist1_buff = numpy.ascontiguousarray(freal1.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rhist2_buff = numpy.ascontiguousarray(freal2.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rhist3_buff = numpy.ascontiguousarray(freal3.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] ihist1_buff = numpy.ascontiguousarray(fimag1.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] ihist2_buff = numpy.ascontiguousarray(fimag2.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] ihist3_buff = numpy.ascontiguousarray(fimag3.history.data, dtype=numpy.float64)

    # get memory views
    cdef double[::1] rdirect1 = rdirect1_buff
    cdef double[::1] rdirect2 = rdirect2_buff
    cdef double[::1] rdirect3 = rdirect3_buff
    cdef double[::1] idirect1 = idirect1_buff
    cdef double[::1] idirect2 = idirect2_buff
    cdef double[::1] idirect3 = idirect3_buff

    cdef double[::1] rrecurs1 = rrecurs1_buff
    cdef double[::1] rrecurs2 = rrecurs2_buff
    cdef double[::1] rrecurs3 = rrecurs3_buff
    cdef double[::1] irecurs1 = irecurs1_buff
    cdef double[::1] irecurs2 = irecurs2_buff
    cdef double[::1] irecurs3 = irecurs3_buff

    cdef double[::1] rhist1 = rhist1_buff
    cdef double[::1] rhist2 = rhist2_buff
    cdef double[::1] rhist3 = rhist3_buff
    cdef double[::1] ihist1 = ihist1_buff
    cdef double[::1] ihist2 = ihist2_buff
    cdef double[::1] ihist3 = ihist3_buff

    # apply filters
    for i in range(max):
        rd = data[i].real
        id = data[i].imag
        # data[i] = lal.IIRFilterREAL8(rd, freal1) + I * lal.IIRFilterREAL8(id, fimag1)
        data[i] = filter_core(rd, nrecurs, &rrecurs1[0], ndirect, &rdirect1[0], nhistory, &rhist1[0]) + I * filter_core(id, nrecurs, &irecurs1[0], ndirect, &idirect1[0], nhistory, &ihist1[0])

        rd = data[i].real
        id = data[i].imag
        # data[i] = lal.IIRFilterREAL8(rd, freal2) + I * lal.IIRFilterREAL8(id, fimag2)
        data[i] = filter_core(rd, nrecurs, &rrecurs2[0], ndirect, &rdirect2[0], nhistory, &rhist2[0]) + I * filter_core(id, nrecurs, &irecurs2[0], ndirect, &idirect2[0], nhistory, &ihist2[0])

        rd = data[i].real
        id = data[i].imag
        #data[i] = lal.IIRFilterREAL8(rd, freal3) + I * lal.IIRFilterREAL8(id, fimag3)
        data[i] = filter_core(rd, nrecurs, &rrecurs3[0], ndirect, &rdirect3[0], nhistory, &rhist3[0]) + I * filter_core(id, nrecurs, &irecurs3[0], ndirect, &idirect3[0], nhistory, &ihist3[0])


cdef void do_filter_backwards(numpy.ndarray[COMPLEX_DTYPE_t, ndim=1] data, freal1, freal2, freal3, fimag1, fimag2, fimag3):
    cdef int i = 0
    cdef int max = len(data)
    cdef double rd = 0.0
    cdef double id = 0.0

    # extract filter data into buffers
    cdef int ndirect = freal1.directCoef.length
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rdirect1_buff = numpy.ascontiguousarray(freal1.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rdirect2_buff = numpy.ascontiguousarray(freal2.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rdirect3_buff = numpy.ascontiguousarray(freal3.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] idirect1_buff = numpy.ascontiguousarray(fimag1.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] idirect2_buff = numpy.ascontiguousarray(fimag2.directCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] idirect3_buff = numpy.ascontiguousarray(fimag3.directCoef.data, dtype=numpy.float64)

    cdef int nrecurs = freal1.recursCoef.length
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rrecurs1_buff = numpy.ascontiguousarray(freal1.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rrecurs2_buff = numpy.ascontiguousarray(freal2.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rrecurs3_buff = numpy.ascontiguousarray(freal3.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] irecurs1_buff = numpy.ascontiguousarray(fimag1.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] irecurs2_buff = numpy.ascontiguousarray(fimag2.recursCoef.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] irecurs3_buff = numpy.ascontiguousarray(fimag3.recursCoef.data, dtype=numpy.float64)

    cdef int nhistory = freal1.history.length
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rhist1_buff = numpy.ascontiguousarray(freal1.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rhist2_buff = numpy.ascontiguousarray(freal2.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] rhist3_buff = numpy.ascontiguousarray(freal3.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] ihist1_buff = numpy.ascontiguousarray(fimag1.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] ihist2_buff = numpy.ascontiguousarray(fimag2.history.data, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1, mode="c"] ihist3_buff = numpy.ascontiguousarray(fimag3.history.data, dtype=numpy.float64)

    # get memory views
    cdef double[::1] rdirect1 = rdirect1_buff
    cdef double[::1] rdirect2 = rdirect2_buff
    cdef double[::1] rdirect3 = rdirect3_buff
    cdef double[::1] idirect1 = idirect1_buff
    cdef double[::1] idirect2 = idirect2_buff
    cdef double[::1] idirect3 = idirect3_buff

    cdef double[::1] rrecurs1 = rrecurs1_buff
    cdef double[::1] rrecurs2 = rrecurs2_buff
    cdef double[::1] rrecurs3 = rrecurs3_buff
    cdef double[::1] irecurs1 = irecurs1_buff
    cdef double[::1] irecurs2 = irecurs2_buff
    cdef double[::1] irecurs3 = irecurs3_buff

    cdef double[::1] rhist1 = rhist1_buff
    cdef double[::1] rhist2 = rhist2_buff
    cdef double[::1] rhist3 = rhist3_buff
    cdef double[::1] ihist1 = ihist1_buff
    cdef double[::1] ihist2 = ihist2_buff
    cdef double[::1] ihist3 = ihist3_buff

    # store history
    historyreal = [
        freal1.history.data.copy(),
        freal2.history.data.copy(),
        freal3.history.data.copy(),
    ]

    historyimag = [
        fimag1.history.data.copy(),
        fimag2.history.data.copy(),
        fimag3.history.data.copy(),
    ]

    # apply filters
    for i in range(max - 1, -1, -1):
        rd = data[i].real
        id = data[i].imag
        # data[i] = lal.IIRFilterREAL8(rd, freal1) + I * lal.IIRFilterREAL8(id, fimag1)
        data[i] = filter_core(rd, nrecurs, &rrecurs1[0], ndirect, &rdirect1[0], nhistory, &rhist1[0]) + I * filter_core(id, nrecurs, &irecurs1[0], ndirect, &idirect1[0], nhistory, &ihist1[0])

        rd = data[i].real
        id = data[i].imag
        # data[i] = lal.IIRFilterREAL8(rd, freal2) + I * lal.IIRFilterREAL8(id, fimag2)
        data[i] = filter_core(rd, nrecurs, &rrecurs2[0], ndirect, &rdirect2[0], nhistory, &rhist2[0]) + I * filter_core(id, nrecurs, &irecurs2[0], ndirect, &idirect2[0], nhistory, &ihist2[0])

        rd = data[i].real
        id = data[i].imag
        #data[i] = lal.IIRFilterREAL8(rd, freal3) + I * lal.IIRFilterREAL8(id, fimag3)
        data[i] = filter_core(rd, nrecurs, &rrecurs3[0], ndirect, &rdirect3[0], nhistory, &rhist3[0]) + I * filter_core(id, nrecurs, &irecurs3[0], ndirect, &idirect3[0], nhistory, &ihist3[0])

    # restore the history to that from the forward pass
    freal1.history.data = historyreal[0]
    freal2.history.data = historyreal[1]
    freal3.history.data = historyreal[2]

    fimag1.history.data = historyimag[0]
    fimag2.history.data = historyimag[1]
    fimag3.history.data = historyimag[2]