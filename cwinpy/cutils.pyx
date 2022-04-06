import numpy as np
cimport numpy as np

from libc.math cimport M_PI, log, fabs, exp

from numpy.math cimport LOGE2, INFINITY

cdef extern from "gsl/gsl_sf_gamma.h":
    double gsl_sf_lngamma(double x)

cdef extern from "gsl/gsl_sf_log.h":
    double gsl_sf_log_1plusx(double x)

np.import_array()

DOUBLE_DTYPE = np.float64

ctypedef np.complex128_t COMPLEX_DTYPE_t
ctypedef np.float64_t DOUBLE_DTYPE_t

LNPI = log(M_PI)
GSL_DBL_EPSILON = 2.2204460492503131e-16


def find_change_point(subdata, minlength):
    return find_change_point_c(subdata, minlength)


cdef find_change_point_c(np.ndarray[COMPLEX_DTYPE_t, ndim=1] subdata, int minlength):
    cdef int dlen = len(subdata), i = 0
    cdef int lsum = dlen - 2 * minlength + 1
    cdef double logsingle = 0.0, logtot = -INFINITY
    cdef double sumforwards = 0.0, sumbackwards = 0.0
    
    cdef double datasum = np.sum(np.abs(subdata) ** 2)

    cdef np.ndarray logdouble = np.zeros(lsum, dtype=DOUBLE_DTYPE)

    if len(subdata) < 2 * minlength:
        return (-INFINITY, 0, 1)

    # calculate the evidence that the data is drawn from a zero mean
    # Gaussian with a single unknown standard deviation
    logsingle = (
        -LOGE2 - dlen * LNPI + log_factorial_c(dlen - 1) - dlen * log(datasum)
    )

    sumforwards = np.sum(np.abs(subdata[:minlength]) ** 2)
    sumbackwards = np.sum(np.abs(subdata[minlength:]) ** 2)

    # go through each possible splitting of the data in two
    for i in range(lsum):
        if np.all(subdata[: minlength + i] == (0.0 + 0 * 1j)) or np.all(
            subdata[minlength + i :] == (0.0 + 0 * 1j)
        ):
            # do this to avoid warnings about np.log(0.0)
            logdouble[i] = -INFINITY
        else:
            dlenf = minlength + i
            dlenb = dlen - (minlength + i)

            logf = (
                -LOGE2
                - dlenf * LNPI
                + log_factorial_c(dlenf - 1)
                - dlenf * log(sumforwards)
            )
            logb = (
                -LOGE2
                - dlenb * LNPI
                + log_factorial_c(dlenb - 1)
                - dlenb * log(sumbackwards)
            )

            # evidence for that split
            logdouble[i] = logf + logb

        adval = np.abs(subdata[minlength + i]) ** 2
        sumforwards += adval
        sumbackwards -= adval

        # evidence for *any* split
        logtot = logplus(logtot, logdouble[i])

    # change point (maximum of the split evidences)
    cp = np.argmax(logdouble) + minlength

    # ratio of any change point compared to no splits
    logratio = logtot - logsingle

    return (logratio, cp, lsum)


def log_factorial(N):
    """
    Calculate the naturial logarithm of the factorial of N
    """

    return log_factorial_c(N)


cdef double log_factorial_c(int N):
    assert N > 0, "N must be a positive number"

    cdef double np1 = (N + 1.0)

    return gsl_sf_lngamma(np1)


cdef logplus(double x, double y):
    """
    Copied from lintegrate.
    """

    cdef double z = INFINITY
    cdef double tmp = x - y
    if x == y or fabs(tmp) < 1e3 * GSL_DBL_EPSILON:
        z = x + LOGE2
    elif x > y:
        z = x + gsl_sf_log_1plusx(exp(-tmp))
    elif x <= y:
        z = y + gsl_sf_log_1plusx(exp(tmp))
    return z
