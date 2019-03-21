"""
General utility functions.
"""

import numpy as np
from scipy.special import gammaln
from math import gcd
from functools import reduce


def logfactorial(n):
    """
    The natural logarithm of the factorial of an integer using the fact that

    .. math::

        \\ln{(n!)} = \\ln{\\left(\\Gamma (n+1)\\right)}

    Parameters
    ----------
    n: int
        An integer for which the natural logarithm of its factorial is
        required.

    Returns
    -------
    float
    """

    if isinstance(n, (int, float, np.int64, np.int32)):
        if isinstance(n, float):
            if not n.is_integer():
                raise ValueError("Can't find the factorial of a non-integer value")

        if n >= 0:
            return gammaln(n+1)
        else:
            raise ValueError("Can't find the factorial of a negative number")
    else:
        raise TypeError("Can't find the factorial of a non-integer value")


def gcd_array(denominators):
    """
    Function to calculate the greatest common divisor of a list of values..
    """

    if not isinstance(denominators, (tuple, list, np.ndarray)):
        raise TypeError("Must have a list or array")

    denoms = np.asarray(denominators).flatten()  # 1d array
    denoms = denoms.astype(np.int)

    if len(denoms) < 2:
        raise ValueError("Must have more than two values")

    return reduce(lambda a, b: gcd(a, b), denoms)
