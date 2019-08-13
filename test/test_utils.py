"""
Test script for utils.py function.
"""

import pytest
import numpy as np

from cwinpy.utils import logfactorial, gcd_array


def test_logfactorial():
    """
    Test log factorial function
    """

    a = 1.5  # non-integer float
    with pytest.raises(ValueError):
        answer = logfactorial(a)

    a = "a"  # non-number type
    with pytest.raises(TypeError):
        answer = logfactorial(a)

    a = -1  # negative number
    with pytest.raises(ValueError):
        answer = logfactorial(a)

    a = 3
    assert logfactorial(a) == np.log(3 * 2 * 1)

    a = 3.0
    assert logfactorial(a) == np.log(3 * 2 * 1)


def test_gcd_array():
    """
    Test greatest common divisor function.
    """

    a = 1  # non-list value
    with pytest.raises(TypeError):
        answer = gcd_array(a)

    a = [1]  # single value
    with pytest.raises(ValueError):
        answer = gcd_array(a)

    a = [5, 25, 90]
    assert gcd_array(a) == 5
