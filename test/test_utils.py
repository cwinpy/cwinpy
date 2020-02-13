"""
Test script for utils.py function.
"""

import numpy as np
import pytest
from cwinpy.utils import gcd_array, logfactorial


def test_logfactorial():
    """
    Test log factorial function
    """

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
        gcd_array(a)

    a = [1]  # single value
    with pytest.raises(ValueError):
        gcd_array(a)

    a = [5, 25, 90]
    assert gcd_array(a) == 5
