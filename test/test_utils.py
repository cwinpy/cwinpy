"""
Test script for utils.py function.
"""

import numpy as np
import pytest
from cwinpy.utils import alphanum, gcd_array, logfactorial


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


def test_alphanum():
    """
    Test integer to alphabetical string conversion.
    """

    pos = 2.3
    with pytest.raises(TypeError):
        alphanum(pos)

    pos = -1
    with pytest.raises(ValueError):
        alphanum(pos)

    assert alphanum(1) == "A"
    assert alphanum(1, case="lower") == "a"
    assert alphanum(26) == "Z"
    assert alphanum(26, case="lower") == "z"
    assert alphanum(27) == "AA"
    assert alphanum(28) == "AB"
    assert alphanum(200) == "GR"
    assert alphanum(1000) == "ALL"
