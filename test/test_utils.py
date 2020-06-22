"""
Test script for utils.py function.
"""

import numpy as np
import pytest
from cwinpy.utils import gcd_array, int_to_alpha, is_par_file, logfactorial


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


def test_int_to_alpha():
    """
    Test integer to alphabetical string conversion.
    """

    pos = 2.3
    with pytest.raises(TypeError):
        int_to_alpha(pos)

    pos = -1
    with pytest.raises(ValueError):
        int_to_alpha(pos)

    assert int_to_alpha(1) == "A"
    assert int_to_alpha(1, case="lower") == "a"
    assert int_to_alpha(26) == "Z"
    assert int_to_alpha(26, case="lower") == "z"
    assert int_to_alpha(27) == "AA"
    assert int_to_alpha(28) == "AB"
    assert int_to_alpha(200) == "GR"
    assert int_to_alpha(1000) == "ALL"


def test_is_par_file():
    """
    Test failure of is_par_file.
    """

    assert is_par_file("blah_blah_blah") is False
