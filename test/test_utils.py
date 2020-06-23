"""
Test script for utils.py function.
"""

import numpy as np
import pytest
from astropy import units as u
from cwinpy.utils import (
    ellipticity_to_q22,
    gcd_array,
    int_to_alpha,
    is_par_file,
    logfactorial,
    q22_to_ellipticity,
)


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


def test_ellipticity_to_q22():
    """
    Test ellipticity conversion to mass quadrupole.
    """

    epsilon = [1e-9, 1e-8]
    expected_q22 = np.array([1e29, 1e30]) * np.sqrt(15.0 / (8.0 * np.pi))
    q22 = ellipticity_to_q22(epsilon[0])

    assert np.isclose(q22, expected_q22[0])

    # test units
    q22units = ellipticity_to_q22(epsilon[0], units=True)

    assert np.isclose(q22units.value, expected_q22[0])
    assert q22units.unit == u.Unit("kg m2")

    # test array like
    q22 = ellipticity_to_q22(epsilon)

    assert len(q22) == len(epsilon)
    assert np.allclose(q22, expected_q22)


def test_q22_to_ellipticity_to_q22():
    """
    Test mass quadrupole conversion to ellipticity.
    """

    q22 = [1e29, 1e30]
    expected_epsilon = np.array([1e-9, 1e-8]) / np.sqrt(15.0 / (8.0 * np.pi))
    epsilon = q22_to_ellipticity(q22[0])

    assert np.isclose(epsilon, expected_epsilon[0])

    # test array like
    epsilon = q22_to_ellipticity(q22)

    assert len(q22) == len(epsilon)
    assert np.allclose(epsilon, expected_epsilon)

    # test no unit
    epsilon = q22_to_ellipticity(q22[0] * u.kg * u.m ** 2)

    assert np.isclose(epsilon, expected_epsilon[0])
    assert not hasattr(epsilon, "unit")
