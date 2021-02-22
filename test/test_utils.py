"""
Test script for utils.py function.
"""

import os

import numpy as np
import pytest
from astropy import units as u
from cwinpy.utils import (
    ellipticity_to_q22,
    gcd_array,
    get_psr_name,
    initialise_ephemeris,
    int_to_alpha,
    is_par_file,
    logfactorial,
    q22_to_ellipticity,
)
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy


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

    # test par files that don't contain required attributes
    brokenpar = "broken.par"

    values = {
        "F": [100.0],
        "RAJ": 0.1,
        "DECJ": -0.1,
        "PSRJ": "J0101-0101",
    }

    for leavekey in list(values.keys()):
        keys = list(values.keys())
        psr = PulsarParametersPy()
        for key in keys:
            if key != leavekey:
                psr[key] = values[key]

        psr.pp_to_par(brokenpar)

        assert is_par_file(brokenpar) is False

        os.remove(brokenpar)


def test_get_psr_name():
    """
    Test extraction of pulsar name.
    """

    for item, name in zip(
        ["PSRJ", "PSRB", "PSR", "NAME"],
        ["J0123+1234", "B0124+12", "J0123+1234", "B0124+12"],
    ):
        psr = PulsarParametersPy()
        psr[item] = name

        assert get_psr_name(psr) == name


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


def test_initialise_ephemeris():
    """
    Test reading of ephemeris files.
    """

    with pytest.raises(ValueError):
        initialise_ephemeris(units="lhfld")

    with pytest.raises(IOError):
        initialise_ephemeris(
            earthfile="jksgdksg", sunfile="lhlbca", timefile="lshdldgls"
        )

    with pytest.raises(IOError):
        initialise_ephemeris(
            earthfile="jksgdksg", sunfile="lhlbca", timefile="lshdldgls"
        )

    with pytest.raises(IOError):
        initialise_ephemeris(timefile="lshdldgls")

    edat, tdat = initialise_ephemeris()

    assert edat.nentriesE == 175322
    assert edat.nentriesS == 17534
    assert edat.dtEtable == 7200.0
    assert edat.dtStable == 72000.0
    assert edat.etype == 2

    assert tdat.nentriesT == 87660
    assert tdat.dtTtable == 14400.0
