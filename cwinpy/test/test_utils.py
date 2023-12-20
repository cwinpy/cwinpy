"""
Test script for utils.py function.
"""

import os

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from cwinpy import PulsarParameters
from cwinpy.parfile import add_alias, get_real_param_from_alias, is_alias_param
from cwinpy.utils import (
    draw_ra_dec,
    ellipticity_to_q22,
    gcd_array,
    get_psr_name,
    initialise_ephemeris,
    int_to_alpha,
    is_par_file,
    is_valid_psr_name,
    logfactorial,
    overlap,
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
        psr = PulsarParameters()
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
        psr = PulsarParameters()
        psr[item] = name

        assert get_psr_name(psr) == name


def test_valid_psr_name():
    """
    Test check for valid pulsar names.
    """

    testnames = {
        "J0962-6529": True,
        "CRAB": True,  # crab is a default allowed name
        "B3485+89": False,  # RA outside range
        "B2182+89": True,
        "J0625-99": False,  # dec outside range
        "HWINJ01": True,  # hwinj is a default allowed name
        "JPULSAR00": True,
        "PSRJNAME": True,  # psr is a default allowed name
        "J0737-3039A": True,
        "B1259-28_other": True,  # stuff after name is allowed
        20: False,  # non-strings return False
        "adjfj*23*": False,
        "J": False,
    }

    for name, test in testnames.items():
        assert is_valid_psr_name(name) == test

    # different allowed names
    assert is_valid_psr_name("Matthew", allowed_names=["matthew"])

    # check when no other names are allowed
    assert not is_valid_psr_name("Crab", allowed_names=[])
    assert is_valid_psr_name("J0657+76", allowed_names=[])


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
    epsilon = q22_to_ellipticity(q22[0] * u.kg * u.m**2)

    assert np.isclose(epsilon, expected_epsilon[0])
    assert not hasattr(epsilon, "unit")


def test_initialise_ephemeris():
    """
    Test reading of ephemeris files.
    """

    with pytest.raises(IOError):
        initialise_ephemeris(units="lhfld")

    with pytest.raises(IOError):
        initialise_ephemeris(
            earthfile="jksgdksg",
            sunfile="lhlbca",
        )

    with pytest.raises(IOError):
        initialise_ephemeris(
            earthfile="jksgdksg",
            sunfile="lhlbca",
        )

    edat, tdat = initialise_ephemeris()

    assert edat.nentriesE == 175322
    assert edat.nentriesS == 17534
    assert edat.dtEtable == 7200.0
    assert edat.dtStable == 72000.0
    assert edat.etype == 2

    assert tdat.nentriesT == 87660
    assert tdat.dtTtable == 14400.0


def test_draw_ra_dec():
    """
    Test draw_ra_dec function.
    """

    # check exceptions
    with pytest.raises(TypeError):
        draw_ra_dec(eclhemi=1)

    with pytest.raises(TypeError):
        draw_ra_dec(eqhemi=1)

    with pytest.raises(ValueError):
        draw_ra_dec(eclhemi="blah")

    with pytest.raises(ValueError):
        draw_ra_dec(eqhemi="blah")

    # draw a single point
    radec = draw_ra_dec()

    assert np.shape(radec) == (2,)
    assert 0.0 <= radec[0] < 2.0 * np.pi
    assert -np.pi / 2.0 < radec[1] < np.pi / 2.0

    # draw multiple points
    radec = draw_ra_dec(100)

    assert np.shape(radec) == (2, 100)

    # check points are within expected ranges
    for ra, dec in np.array(radec).T:
        assert 0.0 <= ra < 2.0 * np.pi
        assert -np.pi / 2.0 < dec < np.pi / 2.0

    # draw multiple points from equatorial north
    radec = draw_ra_dec(100, eqhemi="north")
    for ra, dec in np.array(radec).T:
        assert 0.0 <= ra < 2.0 * np.pi
        assert 0.0 < dec < np.pi / 2.0

    # draw multiple points from equatorial south
    radec = draw_ra_dec(100, eqhemi="south")
    for ra, dec in np.array(radec).T:
        assert 0.0 <= ra < 2.0 * np.pi
        assert -np.pi / 2.0 < dec < 0.0

    # draw multiple points from ecliptic north
    radec = draw_ra_dec(100, eclhemi="north")
    for ra, dec in np.array(radec).T:
        pos = SkyCoord(ra, dec, unit="rad")
        lon = pos.barycentrictrueecliptic.lon.rad
        lat = pos.barycentrictrueecliptic.lat.rad
        assert 0.0 <= lon < 2.0 * np.pi
        assert 0.0 < lat < np.pi / 2.0

    # draw multiple points from ecliptic south
    radec = draw_ra_dec(100, eclhemi="south")
    for ra, dec in np.array(radec).T:
        pos = SkyCoord(ra, dec, unit="rad")
        lon = pos.barycentrictrueecliptic.lon.rad
        lat = pos.barycentrictrueecliptic.lat.rad
        assert 0.0 <= lon < 2.0 * np.pi
        assert -np.pi / 2.0 < lat < 0.0


def test_overlap():
    """
    Test heterodyned model overlap between two sky positions.
    """

    T = 86400 * 365.25  # one year observation
    f0 = 100.0  # frequency
    det = "H1"  # detector
    dt = 600  # time step

    # check same position gives overlap of 1
    pos1 = SkyCoord(2.3, 0.4, unit="rad")

    assert overlap(pos1, pos1, f0, T, det=det, dt=dt) == 1.0

    # check for small overlap at offset sky position
    pos2 = SkyCoord(2.3 + 0.1, 0.4 - 0.1, unit="rad")

    assert overlap(pos1, pos2, f0, T, det=det, dt=dt) < 1e-3


def test_PulsarParameters_aliases():
    """
    Test PulsarParameters aliases and related functions
    """

    # test is_alias_param()
    assert is_alias_param("ALIAS_PAR")
    assert not is_alias_param("PAR")

    # test add_alias() and get_real_param_from_alias()
    # - ALIAS_HF0 is an alias for half the value of F0
    add_alias("ALIAS_HF0", lambda pp: 0.5 * pp["F0"], "F0", lambda hf0, pp: 2 * hf0)
    assert get_real_param_from_alias("ALIAS_HF0") == "F0"
    assert get_real_param_from_alias("F0") == "F0"

    # test getting and setting alias parameters
    pp = PulsarParameters()
    # - F0 has not been set yet, should raise an error
    with pytest.raises(TypeError):
        pp["ALIAS_HF0"]
    # - test setting alias and getting real parameter
    pp["ALIAS_HF0"] = 50
    assert pp["F0"] == 100
    # - test setting real parameter and getting alias
    pp["F0"] = 1000
    assert pp["ALIAS_HF0"] == 500

    # test braking index alias
    # - required real parameters have not been set yet, should raise an error
    with pytest.raises(TypeError):
        pp["ALIAS_N"]
    # - check definition is correct
    pp["F1"] = -1e-8
    pp["ALIAS_N"] = 5
    assert np.isclose(pp["F2"], pp["ALIAS_N"] * pp["F1"] ** 2 / pp["F0"])
