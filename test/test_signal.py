"""
Test the HeterodynedCWSimulator in the signal module.
"""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from cwinpy.signal import HeterodynedCWSimulator
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy


class TestSignal(object):
    """
    Test the HeterodynedCWSimulator class.
    """

    @classmethod
    def setup_class(cls):
        # set observation times
        cls.times = np.arange(1000000000.0, 1000086400.0, 3600, dtype=np.float128)

        # set detectors
        cls.detectors = ["H1", "L1", "V1", "G1"]

    def test_bad_par_file(self):
        """
        Test correct exceptions raised for invalid parameter files.
        """

        with pytest.raises(TypeError):
            # pass a float rather than a string
            _ = HeterodynedCWSimulator(par=4.5, det="H1")

        with pytest.raises(IOError):
            # pass non-existant file string
            _ = HeterodynedCWSimulator(par="blah.par", det="H1")

    def test_dc_signal(self):
        """
        Test creating signals from a triaxial source heterodyned so that the
        signal only varies due to the antenna response of the detector.
        """

        par = PulsarParametersPy()
        pos = SkyCoord("01:23:34.5 -45:01:23.4", unit=("hourangle", "deg"))
        par["RAJ"] = pos.ra.rad
        par["DECJ"] = pos.dec.rad
        par["F"] = [123.456789, -9.87654321e-12]  # frequency and first derivative
        par["PEPOCH"] = Time(58000, format="mjd", scale="tt").gps  # frequency epoch
        par["H0"] = 5.6e-26  # GW amplitude
        par["COSIOTA"] = -0.9  # cosine of inclination angle
        par["PSI"] = 0.4  # polarization angle (rads)
        par["PHI0"] = 2.3  # initial phase (rads)

        for det in self.detectors:
            het = HeterodynedCWSimulator(par=par, det=det, times=self.times)
            model = het.model(usephase=True)

            assert len(model) == len(self.times)
            assert model.dtype == complex
            assert np.all(np.abs(model) < par["H0"])

    def test_dc_signal_1f(self):
        """
        Test creating signals from a source emitting l=2, m=1 modes heterodyned
        so that the signal only varies due to the antenna response of the
        detector.
        """

        par = PulsarParametersPy()
        pos = SkyCoord("01:23:34.5 -45:01:23.4", unit=("hourangle", "deg"))
        par["RAJ"] = pos.ra.rad
        par["DECJ"] = pos.dec.rad
        par["F"] = [123.456789, -9.87654321e-12]  # frequency and first derivative
        par["PEPOCH"] = Time(58000, format="mjd", scale="tt").gps  # frequency epoch
        par["C21"] = 5.6e-26  # GW amplitude
        par["COSIOTA"] = -0.9  # cosine of inclination angle
        par["PSI"] = 0.4  # polarization angle (rads)
        par["PHI0"] = 2.3  # initial phase (rads)

        for det in self.detectors:
            het = HeterodynedCWSimulator(par=par, det=det, times=self.times)
            model = het.model(usephase=True, freqfactor=1)

            assert len(model) == len(self.times)
            assert model.dtype == complex
            assert np.all(np.abs(model) < par["C21"] / 4)
