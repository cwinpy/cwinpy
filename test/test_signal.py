"""
Test the HeterodynedCWSimulator in the signal module.


For comparison quasi-independent model heterodyned signals have been produced
using the heterodyned_pulsar_signal function in lalapps:

from lalapps.pulsarpputils import heterodyned_pulsar_signal
from astropy.coordinates import SkyCoord
import numpy as np

pos = SkyCoord("01:23:34.5 -45:01:23.4", unit=("hourangle", "deg"))
pardict = {
    "C22": 5.6e-26,
    "C21": 3.4e-26,
    "cosiota": -0.9,
    "psi": 0.4,
    "phi21": 2.3,
    "phi22": 0.3,
    "ra": pos.ra.rad,
    "dec": pos.dec.rad,
}

times = np.arange(1000000000.0, 1000086400.0, 3600)

for det in ["H1", "L1", "V1", "G1"]:
    signal = heterodyned_pulsar_signal(
        pardict,
        det,
        datatimes=times,
    )

    for i in range(2):
        np.savetxt(f"test_signal_model_{det}_{i+1}.txt.gz", signal[1][i].view(np.float32).reshape(-1, 2))
"""

import os

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

        # set comparison signal parameters
        cls.comparison = PulsarParametersPy()
        pos = SkyCoord("01:23:34.5 -45:01:23.4", unit=("hourangle", "deg"))
        cls.comparison["RAJ"] = pos.ra.rad
        cls.comparison["DECJ"] = pos.dec.rad
        cls.comparison["F"] = [
            123.456789,
            -9.87654321e-12,
        ]  # frequency and first derivative
        cls.comparison["PEPOCH"] = Time(
            58000, format="mjd", scale="tt"
        ).gps  # frequency epoch
        cls.comparison["C22"] = 5.6e-26  # GW amplitude
        cls.comparison["COSIOTA"] = -0.9  # cosine of inclination angle
        cls.comparison["PSI"] = 0.4  # polarization angle (rads)
        cls.comparison["PHI21"] = 2.3  # initial phase (rads)
        cls.comparison["PHI22"] = 0.3
        cls.comparison["C21"] = 5.6e-26

        # read in comparitor files
        cls.compare2f = {}
        cls.compare1f = {}
        for det in cls.detectors:
            cls.compare1f[det] = (
                np.loadtxt(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "data",
                        f"test_signal_model_{det}_1.txt.gz",
                    )
                )
                .view(complex)
                .reshape(-1)
            )

            cls.compare2f[det] = (
                np.loadtxt(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "data",
                        f"test_signal_model_{det}_2.txt.gz",
                    )
                )
                .view(complex)
                .reshape(-1)
            )

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

        for det in self.detectors:
            het = HeterodynedCWSimulator(par=self.comparison, det=det, times=self.times)
            model = het.model(usephase=True, freqfactor=2)

            assert len(model) == len(self.times)
            assert model.dtype == complex
            assert np.allclose(model, self.compare2f[det])

    def test_dc_signal_1f(self):
        """
        Test creating signals from a source emitting l=2, m=1 modes heterodyned
        so that the signal only varies due to the antenna response of the
        detector.
        """

        for det in self.detectors:
            het = HeterodynedCWSimulator(par=self.comparison, det=det, times=self.times)
            model = het.model(usephase=True, freqfactor=1)

            assert len(model) == len(self.times)
            assert model.dtype == complex
            assert np.allclose(model, self.compare1f[det])
