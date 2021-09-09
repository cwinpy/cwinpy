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
from copy import deepcopy

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from cwinpy import PulsarParameters
from cwinpy.signal import HeterodynedCWSimulator


def mismatch(model1, model2):
    """
    Compute the mismatch between two models.
    """

    return 1.0 - np.abs(np.vdot(model1, model2) / np.vdot(model1, model1))


class TestSignal(object):
    """
    Test the HeterodynedCWSimulator class.
    """

    @classmethod
    def setup_class(cls):
        # set observation times
        cls.timesfixed = np.arange(1000000000.0, 1000086400.0, 3600, dtype=np.float128)
        cls.timesvary = cls.timesfixed + 87789.0

        # set detectors
        cls.detectors = ["H1", "L1", "V1", "G1"]

        # set comparison signal parameters
        cls.comparison = PulsarParameters()
        pos = SkyCoord("01:23:34.5 -45:01:23.4", unit=("hourangle", "deg"))
        cls.comparison["PSRJ"] = "J0123-4501"
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

        cls.comparison["EPHEM"] = "DE421"
        cls.comparison["UNITS"] = "TCB"

        # binary parameters
        cls.comparison["BINARY"] = "BT"
        cls.comparison["ECC"] = 0.1
        cls.comparison["A1"] = 1.5
        cls.comparison["T0"] = 1000000000.0 + 86400 / 2
        cls.comparison["OM"] = 0.4
        cls.comparison["PB"] = 0.1 * 86400

        # FITWAVES parameters
        cls.comparison["WAVEEPOCH"] = 999998009.0
        cls.comparison["WAVE_OM"] = 0.00279
        cls.comparison["WAVESIN"] = [-6.49330613e02, 7.78086689e01]
        cls.comparison["WAVECOS"] = [-9.42072776e02, -1.40573292e02]

        # glitch parameters
        cls.comparison["GLEP"] = [1000086400.0 + 10785.0]
        cls.comparison["GLPH"] = [0.4]
        cls.comparison["GLF0"] = [3.6e-6]
        cls.comparison["GLF1"] = [-5.3e-14]
        cls.comparison["GLF0D"] = [7.9e-7]
        cls.comparison["GLTD"] = [86400 / 2]

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

        # set offset parameters
        cls.offset = deepcopy(cls.comparison)

        # offset frequency
        cls.offset["F"] = [
            cls.comparison["F0"] + 2e-4,
            cls.comparison["F1"] - 1e-13,
        ]

        # offset the sky positions
        cls.offset["RAJ"] = pos.ra.rad + 0.0005
        cls.offset["DECJ"] = pos.dec.rad - 0.0005

        # offset the binary parameters
        cls.offset["OM"] = cls.comparison["OM"] + 0.003
        cls.offset["ECC"] = cls.comparison["ECC"] + 0.00001
        cls.offset["PB"] = cls.comparison["PB"] + 0.0001 * 86400
        cls.offset["A1"] = cls.comparison["A1"] + 0.003

        # offset FITWAVES parameters
        cls.offset["WAVE_OM"] = cls.comparison["WAVE_OM"] + 0.0001
        cls.offset["WAVESIN"] = [
            cls.comparison["WAVESIN"][0] + 2,
            cls.comparison["WAVESIN"][1] + 3,
        ]
        cls.offset["WAVECOS"] = [
            cls.comparison["WAVECOS"][0] + 2,
            cls.comparison["WAVECOS"][1] + 3,
        ]

        # offset glitch parameters
        cls.offset["GLPH"] = [cls.comparison["GLPH"][0] + 0.1]
        cls.offset["GLF0"] = [cls.comparison["GLF0"][0] + 2e-7]

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
            het = HeterodynedCWSimulator(
                par=self.comparison, det=det, times=self.timesfixed
            )
            model = het.model(freqfactor=2)

            assert len(model) == len(self.timesfixed)
            assert model.dtype == complex
            assert np.allclose(model, self.compare2f[det])

    def test_dc_signal_1f(self):
        """
        Test creating signals from a source emitting l=2, m=1 modes heterodyned
        so that the signal only varies due to the antenna response of the
        detector.
        """

        for det in self.detectors:
            het = HeterodynedCWSimulator(
                par=self.comparison, det=det, times=self.timesfixed
            )
            model = het.model(freqfactor=1)

            assert len(model) == len(self.timesfixed)
            assert model.dtype == complex
            assert np.allclose(model, self.compare1f[det])

    def test_offset_signal(self):
        """
        Test signals generated with an offset of parameters using the LAL and
        TEMPO2 options.
        """

        for det in self.detectors:
            # using LAL routines
            het = HeterodynedCWSimulator(
                par=self.comparison, det=det, times=self.timesvary
            )
            modellal = het.model(newpar=self.offset, freqfactor=2, updateSSB=True)

            # using TEMPO2 routines
            hettempo = HeterodynedCWSimulator(
                par=self.comparison, det=det, times=self.timesvary, usetempo2=True
            )
            modeltempo = hettempo.model(newpar=self.offset, freqfactor=2)

            assert len(modellal) == len(self.timesvary)
            assert len(modeltempo) == len(self.timesvary)

            # test mismatch between using LAL and TEMPO
            assert mismatch(modellal, modeltempo) < 5e-5

            # test angle between models (we expected there to be a constant phase shift
            # between the TEMPO2 and LAL calculations due to binary system signals not
            # being referenced as expected)
            # NOTE: the test tolerances are higher that generally needed due to one
            # time stamp for H1 that is more significantly different than for the
            # other detectors - this is probably a Shapiro delay difference that has
            # particular effect on the line of site for this source and that detector
            phasediff = np.mod(
                np.angle(modellal, deg=True) - np.angle(modeltempo, deg=True), 360
            )
            assert np.std(phasediff) < 0.01
            assert np.max(phasediff) - np.min(phasediff) < 0.1
