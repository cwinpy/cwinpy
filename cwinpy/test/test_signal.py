"""
Test the HeterodynedCWSimulator in the signal module.


For comparison quasi-independent model heterodyned signals have been produced
using the heterodyned_pulsar_signal function in lalpulsar:

from lalpulsar.pulsarpputils import heterodyned_pulsar_signal
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
import shutil
import subprocess as sp
from copy import deepcopy

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from solar_system_ephemerides.paths import body_ephemeris_path

from cwinpy import PulsarParameters
from cwinpy.data import HeterodynedData
from cwinpy.heterodyne import heterodyne
from cwinpy.signal import HeterodynedCWSimulator


def mismatch(model1, model2):
    """
    Compute the mismatch between two models.
    """

    return 1.0 - np.abs(np.vdot(model1, model2) / np.vdot(model1, model1))


class TestSignal:
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


class TestPhaseOnly:
    """
    Test the simulator when it just outputs phase.
    """

    def test_slow_signal(self):
        """
        Test a slowly varying signal at 0.0001 Hz for the correct phase.
        """

        times = np.arange(1000000000.0, 1008640000.0, 10000, dtype=np.float128)

        par = PulsarParameters()
        par["F"] = [0.0001]
        par["PEPOCH"] = 55818.0  # roughly GPS time of 1000000000

        # set pulsar at roughly the ecliptic pole
        par["DECJ"] = 1.16
        par["RAJ"] = 4.71

        pulsar = HeterodynedCWSimulator(par=par, det="H1", times=times)

        phase = pulsar.model(freqfactor=1.0, phase_only=True)
        phase -= phase[0]

        assert np.allclose(phase, np.zeros_like(phase), atol=1e-4)

    def test_signal_frequency(self):
        """
        Calculate the signal frequency using the derivative of the phase and
        check it matches the expected frequency.
        """

        dt = 0.01
        duration = 86400 * 365  # one year
        times = np.arange(1000000000.0, 1000000000 + duration, 10, dtype=np.float128)
        times2 = times + dt

        par = PulsarParameters()
        par["F"] = [1.0]
        par["PEPOCH"] = 55818.0  # roughly GPS time of 1000000000

        # set pulsar near ecliptic plane (first point of Aries)
        par["DECJ"] = 0.0
        par["RAJ"] = 0.0

        pulsar = HeterodynedCWSimulator(par=par, det="L1", times=times)
        pulsar2 = HeterodynedCWSimulator(par=par, det="L1", times=times2)

        phase1 = pulsar.model(freqfactor=1.0, phase_only=True)

        # use phase_evolution method interface as test
        phase2 = pulsar2.phase_evolution(freqfactor=1.0)

        freqs = np.mod(phase2 - phase1, 1.0) / dt
        maxdfplane = np.max(np.abs(freqs - par["F0"]))

        # approximate maximum Doppler shift from Earth orbit + rotation
        dfmax = par["F0"] * ((29.78e3 + 460) / 3e8)

        # make sure frequencies are below max Doppler shift (with 10% leeway)
        assert maxdfplane < 1.1 * dfmax

        # set pulsar near ecliptic pole
        par["DECJ"] = 1.16
        par["RAJ"] = 4.71

        pulsar = HeterodynedCWSimulator(par=par, det="L1", times=times)
        pulsar2 = HeterodynedCWSimulator(par=par, det="L1", times=times2)

        # use call method
        phase1 = pulsar(freqfactor=1.0, phase_only=True)
        phase2 = pulsar2.phase_evolution(freqfactor=1.0)

        freqs = np.mod(phase2 - phase1, 1.0) / dt
        maxdfpole = np.max(np.abs(freqs - par["F0"]))

        # make sure frequencies are below 20% of max Doppler shift
        assert maxdfpole < 0.2 * dfmax
        assert maxdfpole < maxdfplane

    def test_tempo2(self):
        """
        Test the phase only when calculated with Tempo2 vs LAL.
        """

        times = np.arange(1000000000.0, 1000086400.0, 1, dtype=np.float128)

        par = PulsarParameters()
        par["F"] = [0.132, -4.5e-10]
        par["PEPOCH"] = 55000.0

        # set pulsar at roughly the ecliptic pole
        par["DECJ"] = 1.16
        par["RAJ"] = 4.71

        pulsar_lal = HeterodynedCWSimulator(par=par, det="H1", times=times)
        pulsar_tempo = HeterodynedCWSimulator(par, "H1", times=times, usetempo2=True)

        phase_lal = 2 * np.pi * pulsar_lal.phase_evolution(freqfactor=1.0)
        phase_tempo2 = 2 * np.pi * pulsar_tempo.phase_evolution(freqfactor=1.0)

        assert np.allclose(np.sin(phase_lal), np.sin(phase_tempo2), atol=1e-5)
        assert np.allclose(np.cos(phase_lal), np.cos(phase_tempo2), atol=1e-5)


class TestHeterodyneRefFreq(object):
    """
    Test heterodyning the data at a reference frequency.
    """

    @classmethod
    def setup_class(cls):
        # create dummy frame cache files
        cls.dummydir = "testing_frame_cache"
        os.makedirs(cls.dummydir, exist_ok=True)
        cls.dummy_cache_files = []
        for i in range(0, 5):
            dummyfile = os.path.join(
                cls.dummydir, f"frame_cache_{i:01d}.cache"  # noqa: E231
            )
            cls.dummy_cache_files.append(dummyfile)
            with open(dummyfile, "w") as fp:
                fp.write("blah\n")

        # create some fake data frames using lalpulsar_Makefakedata_v5
        mfd = shutil.which("lalpulsar_Makefakedata_v5")

        cls.fakedatadir = "testing_fake_frame_cache"
        cls.fakedatadetector = "H1"
        cls.fakedatachannel = f"{cls.fakedatadetector}:FAKE_DATA"  # noqa: E231
        cls.fakedatastart = 1000000000
        cls.fakedataduration = 86400

        os.makedirs(cls.fakedatadir, exist_ok=True)

        cls.fakedatabandwidth = 8  # Hz
        sqrtSn = 1e-29  # noise amplitude spectral density
        cls.fakedataname = "FAKEDATA"

        # Create pulsars to inject
        cls.fakepulsarpar = PulsarParameters()

        # requirements for Makefakedata pulsar input files
        isolatedstr = """\
Alpha = {alpha}
Delta = {delta}
Freq = {f0}
f1dot = {f1}
f2dot = {f2}
refTime = {pepoch}
h0 = {h0}
cosi = {cosi}
psi = {psi}
phi0 = {phi0}
"""

        # FIRST PULSAR (ISOLATED)
        f0 = 6.9456 / 2.0  # source rotation frequency (Hz)
        f1 = -9.87654e-11 / 2.0  # source rotational frequency derivative (Hz/s)
        f2 = 2.34134e-18 / 2.0  # second frequency derivative (Hz/s^2)
        f3 = -2.1336e-26 / 2.0  # third frequency derivative (Hz/s^3)
        alpha = 0.0  # source right ascension (rads)
        delta = 0.0  # source declination (rads)
        pepoch = 1000000000  # frequency epoch (GPS)

        # GW parameters
        h0 = 3.0e-24  # GW amplitude
        phi0 = 1.0  # GW initial phase (rads)
        cosiota = 0.1  # cosine of inclination angle
        psi = 0.5  # GW polarisation angle (rads)

        mfddic = {
            "alpha": alpha,
            "delta": delta,
            "f0": 2 * f0,
            "f1": 2 * f1,
            "f2": 2 * f2,
            "f3": 2 * f3,
            "pepoch": pepoch,
            "h0": h0,
            "cosi": cosiota,
            "psi": psi,
            "phi0": phi0,
        }

        cls.fakepulsarpar["PSRJ"] = "J0000+0000"
        cls.fakepulsarpar["H0"] = h0
        cls.fakepulsarpar["PHI0"] = phi0 / 2.0
        cls.fakepulsarpar["PSI"] = psi
        cls.fakepulsarpar["COSIOTA"] = cosiota
        cls.fakepulsarpar["F"] = [f0, f1, f2, f3]
        cls.fakepulsarpar["RAJ"] = alpha
        cls.fakepulsarpar["DECJ"] = delta
        cls.fakepulsarpar["PEPOCH"] = pepoch
        cls.fakepulsarpar["EPHEM"] = "DE405"
        cls.fakepulsarpar["UNITS"] = "TDB"

        cls.ref_heterodyne = PulsarParameters()
        cls.ref_heterodyne["PSRJ"] = "J0000+0000"
        cls.ref_heterodyne["F"] = [np.ceil(f0 * 2) / 2]
        cls.ref_heterodyne["RAJ"] = alpha
        cls.ref_heterodyne["DECJ"] = delta
        cls.ref_heterodyne["PEPOCH"] = pepoch - 100 * 86400

        cls.fakepardir = "testing_fake_par_dir"
        os.makedirs(cls.fakepardir, exist_ok=True)
        cls.fakeparfile = os.path.join(cls.fakepardir, "J0000+0000.par")
        cls.fakepulsarpar.pp_to_par(cls.fakeparfile)

        cls.refparfile = os.path.join(cls.fakepardir, "J0000+0000_ref.par")
        cls.ref_heterodyne.pp_to_par(cls.refparfile)

        injfile = os.path.join(cls.fakepardir, "inj.dat")
        with open(injfile, "w") as fp:
            fp.write("[Pulsar 1]\n")
            fp.write(isolatedstr.format(**mfddic))
            fp.write("\n")

        # set ephemeris files
        efile = body_ephemeris_path(body="earth", jplde="DE405", string=True)
        sfile = body_ephemeris_path(body="sun", jplde="DE405", string=True)

        cmds = [
            "-F",
            cls.fakedatadir,
            f"--outFrChannels={cls.fakedatachannel}",
            "-I",
            cls.fakedatadetector,
            "--sqrtSX={0:.1e}".format(sqrtSn),
            "-G",
            str(cls.fakedatastart),
            f"--duration={cls.fakedataduration}",
            f"--Band={cls.fakedatabandwidth}",
            "--fmin",
            "0",
            f'--injectionSources="{injfile}"',
            f"--outLabel={cls.fakedataname}",
            f'--ephemEarth="{efile}"',
            f'--ephemSun="{sfile}"',
        ]

        # run makefakedata
        sp.run([mfd] + cmds)

    @classmethod
    def teardown_class(cls):
        """
        Remove test simulation directory.
        """

        shutil.rmtree(cls.dummydir)
        shutil.rmtree(cls.fakepardir)
        shutil.rmtree(cls.fakedatadir)

    def test_ref_freq_heterodyne(self):
        segments = [
            (self.fakedatastart, self.fakedatastart + self.fakedataduration),
        ]

        outdir = os.path.join(self.fakedatadir, "heterodyne_output")

        # heterodyne the data at a fixed frequency
        H = heterodyne(
            starttime=segments[0][0],
            endtime=segments[-1][-1],
            pulsarfiles=self.refparfile,
            segmentlist=segments,
            framecache=self.fakedatadir,
            channel=self.fakedatachannel,
            freqfactor=2,
            stride=86400,
            output=outdir,
            resamplerate=1,
            includessb=False,
        )

        het = HeterodynedData(H.outputfiles[self.ref_heterodyne["PSRJ"]])

        for usetempo2 in [False, True]:
            # calculate the phase difference compared to the reference phase
            phase_diff = HeterodynedCWSimulator(
                times=het.times,
                det=self.fakedatadetector,
                units=self.fakepulsarpar["UNITS"],
                ephem=self.fakepulsarpar["EPHEM"],
                ref_freq=self.ref_heterodyne["F0"],
                ref_epoch=self.ref_heterodyne["PEPOCH"],
                usetempo2=usetempo2,
            ).phase_evolution(newpar=self.fakepulsarpar)

            # manually heterodyne the data
            bm = het.heterodyne(-2 * np.pi * phase_diff)

            # get the theoretical heterodyned model
            b = HeterodynedCWSimulator(
                par=self.fakepulsarpar,
                times=bm.times,
                det=self.fakedatadetector,
            )()

            # compare absolute model values
            # - ignore last few points due to filter impulse response
            # - use absolute values as initial phase will be offset due
            #   to correcting to the reference frequency epoch rather
            #   than the signal epoch
            assert np.allclose(
                np.abs(b[:-4]),
                np.abs(bm.data[:-4]),
                atol=self.fakepulsarpar["H0"] / 250,
            )
