"""
Test code for known pulsar pipeline code.
"""

import os
import shutil
import subprocess as sp

import pytest
from astropy.utils.data import download_file
from cwinpy.knope import knope
from cwinpy.utils import LAL_EPHEMERIS_URL
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy


class TestKnope(object):
    """
    Test the knope setup.
    """

    @classmethod
    def setup_class(cls):
        # create some fake data frames using lalapps_Makefakedata_v5
        mfd = shutil.which("lalapps_Makefakedata_v5")

        cls.fakedatadir = "testing_fake_frame_cache"
        cls.fakedatadetectors = ["H1", "L1"]
        cls.fakedatachannels = [
            "{}:FAKE_DATA".format(det) for det in cls.fakedatadetectors
        ]
        cls.fakedatastarts = [1000000000]
        cls.fakedataduration = [86400]

        os.makedirs(cls.fakedatadir, exist_ok=True)

        cls.fakedatabandwidth = 8  # Hz
        sqrtSn = 5e-24  # noise amplitude spectral density
        cls.fakedataname = "FAKEDATA"

        # Create two pulsars to inject: one isolated and one binary
        cls.fakepulsarpar = []

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

        # pulsar
        f0 = 6.9456 / 2.0  # source rotation frequency (Hz)
        f1 = -9.87654e-11 / 2.0  # source rotational frequency derivative (Hz/s)
        f2 = 2.34134e-18 / 2.0  # second frequency derivative (Hz/s^2)
        alpha = 0.0  # source right ascension (rads)
        delta = 0.5  # source declination (rads)
        pepoch = 1000000000  # frequency epoch (GPS)

        # GW parameters
        h0 = 0.0  # GW amplitude (no signal)
        phi0 = 0.0  # GW initial phase (rads)
        cosiota = 0.0  # cosine of inclination angle
        psi = 0.0  # GW polarisation angle (rads)

        mfddic = {
            "alpha": alpha,
            "delta": delta,
            "f0": 2 * f0,
            "f1": 2 * f1,
            "f2": 2 * f2,
            "pepoch": pepoch,
            "h0": h0,
            "cosi": cosiota,
            "psi": psi,
            "phi0": phi0,
        }

        cls.fakepulsarpar.append(PulsarParametersPy())
        cls.fakepulsarpar[0]["PSRJ"] = "J0000+0000"
        cls.fakepulsarpar[0]["H0"] = h0
        cls.fakepulsarpar[0]["PHI0"] = phi0 / 2.0
        cls.fakepulsarpar[0]["PSI"] = psi
        cls.fakepulsarpar[0]["COSIOTA"] = cosiota
        cls.fakepulsarpar[0]["F"] = [f0, f1, f2]
        cls.fakepulsarpar[0]["RAJ"] = alpha
        cls.fakepulsarpar[0]["DECJ"] = delta
        cls.fakepulsarpar[0]["PEPOCH"] = pepoch
        cls.fakepulsarpar[0]["EPHEM"] = "DE405"
        cls.fakepulsarpar[0]["UNITS"] = "TDB"

        cls.fakepardir = "testing_fake_par_dir"
        os.makedirs(cls.fakepardir, exist_ok=True)
        cls.fakeparfile = []
        cls.fakeparfile.append(os.path.join(cls.fakepardir, "J0000+0000.par"))
        cls.fakepulsarpar[0].pp_to_par(cls.fakeparfile[-1])

        injfile = os.path.join(cls.fakepardir, "inj.dat")
        with open(injfile, "w") as fp:
            fp.write("[Pulsar 1]\n")
            fp.write(isolatedstr.format(**mfddic))
            fp.write("\n")

        # set ephemeris files
        efile = download_file(
            LAL_EPHEMERIS_URL.format("earth00-40-DE405.dat.gz"), cache=True
        )
        sfile = download_file(
            LAL_EPHEMERIS_URL.format("sun00-40-DE405.dat.gz"), cache=True
        )

        for j, datastart in enumerate(cls.fakedatastarts):
            for i in range(len(cls.fakedatachannels)):
                cmds = [
                    "-F",
                    cls.fakedatadir,
                    "--outFrChannels={}".format(cls.fakedatachannels[i]),
                    "-I",
                    cls.fakedatadetectors[i],
                    "--sqrtSX={0:.1e}".format(sqrtSn),
                    "-G",
                    str(datastart),
                    "--duration={}".format(cls.fakedataduration[j]),
                    "--Band={}".format(cls.fakedatabandwidth),
                    "--fmin",
                    "0",
                    '--injectionSources="{}"'.format(injfile),
                    "--outLabel={}".format(cls.fakedataname),
                    '--ephemEarth="{}"'.format(efile),
                    '--ephemSun="{}"'.format(sfile),
                ]

                # run makefakedata
                sp.run([mfd] + cmds)

    @classmethod
    def teardown_class(cls):
        """
        Remove test simulation directory.
        """

        shutil.rmtree(cls.fakepardir)
        shutil.rmtree(cls.fakedatadir)

    def test_knope_input_exceptions(self):
        """
        Test input exceptions are correctly returned.
        """

        with pytest.raises(TypeError):
            # test error if configuration file is not a string
            knope(heterodyne_config=2.3)

        with pytest.raises(ValueError):
            # test error if not configuration files are given
            knope()

        with pytest.raises(ValueError):
            # test error if no PE configuration is given
            knope(hetkwargs={"pulsarfiles": "dummyfile"})
