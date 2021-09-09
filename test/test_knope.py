"""
Test code for known pulsar pipeline code.
"""

import os
import shutil
import subprocess as sp

import numpy as np
import pytest
from astropy.utils.data import download_file
from bilby.core.prior import PriorDict, Uniform
from cwinpy import HeterodynedData, PulsarParameters
from cwinpy.knope import knope
from cwinpy.utils import LAL_EPHEMERIS_URL


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

        cls.fakepulsarpar.append(PulsarParameters())
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

    def test_knope(self):
        """
        Test full knope pipeline.
        """

        segments = [
            (self.fakedatastarts[i], self.fakedatastarts[i] + self.fakedataduration[i])
            for i in range(len(self.fakedatastarts))
        ]

        fulloutdir = os.path.join(self.fakedatadir, "full_heterodyne_output")

        # USING KEYWORD ARGUMENTS

        # heterodyned keyword arguments
        hetkwargs = dict(
            starttime=segments[0][0],
            endtime=segments[-1][-1],
            pulsarfiles=self.fakeparfile,
            segmentlist=segments,
            framecache=self.fakedatadir,
            channel=self.fakedatachannels[0],
            freqfactor=2,
            stride=86400 // 2,
            resamplerate=1 / 60,
            includessb=True,
            output=fulloutdir,
            label="heterodyne_kwargs_{psr}_{det}_{freqfactor}.hdf5",
        )

        # parameter estimation keyword arguments
        prior = PriorDict({"h0": Uniform(name="h0", minimum=0.0, maximum=5e-25)})
        pekwargs = dict(
            prior=prior, grid=True, grid_kwargs={"grid_size": 100, "label": "pe_kwargs"}
        )

        # run knope
        hetkw, perunkw = knope(hetkwargs=hetkwargs, pekwargs=pekwargs)

        # USING CONFIGURATION FILES
        hetconfigstr = (
            "detector = {}\n"
            "starttime = {}\n"
            "endtime = {}\n"
            "pulsarfiles = {}\n"
            "framecache = {}\n"
            "channel = {}\n"
            'segmentlist = "{}"\n'
            "output = {}\n"
            "stride = {}\n"
            "freqfactor = {}\n"
            "resamplerate = {}\n"
            "includessb = {}\n"
            "label = heterodyne_config_{{psr}}_{{det}}_{{freqfactor}}.hdf5\n"
        )

        hetconfigfile = "hetconfig.ini"
        with open(hetconfigfile, "w") as fp:
            fp.write(
                hetconfigstr.format(
                    self.fakedatadetectors[0],
                    hetkwargs["starttime"],
                    hetkwargs["endtime"],
                    hetkwargs["pulsarfiles"][0],
                    hetkwargs["framecache"],
                    hetkwargs["channel"],
                    hetkwargs["segmentlist"],
                    hetkwargs["output"],
                    hetkwargs["stride"],
                    hetkwargs["freqfactor"],
                    hetkwargs["resamplerate"],
                    hetkwargs["includessb"],
                )
            )

        prior.to_file(outdir=".", label="knope_test")
        peconfigstr = (
            "prior = knope_test.prior\n"
            "grid = {}\n"
            "grid_kwargs = {}\n"
            "label = pe_config\n"
        )

        peconfigfile = "peconfig.ini"
        pekwargs["grid_kwargs"]["label"] = "pe_config"
        with open(peconfigfile, "w") as fp:
            fp.write(
                peconfigstr.format(
                    pekwargs["grid"],
                    pekwargs["grid_kwargs"],
                )
            )

        # run knope
        hetcon, peruncon = knope(
            heterodyne_config=hetconfigfile, pe_config=peconfigfile
        )

        # check for consistent heterodyne outputs
        hkw = HeterodynedData.read(
            hetkw[2.0][0].outputfiles[self.fakepulsarpar[0]["PSRJ"]]
        )
        hc = HeterodynedData.read(
            hetcon[2.0][0].outputfiles[self.fakepulsarpar[0]["PSRJ"]]
        )

        assert np.array_equal(hkw.times.value, hc.times.value)
        assert np.array_equal(hkw.data, hc.data)

        # check for consistent parameter estimation outputs
        assert (
            perunkw[self.fakepulsarpar[0]["PSRJ"]].grid.ln_evidence
            == peruncon[self.fakepulsarpar[0]["PSRJ"]].grid.ln_evidence
        )
        assert np.array_equal(
            perunkw[self.fakepulsarpar[0]["PSRJ"]].grid.mesh_grid[0],
            peruncon[self.fakepulsarpar[0]["PSRJ"]].grid.mesh_grid[0],
        )
        assert np.array_equal(
            perunkw[self.fakepulsarpar[0]["PSRJ"]].grid.ln_posterior,
            peruncon[self.fakepulsarpar[0]["PSRJ"]].grid.ln_posterior,
        )

        os.remove("knope_test.prior")
        os.remove(hetconfigfile)
        os.remove(peconfigfile)
