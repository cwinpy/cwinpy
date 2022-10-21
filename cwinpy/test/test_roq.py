"""
Test script for ROQ usage.
"""

import copy
import os
import shutil
import subprocess as sp

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from bilby.core.prior import PriorDict, Sine, Uniform
from cwinpy.data import HeterodynedData, MultiHeterodynedData
from cwinpy.heterodyne import heterodyne
from cwinpy.parfile import PulsarParameters
from cwinpy.pe import pe
from cwinpy.pe.likelihood import TargetedPulsarLikelihood
from cwinpy.pe.roq import GenerateROQ
from cwinpy.utils import LAL_EPHEMERIS_URL, download_ephemeris_file, logfactorial


def full_log_likelihood(model, data, like="studentst", sigma=1.0):
    """
    Calculate the likelihood for the full data set.
    """

    dd = np.vdot(data, data).real
    mm = np.vdot(model, model).real
    dm = np.vdot(data, model).real

    chisq = (dd - 2.0 * dm + mm) / sigma**2

    if like == "studentst":
        return (
            logfactorial(len(data) - 1)
            - np.log(2.0)
            - len(data) * (np.log(np.pi * chisq))
        )
    else:
        N = len(data) if data.dtype == complex else len(data) / 2
        return -0.5 * chisq - N * np.log(2.0 * np.pi * sigma**2)


class TestGenericModelROQ:
    """
    Test the reduced order quadrature for a generic model function.
    """

    @staticmethod
    def generic_complex_model(t, A=0, phi0=0, m=0, c=0):
        """
        A (complex) sinusoid and a straight line model.
        """

        return A * np.exp(2.0 * np.pi * 1.4 * t * 1j + phi0) + m * t + c

    @staticmethod
    def generic_real_model(t, A=0, phi0=0, m=0, c=0):
        """
        A sinusoid and a straight line model.
        """

        return A * np.sin(2.0 * np.pi * 1.4 * t + phi0) + m * t + c

    @classmethod
    def setup_class(cls):
        # create some complex data using generic_complex_model
        cls.A_true = 1.2
        cls.phi0_true = 2.3
        cls.m_true = 0.7
        cls.c_true = -1.2

        cls.N = 100
        cls.times = np.linspace(0, 5, cls.N)

        cls.comp_model = cls.generic_complex_model(
            cls.times,
            A=cls.A_true,
            phi0=cls.phi0_true,
            m=cls.m_true,
            c=cls.c_true,
        )

        cls.sigma = 1.5

        cls.comp_data = cls.comp_model + (
            np.random.normal(loc=0.0, scale=cls.sigma, size=cls.N)
            + 1j * np.random.normal(loc=0.0, scale=cls.sigma, size=cls.N)
        )

        cls.real_model = cls.generic_real_model(
            cls.times,
            A=cls.A_true,
            phi0=cls.phi0_true,
            m=cls.m_true,
            c=cls.c_true,
        )

        cls.real_data = cls.real_model + np.random.normal(
            loc=0.0, scale=cls.sigma, size=cls.N
        )

        cls.priors = PriorDict()
        cls.priors["A"] = Uniform(0, 10, name="A")
        cls.priors["phi0"] = Uniform(0, 2.0 * np.pi, name="phi0")
        cls.priors["m"] = Uniform(-5, 5, name="m")
        cls.priors["c"] = Uniform(-5, 5, name="c")

    def test_exceptions(self):
        """
        Test exceptions for GenerateROQ class.
        """

        # x-value exceptions
        with pytest.raises(TypeError):
            GenerateROQ(self.real_data, "sdjf", self.priors)

        with pytest.raises(ValueError):
            GenerateROQ(self.real_data, np.array([[1.2, 2.3], [2.3, 4.5]]), self.priors)

        with pytest.raises(TypeError):
            GenerateROQ(self.real_data, ["1", "2"], self.priors)

        # data exceptions
        with pytest.raises(TypeError):
            GenerateROQ(1.2, self.times, self.priors)

        with pytest.raises(ValueError):
            GenerateROQ([[1.0, 2.0], [3.0, 4.0]], self.times, self.priors)

        with pytest.raises(TypeError):
            GenerateROQ(["1", "2"], self.times, self.priors)

        with pytest.raises(ValueError):
            GenerateROQ([1.0, 2.0], self.times, self.priors)

        with pytest.raises(TypeError):
            GenerateROQ(self.real_data, self.times, self.priors, sigma="blah")

        with pytest.raises(ValueError):
            GenerateROQ(self.real_data, self.times, self.priors, sigma=[-1.0, -1.0])

        # prior exceptions
        with pytest.raises(TypeError):
            GenerateROQ(self.real_data, self.times, 1.0)

        # model exceptions
        with pytest.raises(TypeError):
            GenerateROQ(self.real_data, self.times, self.priors, model=1.0)

    def test_real_model_roq(self):
        ntraining = 500

        # generate ROQ
        roq = GenerateROQ(
            self.real_data,
            self.times,
            self.priors,
            model=self.generic_real_model,
            store_training_data=True,
            ntraining=ntraining,
            sigma=self.sigma,
        )

        assert roq.training_data.shape == (ntraining, self.N)
        assert roq.nbases > 0
        assert roq.nbases2 > 0

        # check likelihood calculation
        Ntests = 100

        for _ in range(Ntests):
            # draw values from prior
            values = self.priors.sample()

            # students-t likelihood
            ll = roq.log_likelihood(**values)
            fullll = full_log_likelihood(
                self.generic_real_model(self.times, **values), self.real_data
            )

            assert np.abs(ll - fullll) < 1e-6

            # Gaussian likelihood
            ll = roq.log_likelihood(**values, likelihood="gaussian")
            fullll = full_log_likelihood(
                self.generic_real_model(self.times, **values),
                self.real_data,
                like="gaussian",
                sigma=self.sigma,
            )

            assert np.abs(ll - fullll) < 1e-6

    def test_complex_model_roq(self):
        ntraining = 500

        # generate ROQ
        roq = GenerateROQ(
            self.comp_data,
            self.times,
            self.priors,
            model=self.generic_complex_model,
            store_training_data=True,
            ntraining=ntraining,
            sigma=self.sigma,
        )

        assert roq.training_data.shape == (ntraining, self.N)
        assert roq.nbases_real > 0
        assert roq.nbases_imag > 0
        assert roq.nbases2 > 0

        # check likelihood calculation
        Ntests = 100

        for _ in range(Ntests):
            # draw values from prior
            values = self.priors.sample()

            # students-t likelihood
            ll = roq.log_likelihood(**values)
            fullll = full_log_likelihood(
                self.generic_complex_model(self.times, **values), self.comp_data
            )

            assert np.abs(ll - fullll) < 1e-6

            # Gaussian likelihood
            ll = roq.log_likelihood(**values, likelihood="gaussian")
            fullll = full_log_likelihood(
                self.generic_complex_model(self.times, **values),
                self.comp_data,
                like="gaussian",
                sigma=self.sigma,
            )

            assert np.abs(ll - fullll) < 1e-6


class TestHeterodynedCWModelROQ:
    """
    Test the reduced order quadrature for a heterodyned CW signal model
    function.
    """

    @classmethod
    def setup_class(cls):
        # create a pulsar
        cls.pulsar = PulsarParameters()
        cls.pulsar["PSRJ"] = "J0123-0123"

        coords = SkyCoord(ra="01:23:00.0", dec="01:23:00.0", unit=("hourangle", "deg"))
        cls.pulsar["RAJ"] = coords.ra.rad
        cls.pulsar["DECJ"] = coords.dec.rad
        cls.pulsar["F"] = [123.456]
        cls.pulsar["H0"] = 9.2e-24
        cls.pulsar["IOTA"] = 0.789
        cls.pulsar["PSI"] = 1.1010101
        cls.pulsar["PHI0"] = 2.87654

        # generate some fake data
        cls.times = np.arange(1000000000, 1000086400, 60)
        cls.detector = "H1"
        cls.het = HeterodynedData(
            times=cls.times,
            par=cls.pulsar,
            injpar=cls.pulsar,
            fakeasd=cls.detector,
            inject=True,
            bbminlength=len(cls.times),  # forced to a single chunk
        )

        # fake multi-detector data with multiple chunks
        het1chunked = HeterodynedData(
            times=cls.times,
            par=cls.pulsar,
            injpar=cls.pulsar,
            fakeasd="H1",
            inject=True,
            bbmaxlength=int(len(cls.times) / 2),  # forced into multiple chunks
        )

        het2chunked = HeterodynedData(
            times=cls.times,
            par=cls.pulsar,
            injpar=cls.pulsar,
            fakeasd="H1",
            inject=True,
            bbmaxlength=int(len(cls.times) / 2),  # forced into multiple chunks
        )

        cls.multihet = MultiHeterodynedData({"H1": het1chunked, "L1": het2chunked})

        # set the prior
        cls.priors = PriorDict()
        cls.priors["h0"] = Uniform(0, 1e-22, name="h0")
        cls.priors["phi0"] = Uniform(0, np.pi, name="phi0")
        cls.priors["psi"] = Uniform(0, np.pi / 2, name="psi")
        cls.priors["iota"] = Sine(name="iota")

    def test_builtin_heterodyned_cw_model(self):
        ntraining = 500

        # generate ROQ
        roq = GenerateROQ(
            self.het.data,
            self.het.times.value,
            self.priors,
            par=self.het.par,
            det=self.het.detector,
            ntraining=ntraining,
            sigma=self.het.stds[0],
        )

        # there should only be two real/imag basis vectors
        assert roq.nbases_real == 2 and roq.nbases_imag == 2
        assert roq.nbases2 == 3

        # check likelihood calculation
        Ntests = 100

        ll = np.zeros(Ntests)
        fullll = np.zeros(Ntests)
        llg = np.zeros(Ntests)
        fullllg = np.zeros(Ntests)

        for i in range(Ntests):
            # draw values from prior
            parcopy = copy.deepcopy(self.pulsar)
            for key, value in self.priors.sample().items():
                parcopy[key] = value

            model = roq.model(
                newpar=parcopy,
                outputampcoeffs=False,
                updateSSB=True,
                updateBSB=True,
                updateglphase=True,
                freqfactor=2,
            )

            # students-t likelihood
            ll[i] = roq.log_likelihood(par=parcopy)
            fullll[i] = full_log_likelihood(model, self.het.data)

            # Gaussian likelihood
            llg[i] = roq.log_likelihood(par=parcopy, likelihood="gaussian")
            fullllg[i] = full_log_likelihood(
                model,
                self.het.data,
                like="gaussian",
                sigma=self.het.stds[0],
            )

        assert np.all(
            np.abs(np.exp(ll - ll.max()) - np.exp(fullll - fullll.max())) < 1e-3
        )
        assert np.all(
            np.abs(np.exp(llg - llg.max()) - np.exp(fullllg - fullllg.max())) < 1e-3
        )

    def test_studentst_likelihood(self):
        ntraining = 500

        # original likelihood
        like_orig = TargetedPulsarLikelihood(self.multihet, self.priors, numba=False)

        # ROQ likelihood
        like_roq = TargetedPulsarLikelihood(
            self.multihet,
            self.priors,
            roq=True,
            ntraining=ntraining,
            likelihood="STUDENTS-T",
        )

        assert len(like_roq._roq_all_nodes) == len(self.multihet)
        for j, het in enumerate(self.multihet):
            # check ROQ has been calculated for each "chunk"
            assert len(like_roq._roq_all_real_node_indices[j]) == het.num_chunks
            assert len(like_roq._roq_all_imag_node_indices[j]) == het.num_chunks
            assert len(like_roq._roq_all_model2_node_indices[j]) == het.num_chunks

            for k in range(het.num_chunks):
                # check number of ROQ nodes is as expected
                assert len(like_roq._roq_all_real_node_indices[j][k]) == 2
                assert len(like_roq._roq_all_imag_node_indices[j][k]) == 2
                assert len(like_roq._roq_all_model2_node_indices[j][k]) == 3

        # check likelihood calculation
        Ntests = 100

        llo = np.zeros(Ntests)
        llr = np.zeros(Ntests)

        for i in range(Ntests):
            parameters = self.priors.sample()

            like_orig.parameters = parameters.copy()
            like_roq.parameters = parameters.copy()

            # get likelihoods
            llo[i] = like_orig.log_likelihood()
            llr[i] = like_roq.log_likelihood()

        assert np.all(np.abs(np.exp(llo - llo.max()) - np.exp(llr - llr.max())) < 1e-3)

    def test_gaussian_likelihood(self):
        ntraining = 500

        # original likelihood
        like_orig = TargetedPulsarLikelihood(
            self.multihet, self.priors, numba=False, likelihood="gaussian"
        )

        # ROQ likelihood
        like_roq = TargetedPulsarLikelihood(
            self.multihet,
            self.priors,
            roq=True,
            ntraining=ntraining,
            likelihood="Normal",
        )

        # check likelihood calculation
        Ntests = 100

        llo = np.zeros(Ntests)
        llr = np.zeros(Ntests)

        for i in range(Ntests):
            parameters = self.priors.sample()

            like_orig.parameters = parameters.copy()
            like_roq.parameters = parameters.copy()

            # get likelihoods
            llo[i] = like_orig.log_likelihood()
            llr[i] = like_roq.log_likelihood()

        assert np.all(np.abs(np.exp(llo - llo.max()) - np.exp(llr - llr.max())) < 1e-3)


class TestROQFrequency(object):
    """
    Test the generation of an ROQ over frequency parameters.
    """

    @classmethod
    def setup_class(cls):
        # create some fake data frames using lalpulsar_Makefakedata_v5
        mfd = shutil.which("lalpulsar_Makefakedata_v5")

        cls.fakedatadir = "testing_fake_data"
        cls.fakedatadetector = "H1"
        cls.fakedatachannel = f"{cls.fakedatadetector}:FAKE_DATA"

        cls.fakedatastart = 1000000000
        cls.fakedataduration = 86400

        os.makedirs(cls.fakedatadir, exist_ok=True)

        cls.fakedatabandwidth = 8  # Hz
        sqrtSn = 2e-23  # noise amplitude spectral density
        cls.fakedataname = "FAKEDATA"

        # Create one pulsars to inject
        cls.fakepulsarpar = []

        # requirements for Makefakedata pulsar input files
        pulsarstr = """\
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

        # pulsar parameters
        f0 = 6.9456 / 2.0  # source rotation frequency (Hz)
        f1 = -9.87654e-11 / 2.0  # source rotational frequency derivative (Hz/s)
        f2 = 2.34134e-18 / 2.0  # second frequency derivative (Hz/s^2)
        alpha = 0.0  # source right ascension (rads)
        delta = 0.5  # source declination (rads)
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
            "pepoch": pepoch,
            "h0": h0,
            "cosi": cosiota,
            "psi": psi,
            "phi0": phi0,
        }

        cls.fakepulsarpar = PulsarParameters()
        cls.fakepulsarpar["PSRJ"] = "J0000+0000"
        cls.fakepulsarpar["H0"] = h0
        cls.fakepulsarpar["PHI0"] = phi0 / 2.0
        cls.fakepulsarpar["PSI"] = psi
        cls.fakepulsarpar["COSIOTA"] = cosiota
        cls.fakepulsarpar["F"] = [f0, f1, f2]
        cls.fakepulsarpar["RAJ"] = alpha
        cls.fakepulsarpar["DECJ"] = delta
        cls.fakepulsarpar["PEPOCH"] = pepoch
        cls.fakepulsarpar["EPHEM"] = "DE405"
        cls.fakepulsarpar["UNITS"] = "TDB"

        cls.fakepardir = "testing_fake_par_dir"
        os.makedirs(cls.fakepardir, exist_ok=True)
        cls.fakeparfile = os.path.join(cls.fakepardir, "J0000+0000.par")
        cls.fakepulsarpar.pp_to_par(cls.fakeparfile)

        injfile = os.path.join(cls.fakepardir, "inj.dat")
        with open(injfile, "w") as fp:
            fp.write("[Pulsar 1]\n")
            fp.write(pulsarstr.format(**mfddic))
            fp.write("\n")

        # set ephemeris files
        efile = download_ephemeris_file(
            LAL_EPHEMERIS_URL.format("earth00-40-DE405.dat.gz")
        )
        sfile = download_ephemeris_file(
            LAL_EPHEMERIS_URL.format("sun00-40-DE405.dat.gz")
        )

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
            "--randSeed=1234",  # for reproducibiliy
        ]

        # run makefakedata
        sp.run([mfd] + cmds)

        # resolution
        res = 1 / cls.fakedataduration

        # set priors for PE
        cls.priors = PriorDict()
        cls.priors["h0"] = Uniform(0, 1e-23, name="h0")
        cls.priors["f0"] = Uniform(f0 - 5 * res, f0 + 5 * res, name="f0")
        cls.priors["f1"] = Uniform(f1 - 5 * res**2, f1 + 5 * res**2, name="f1")

    @classmethod
    def teardown_class(cls):
        """
        Remove test simulation directory.
        """

        shutil.rmtree(cls.fakepardir)
        shutil.rmtree(cls.fakedatadir)

    def test_heterodyne_exact(self):
        """
        Heterodyne with the exact phase parameters
        """

        segments = [(self.fakedatastart, self.fakedatastart + self.fakedataduration)]

        fulloutdir = os.path.join(self.fakedatadir, "heterodyne_output")

        inputkwargs = dict(
            starttime=segments[0][0],
            endtime=segments[-1][-1],
            pulsarfiles=self.fakeparfile,
            segmentlist=segments,
            framecache=self.fakedatadir,
            channel=self.fakedatachannel,
            freqfactor=2,
            stride=86400 // 2,
            resamplerate=1 / 60,
            includessb=True,
            includebsb=True,
            includeglitch=True,
            output=fulloutdir,
            label="heterodyne_{psr}_{det}_{freqfactor}.hdf5",
        )

        het = heterodyne(**inputkwargs)

        # PE grid
        gridspace = {
            "h0": np.linspace(0, self.priors["h0"].maximum, 50),
            "f0": np.linspace(self.priors["f0"].minimum, self.priors["f0"].maximum, 50),
            "f1": np.linspace(self.priors["f1"].minimum, self.priors["f1"].maximum, 50),
        }

        peoutdir = os.path.join(self.fakedatadir, "pe_output")
        pelabel = "nonroq"

        # run non-ROQ PE over grid
        pekwargs = {
            "grid": True,
            "grid_kwargs": {"grid_size": gridspace, "save": True},
            "outdir": peoutdir,
            "label": pelabel,
            "prior": self.priors,
            "likelihood": "studentst",
            "detector": self.fakedatadetector,
            "data_file": list(het.outputfiles.values()),
            "par_file": copy.deepcopy(self.fakepulsarpar),
        }

        gridstandard = pe(**copy.deepcopy(pekwargs)).grid

        # set ROQ likelihood
        ntraining = 2000
        pekwargs["roq"] = True
        pekwargs["roq_kwargs"] = {"ntraining": ntraining}
        pekwargs["label"] = "roq"

        gridroq = pe(**pekwargs).grid

        # compare marginalised likelihoods for each parameter
        for par in gridspace:
            assert np.allclose(
                gridstandard.marginalize_ln_likelihood(not_parameters=par),
                gridroq.marginalize_ln_likelihood(not_parameters=par),
            )

    def test_heterodyne_offset(self):
        """
        Heterodyne with offset phase parameters
        """

        segments = [(self.fakedatastart, self.fakedatastart + self.fakedataduration)]

        fulloutdir = os.path.join(self.fakedatadir, "heterodyne_output")

        # create par file with offset frequency and frequency derivative
        offsetpar = copy.deepcopy(self.fakepulsarpar)
        res = 1 / self.fakedataduration
        offsetpar["F"] = [
            self.fakepulsarpar["F0"] - 2.3 * res,
            self.fakepulsarpar["F1"] + 1.8 * res**2,
        ]
        offsetparfile = os.path.join(self.fakepardir, "J0001+0001.par")
        offsetpar.pp_to_par(offsetparfile)

        inputkwargs = dict(
            starttime=segments[0][0],
            endtime=segments[-1][-1],
            pulsarfiles=offsetparfile,
            segmentlist=segments,
            framecache=self.fakedatadir,
            channel=self.fakedatachannel,
            freqfactor=2,
            stride=86400 // 2,
            resamplerate=1 / 60,
            includessb=True,
            includebsb=True,
            includeglitch=True,
            output=fulloutdir,
            label="heterodyne_{psr}_{det}_{freqfactor}.hdf5",
        )

        het = heterodyne(**inputkwargs)

        # PE grid
        gridspace = {
            "h0": np.linspace(0, self.priors["h0"].maximum, 50),
            "f0": np.linspace(self.priors["f0"].minimum, self.priors["f0"].maximum, 60),
            "f1": np.linspace(self.priors["f1"].minimum, self.priors["f1"].maximum, 60),
        }

        peoutdir = os.path.join(self.fakedatadir, "pe_output")
        pelabel = "nonroqoffset"

        # run non-ROQ PE over grid
        pekwargs = {
            "grid": True,
            "grid_kwargs": {"grid_size": gridspace, "save": True},
            "outdir": peoutdir,
            "label": pelabel,
            "prior": self.priors,
            "likelihood": "studentst",
            "detector": self.fakedatadetector,
            "data_file": list(het.outputfiles.values()),
            "par_file": copy.deepcopy(offsetpar),
        }

        gridstandard = pe(**copy.deepcopy(pekwargs)).grid

        # set ROQ likelihood
        ntraining = 2000
        pekwargs["roq"] = True
        pekwargs["roq_kwargs"] = {"ntraining": ntraining}
        pekwargs["label"] = "roqoffset"

        gridroq = pe(**pekwargs).grid

        # compare marginalised likelihoods for each parameter
        for par in gridspace:
            assert np.allclose(
                gridstandard.marginalize_ln_likelihood(not_parameters=par),
                gridroq.marginalize_ln_likelihood(not_parameters=par),
            )

        # check frequency and frequency derivative posterior peak at the
        # correct place
        f0idx = gridroq.marginalize_ln_posterior(not_parameters="f0").argmax()
        f0val = gridroq.sample_points["f0"][f0idx]
        assert f0val - res < self.fakepulsarpar["F0"] < f0val + res

        f1idx = gridroq.marginalize_ln_posterior(not_parameters="f1").argmax()
        f1val = gridroq.sample_points["f1"][f1idx]
        assert f1val - res**2 < self.fakepulsarpar["F1"] < f1val + res**2
