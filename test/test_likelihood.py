"""
Test script for data.py classes.
"""

import os

import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform
from cwinpy import HeterodynedData, MultiHeterodynedData, TargetedPulsarLikelihood


class TestTargetedPulsarLikelhood(object):
    """
    Tests for the TargetedPulsarLikelihood class.
    """

    parfile = "J0123+3456.par"
    times = np.linspace(1000000000.0, 1000086340.0, 1440)
    data = np.random.normal(0.0, 1e-25, size=(1440, 2))
    onesdata = np.ones((1440, 2))
    detector = "H1"

    @classmethod
    def setup_class(cls):
        # create a pulsar parameter file
        parcontent = """\
PSRJ     J0123+3456
RAJ      01:23:45.6789
DECJ     34:56:54.321
F0       567.89
F1       -1.2e-12
PEPOCH   56789
H0       9.87e-26
COSIOTA  0.3
PSI      1.1
PHI0     2.4
"""

        # add content to the par file
        with open("J0123+3456.par", "w") as fp:
            fp.write(parcontent)

    @classmethod
    def teardown_class(cls):
        os.remove("J0123+3456.par")

    def test_wrong_inputs(self):
        """
        Test that exceptions are raised for incorrect inputs to the
        TargetedPulsarLikelihood.
        """

        with pytest.raises(TypeError):
            TargetedPulsarLikelihood(None, None)

        # create HeterodynedData object (no par file)
        het = HeterodynedData(self.data, times=self.times, detector=self.detector)

        priors = dict()
        priors["h0"] = Uniform(0.0, 1.0e-23, "h0")

        # error with no par file
        with pytest.raises(ValueError):
            TargetedPulsarLikelihood(het, PriorDict(priors))

        het = HeterodynedData(
            self.data, times=self.times, detector=self.detector, par=self.parfile
        )
        mhet = MultiHeterodynedData(het)  # multihet object for testing

        with pytest.raises(TypeError):
            TargetedPulsarLikelihood(het, None)

        with pytest.raises(TypeError):
            TargetedPulsarLikelihood(mhet, None)

    def test_priors(self):
        """
        Test the parsed priors.
        """

        # bad priors (unexpected parameter names)
        priors = dict()
        priors["a"] = Uniform(0.0, 1.0, "blah")
        priors["b"] = 2.0

        het = HeterodynedData(
            self.data, times=self.times, detector=self.detector, par=self.parfile
        )

        with pytest.raises(ValueError):
            _ = TargetedPulsarLikelihood(het, PriorDict(priors))

    def test_wrong_likelihood(self):
        """
        Test with a bad likelihood name.
        """

        het = HeterodynedData(
            self.data, times=self.times, detector=self.detector, par=self.parfile
        )

        priors = dict()
        priors["h0"] = Uniform(0.0, 1.0e-23, "h0")

        with pytest.raises(ValueError):
            _ = TargetedPulsarLikelihood(het, PriorDict(priors), likelihood="blah")

    def test_likelihood_null_likelihood(self):
        """
        Test likelihood and null likelihood.
        """

        het = HeterodynedData(
            self.data, times=self.times, detector=self.detector, par=self.parfile
        )

        priors = dict()
        priors["h0"] = Uniform(0.0, 1.0e-23, "h0")

        for likelihood in ["gaussian", "studentst"]:
            like = TargetedPulsarLikelihood(
                het, PriorDict(priors), likelihood=likelihood
            )
            like.parameters = {"h0": 0.0}

            assert like.log_likelihood() == like.noise_log_likelihood()

    def test_numba_likelihood(self):
        """
        Test likelihood using numba against the standard likelihood.
        """

        het = HeterodynedData(
            self.data, times=self.times, detector=self.detector, par=self.parfile
        )

        priors = dict()
        priors["h0"] = Uniform(0.0, 1.0e-23, "h0")

        for likelihood in ["gaussian", "studentst"]:
            like1 = TargetedPulsarLikelihood(
                het, PriorDict(priors), likelihood=likelihood
            )
            like1.parameters = {"h0": 1e-24}

            like2 = TargetedPulsarLikelihood(
                het, PriorDict(priors), likelihood=likelihood, numba=True
            )
            like2.parameters = {"h0": 1e-24}

            assert np.allclose(
                [like1.log_likelihood()], [like2.log_likelihood()], atol=1e-10, rtol=0.0
            )
