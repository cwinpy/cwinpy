"""
Test script for data.py classes.
"""

import os
import pytest
import numpy as np
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy

from cwinpy import HeterodynedData, MultiHeterodynedData
from cwinpy import TargetedPulsarLikelihood
from bilby.core.prior import PriorDict, Uniform


class TestTargetedPulsarLikelhood(object):
    """
    Tests for the TargetedPulsarLikelihood class.
    """

    parfile = 'J0123+3456.par'

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
        with open('J0123+3456.par', 'w') as fp:
            fp.write(parcontent)

    @classmethod
    def teardown_class(cls):
        os.remove('J0123+3456.par')

    def test_wrong_inputs(self):
        """
        Test that exceptions are raised for incorrect inputs to the
        TargetedPulsarLikelihood.
        """

        with pytest.raises(TypeError):
            TargetedPulsarLikelihood(None, None)

        times = np.linspace(1000000000., 1000086340., 1440)
        data = np.random.normal(0., 1e-25, size=(1440, 2))
        detector = 'H1'

        # create HeterodynedData object
        het = HeterodynedData(data, times=times, detector=detector, par=self.parfile)
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
        priors['a'] = Uniform(0., 1., 'blah')
        priors['b'] = 2.

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

        parfile = 'J0123+3456.par'

        # add content to the par file
        with open(parfile, 'w') as fp:
            fp.write(parcontent)

        times = np.linspace(1000000000., 1000086340., 1440)
        data = np.random.normal(0., 1e-25, size=(1440, 2))
        detector = 'H1'

        het = HeterodynedData(data, times=times, detector=detector, par=self.parfile)

        with pytest.raises(ValueError):
            like = TargetedPulsarLikelihood(het, PriorDict(priors))
