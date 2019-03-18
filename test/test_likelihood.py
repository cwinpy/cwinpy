"""
Test script for data.py classes.
"""

import os
import pytest
import numpy as np
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy

from cwinpy import HeterodynedData, MultiHeterodynedData
from cwinpy import TargetedPulsarLikelihood


class TestTargetedPulsarLikelhood(object):
    """
    Tests for the TargetedPulsarLikelihood class.
    """

    def test_wrong_inputs(self):
        """
        Test that exceptions are raised for incorrect inputs to the
        TargetedPulsarLikelihood.
        """

        with pytest.raises(TypeError):
            TargetedPulsarLikelihood(None, None)

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

        # create HeterodynedData object
        het = HeterodynedData(data, times=times, detector=detector, par=parfile)
        mhet = MultiHeterodynedData(het)  # multihet object for testing

        with pytest.raises(TypeError):
            TargetedPulsarLikelihood(het, None)

        with pytest.raises(TypeError):
            TargetedPulsarLikelihood(mhet, None)

        os.remove(parfile)
