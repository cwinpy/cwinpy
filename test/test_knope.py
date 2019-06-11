"""
Test script for knope analysis.
"""

import os
import pytest
import numpy as np

from cwinpy.knope.knope import knope


class TestKnope(object):
    @classmethod
    def setup_class(cls):
        """
        Create data set files for use.
        """

        seed = 88523  # random seed
        start = 1000000000  # GPS start
        end = 1000086400    # GPS end
        step = 60  # time step size

        # time stamp array
        self.times = np.arange(start, end, step)
        size = len(times)

        # set random seed
        np.random.seed(seed)

        # create simulated H1 data
        H1sigma = 1e-24
        self.H1data = np.vstack(
            (self.times,
             H1sigma * np.random.randn(size),
             H1sigma * np.random.randn(size))
        ).T

        self.H1file = 'H1data.txt'
        np.savetxt(self.H1file)

        # create simulated L1 data
        L1sigma = 0.7e-24
        self.L1data = np.hstack(
            (self.times,
             L1sigma * np.random.randn(size),
             L1sigma * np.random.randn(size))
        ).T

        self.L1file = 'L1data.txt'
        np.savetxt(self.L1file)

        # create pulsar parameter file
        parcontent = (
            "PSRJ     J0341-1253\n"
            "F0       100.1\n"
            "F1       6.5e-12\n"
            "RAJ      03:41:00.0\n"
            "DECJ     -12:53:00.0\n"
            "PEPOCH   56789"
        )

        self.parfile = 'knope_test.par'
        with open(self.parfile, 'w') as fp:
            fp.write(parcontent)


    @classmethod
    def teardown_class(cls):
        """
        Remove data set files.
        """

        os.remove(self.H1file)
        os.remove(self.L1file)
        os.remove(self.parfile)

