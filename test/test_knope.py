"""
Test script for knope analysis.
"""

import os
import pytest
import numpy as np

from cwinpy.knope.knope import (knope, KnopeRunner)


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
        cls.times = np.arange(start, end, step)
        size = len(cls.times)

        # set random seed
        np.random.seed(seed)

        # create simulated H1 data (at 1 and 2f)
        cls.H1data = []
        cls.H1file = []
        H1sigma = 1e-24
        for i in [1, 2]:
            cls.H1data.append(
                np.vstack(
                    (cls.times,
                     H1sigma * np.random.randn(size),
                     H1sigma * np.random.randn(size))
                ).T
            )

            cls.H1file.append('H1data{}f.txt'.format(i))
            np.savetxt(cls.H1file[-1], cls.H1data[-1])

        # create simulated L1 data
        cls.L1data = []
        cls.L1file = []
        L1sigma = 0.7e-24
        for i in [1, 2]:
            cls.L1data.append(
                np.vstack(
                    (cls.times,
                     L1sigma * np.random.randn(size),
                     L1sigma * np.random.randn(size))
                ).T
            )

            cls.L1file.append('L1data{}f.txt'.format(i))
            np.savetxt(cls.L1file[-1], cls.L1data[-1])

        # create pulsar parameter file
        cls.f0 = 100.1  # frequency
        parcontent = (
            "PSRJ     J0341-1253\n"
            "F0       {}\n"
            "F1       6.5e-12\n"
            "RAJ      03:41:00.0\n"
            "DECJ     -12:53:00.0\n"
            "PEPOCH   56789"
        )

        cls.parfile = 'knope_test.par'
        with open(cls.parfile, 'w') as fp:
            fp.write(parcontent.format(cls.f0))

    @classmethod
    def teardown_class(cls):
        """
        Remove data set files.
        """

        for dfile in cls.H1file + cls.L1file:
            os.remove(dfile)
        os.remove(cls.parfile)

    def test_knope_runner_input(self):
        """
        Test the KnopeRunner class fails as expected for wrong input types.
        """

        for inputs in [1.0, 'hello', 1, True]:
            with pytest.raises(TypeError):
                KnopeRunner(inputs)

    def test_no_par_input(self):
        """
        Test no pulsar parameter file input.
        """

        config = "{} = {}"
        configfile = 'config_test.ini'

        # try no par file
        with open(configfile, 'w') as fp:
            fp.write(config.format('blah', 'blah'))

        with pytest.raises(KeyError):
            knope(config=configfile)

        with pytest.raises(KeyError):
            knope(detector='H1')

        os.remove(configfile)

    def test_data_input(self):
        """
        Test input data
        """

        # single detector and single data file
        config = (
            "par-file = {}"
            "data-file = {}"
        )
        configfile = 'config_test.ini'

        datafile = self.H1file[0]

        with open(configfile, 'w') as fp:
            fp.write(config.format(
                self.parfile,
                datafile))

        # no detector specified
        with pytest.raises(ValueError):
            knope(config=configfile)
        
        with pytest.raises(ValueError):
            knope(
                par_file=self.parfile,
                data_file=datafile
            )

        os.remove(configfile)