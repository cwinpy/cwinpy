"""
Test script for knope analysis.
"""

import os
import pytest
import numpy as np

from cwinpy.knope.knope import (knope, KnopeRunner)
from bilby.core.prior import PriorDict


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

        # create a prior file
        cls.priorfile = 'knope_test.prior'
        cls.priormin = 0.
        cls.priormax = 1e-22
        priorcontent = (
            "h0 = Uniform(name='h0', minimum={}, maximum={})"
        )
        with open(cls.priorfile, 'w') as fp:
            fp.write(priorcontent.format(cls.priormin, cls.priormax))
        cls.priorbilby = PriorDict(cls.priorfile)

    @classmethod
    def teardown_class(cls):
        """
        Remove data set files.
        """

        for dfile in cls.H1file + cls.L1file:
            os.remove(dfile)
        os.remove(cls.parfile)
        os.remove(cls.priorfile)

    def test_knope_runner_input(self):
        """
        Test the KnopeRunner class fails as expected for wrong input types.
        """

        for inputs in [1.0, 'hello', 1, True]:
            with pytest.raises(TypeError):
                KnopeRunner(inputs)

    def test_data_input(self):
        """
        Test input data
        """

        # single detector and single data file
        config = (
            "par-file = {}\n"
            "data-file = {}\n"
            "prior = {}\n"
        )
        configfile = 'config_test.ini'

        datafile = self.H1file[1]

        with open(configfile, 'w') as fp:
            fp.write(config.format(
                self.parfile,
                datafile,
                self.priorfile)
            )

        # no detector specified
        with pytest.raises(ValueError):
            knope(config=configfile)
        
        with pytest.raises(ValueError):
            knope(
                par_file=self.parfile,
                data_file=datafile
            )

        # not prior file specified
        with pytest.raises(ValueError):
            knope(
                par_file=self.parfile,
                data_file=datafile,
                detector='H1'
            )

        # comparisons

        # pass as keyword arguments (detector as keyword)
        t1kw1 = knope(
            par_file=self.parfile,
            data_file=datafile,
            detector='H1',
            prior=self.priorbilby
        )

        # pass as keyword arguments (detector in data file string)
        t1kw2 = knope(
            par_file=self.parfile,
            data_file='{}:{}'.format('H1', datafile),
            prior=self.priorbilby
        )

        # pass as keyword arguments (detector in data file dict)
        t1kw3 = knope(
            par_file=self.parfile,
            data_file={'H1': datafile},
            prior=self.priorbilby
        )

        # pass as config file
        config = (
            "par-file = {}\n"
            "data-file = {}\n"
            "prior = {}\n"
            "detector = H1"
        )
        with open(configfile, 'w') as fp:
            fp.write(config.format(
                self.parfile,
                datafile,
                self.priorfile)
            )
        t1c1 = knope(config=configfile)

        # use the data_file_2f option instead
        t1kw4 = knope(
            par_file=self.parfile,
            data_file=datafile,
            detector='H1',
            prior=self.priorbilby
        )

        # pass as keyword arguments (detector in data file string)
        t1kw5 = knope(
            par_file=self.parfile,
            data_file='{}:{}'.format('H1', datafile),
            prior=self.priorbilby
        )

        # pass as keyword arguments (detector in data file dict)
        t1kw6 = knope(
            par_file=self.parfile,
            data_file_2f={'H1': datafile},
            prior=self.priorbilby
        )

        # pass as config file
        config = (
            "par-file = {}\n"
            "data-file-2f = {}\n"
            "prior = {}\n"
            "detector = H1"
        )
        with open(configfile, 'w') as fp:
            fp.write(config.format(
                self.parfile,
                datafile,
                self.priorfile)
            )
        t1c2 = knope(config=configfile)

        # perform consistency checks
        for tv in [t1kw1, t1kw2, t1kw3, t1c1, t1kw4, t1kw5, t1kw6, t1c2]:
            assert len(tv.hetdata) == 1
            assert tv.hetdata.detectors[0] == 'H1'
            assert tv.hetdata.freq_factors[0] == 2
            assert np.allclose(tv.hetdata['H1'][0].data.real, self.H1data[1][:, 1])
            assert np.allclose(tv.hetdata['H1'][0].data.imag, self.H1data[1][:, 2])
            assert np.allclose(tv.hetdata['H1'][0].times, self.times)
            assert PriorDict(tv.prior) == self.priorbilby

        # now pass two detectors
        # pass as keyword arguments (detector as keyword)
        t2kw1 = knope(
            par_file=self.parfile,
            data_file=[self.H1file[1], self.L1file[1]],
            detector=['H1', 'L1'],
            prior=self.priorbilby
        )

        # pass as keyword arguments (detector in data file string)
        t2kw2 = knope(
            par_file=self.parfile,
            data_file=['{}:{}'.format('H1', self.H1file[1]),
                       '{}:{}'.format('L1', self.L1file[1])],
            prior=self.priorbilby
        )

        # pass as keyword arguments (detector in data file dict)
        t2kw3 = knope(
            par_file=self.parfile,
            data_file={'H1': self.H1file[1], 'L1': self.L1file[1]},
            prior=self.priorbilby
        )

        # pass as config file
        config = (
            "par-file = {}\n"
            "data-file = [{}, {}]\n"
            "prior = {}\n"
            "detector = [H1, L1]"
        )
        with open(configfile, 'w') as fp:
            fp.write(config.format(
                self.parfile,
                self.H1file[1],
                self.L1file[1],
                self.priorfile)
            )
        t2c1 = knope(config=configfile)

        # use the data_file_2f option instead
        t2kw4 = knope(
            par_file=self.parfile,
            data_file_2f=[self.H1file[1], self.L1file[1]],
            detector=['H1', 'L1'],
            prior=self.priorbilby
        )

        # pass as keyword arguments (detector in data file string)
        t2kw5 = knope(
            par_file=self.parfile,
            data_file_2f=['{}:{}'.format('H1', self.H1file[1]),
                          '{}:{}'.format('L1', self.L1file[1])],
            prior=self.priorbilby
        )

        # pass as keyword arguments (detector in data file dict)
        t2kw6 = knope(
            par_file=self.parfile,
            data_file_2f={'H1': self.H1file[1], 'L1': self.L1file[1]},
            prior=self.priorbilby
        )

        # pass as config file
        config = (
            "par-file = {}\n"
            "data-file-2f = [{}, {}]\n"
            "prior = {}\n"
            "detector = [H1, L1]"
        )
        with open(configfile, 'w') as fp:
            fp.write(config.format(
                self.parfile,
                self.H1file[1],
                self.L1file[1],
                self.priorfile)
            )
        t2c2 = knope(config=configfile)

        # perform consistency checks
        for tv in [t2kw1, t2kw2, t2kw3, t2c1, t2kw4, t2kw5, t2kw6, t2c2]:
            assert len(tv.hetdata) == 2
            for i, det, data in zip(range(2), ['H1', 'L1'], [self.H1data[1], self.L1data[1]]):
                assert tv.hetdata.detectors[i] == det
                assert tv.hetdata.freq_factors[0] == 2
                assert np.allclose(tv.hetdata[det][0].data.real, data[:, 1])
                assert np.allclose(tv.hetdata[det][0].data.imag, data[:, 2])
                assert np.allclose(tv.hetdata[det][0].times, self.times)
                assert PriorDict(tv.prior) == self.priorbilby

        os.remove(configfile)