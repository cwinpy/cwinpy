"""
Test script for PE analysis.
"""

import os

import numpy as np
import pytest
from bilby.core.prior import PriorDict
from cwinpy.pe.pe import PERunner, pe


class TestPE(object):
    @classmethod
    def setup_class(cls):
        """
        Create data set files for use.
        """

        seed = 88523  # random seed
        start = 1000000000  # GPS start
        end = 1000086400  # GPS end
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
                    (
                        cls.times,
                        H1sigma * np.random.randn(size),
                        H1sigma * np.random.randn(size),
                    )
                ).T
            )

            cls.H1file.append("H1data{}f.txt".format(i))
            np.savetxt(cls.H1file[-1], cls.H1data[-1])

        # create simulated L1 data
        cls.L1data = []
        cls.L1file = []
        L1sigma = 0.7e-24
        for i in [1, 2]:
            cls.L1data.append(
                np.vstack(
                    (
                        cls.times,
                        L1sigma * np.random.randn(size),
                        L1sigma * np.random.randn(size),
                    )
                ).T
            )

            cls.L1file.append("L1data{}f.txt".format(i))
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

        cls.parfile = "pe_test.par"
        with open(cls.parfile, "w") as fp:
            fp.write(parcontent.format(cls.f0))

        # create a pulsar parameter file containing GW signal parameters
        # (for comparison with lalapps_pulsar_parameter_estimation_nested)
        parcontent = (
            "PSRJ     J0341-1253\n"
            "F0       {}\n"
            "F1       6.5e-12\n"
            "RAJ      03:41:00.0\n"
            "DECJ     -12:53:00.0\n"
            "PEPOCH   56789\n"
            "C21      6.2e-24\n"
            "C22      3.4e-25\n"
            "PHI21    0.4\n"
            "PHI22    1.3\n"
            "PSI      1.1\n"
            "IOTA     0.9\n"
            "UNITS    TCB"
        )

        cls.parfilesig = "pe_test_sig.par"
        with open(cls.parfilesig, "w") as fp:
            fp.write(parcontent.format(cls.f0))

        # set data pre-produced using lalapps_pulsar_parameter_estimation_nested
        # with the same parameter file
        cls.sigH11f = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                "inj_test.txt_H1_1.0_signal_only",
            )
        )
        cls.sigL11f = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                "inj_test.txt_L1_1.0_signal_only",
            )
        )
        cls.sigH12f = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                "inj_test.txt_H1_2.0_signal_only",
            )
        )
        cls.sigL12f = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                "inj_test.txt_L1_2.0_signal_only",
            )
        )

        # create a prior file
        cls.priorfile = "pe_test.prior"
        cls.priormin = 0.0
        cls.priormax = 1e-22
        priorcontent = "h0 = Uniform(name='h0', minimum={}, maximum={})"
        with open(cls.priorfile, "w") as fp:
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
        os.remove(cls.parfilesig)
        os.remove(cls.priorfile)

    def test_pe_runner_input(self):
        """
        Test the PERunner class fails as expected for wrong input types.
        """

        for inputs in [1.0, "hello", 1, True]:
            with pytest.raises(TypeError):
                PERunner(inputs)

    def test_data_input(self):
        """
        Test input data
        """

        # single detector and single data file
        config = "par-file = {}\n" "data-file = {}\n" "prior = {}\n"
        configfile = "config_test.ini"

        datafile = self.H1file[1]

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, datafile, self.priorfile))

        # no detector specified
        with pytest.raises(ValueError):
            pe(config=configfile)

        with pytest.raises(ValueError):
            pe(par_file=self.parfile, data_file=datafile)

        # not prior file specified
        with pytest.raises(ValueError):
            pe(par_file=self.parfile, data_file=datafile, detector="H1")

        # comparisons

        # pass as keyword arguments (detector as keyword)
        t1kw1 = pe(
            par_file=self.parfile,
            data_file=datafile,
            detector="H1",
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file string)
        t1kw2 = pe(
            par_file=self.parfile,
            data_file="{}:{}".format("H1", datafile),
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file dict)
        t1kw3 = pe(
            par_file=self.parfile, data_file={"H1": datafile}, prior=self.priorbilby
        )

        # pass as config file
        config = "par-file = {}\n" "data-file = {}\n" "prior = {}\n" "detector = H1"
        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, datafile, self.priorfile))
        t1c1 = pe(config=configfile)

        # use the data_file_2f option instead
        t1kw4 = pe(
            par_file=self.parfile,
            data_file_2f=datafile,
            detector="H1",
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file string)
        t1kw5 = pe(
            par_file=self.parfile,
            data_file_2f="{}:{}".format("H1", datafile),
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file dict)
        t1kw6 = pe(
            par_file=self.parfile, data_file_2f={"H1": datafile}, prior=self.priorbilby
        )

        # pass as config file
        config = "par-file = {}\n" "data-file-2f = {}\n" "prior = {}\n" "detector = H1"
        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, datafile, self.priorfile))
        t1c2 = pe(config=configfile)

        # perform consistency checks
        for tv in [t1kw1, t1kw2, t1kw3, t1c1, t1kw4, t1kw5, t1kw6, t1c2]:
            assert len(tv.hetdata) == 1
            assert tv.hetdata["H1"][0].par["F"][0] == self.f0
            assert tv.hetdata.detectors[0] == "H1"
            assert tv.hetdata.freq_factors[0] == 2
            assert np.allclose(tv.hetdata["H1"][0].data.real, self.H1data[1][:, 1])
            assert np.allclose(tv.hetdata["H1"][0].data.imag, self.H1data[1][:, 2])
            assert np.allclose(tv.hetdata["H1"][0].times.value, self.times)
            assert PriorDict(tv.prior) == self.priorbilby

        # now pass two detectors
        # pass as keyword arguments (detector as keyword)
        t2kw1 = pe(
            par_file=self.parfile,
            data_file=[self.H1file[1], self.L1file[1]],
            detector=["H1", "L1"],
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file string)
        t2kw2 = pe(
            par_file=self.parfile,
            data_file=[
                "{}:{}".format("H1", self.H1file[1]),
                "{}:{}".format("L1", self.L1file[1]),
            ],
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file dict)
        t2kw3 = pe(
            par_file=self.parfile,
            data_file={"H1": self.H1file[1], "L1": self.L1file[1]},
            prior=self.priorbilby,
        )

        # pass as config file
        config = (
            "par-file = {}\n"
            "data-file = [{}, {}]\n"
            "prior = {}\n"
            "detector = [H1, L1]"
        )
        with open(configfile, "w") as fp:
            fp.write(
                config.format(
                    self.parfile, self.H1file[1], self.L1file[1], self.priorfile
                )
            )
        t2c1 = pe(config=configfile)

        # use the data_file_2f option instead
        t2kw4 = pe(
            par_file=self.parfile,
            data_file_2f=[self.H1file[1], self.L1file[1]],
            detector=["H1", "L1"],
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file string)
        t2kw5 = pe(
            par_file=self.parfile,
            data_file_2f=[
                "{}:{}".format("H1", self.H1file[1]),
                "{}:{}".format("L1", self.L1file[1]),
            ],
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file dict)
        t2kw6 = pe(
            par_file=self.parfile,
            data_file_2f={"H1": self.H1file[1], "L1": self.L1file[1]},
            prior=self.priorbilby,
        )

        # pass as config file
        config = (
            "par-file = {}\n"
            "data-file-2f = [{}, {}]\n"
            "prior = {}\n"
            "detector = [H1, L1]"
        )
        with open(configfile, "w") as fp:
            fp.write(
                config.format(
                    self.parfile, self.H1file[1], self.L1file[1], self.priorfile
                )
            )
        t2c2 = pe(config=configfile)

        # perform consistency checks
        for tv in [t2kw1, t2kw2, t2kw3, t2c1, t2kw4, t2kw5, t2kw6, t2c2]:
            assert len(tv.hetdata) == 2
            for i, det, data in zip(
                range(2), ["H1", "L1"], [self.H1data[1], self.L1data[1]]
            ):
                assert tv.hetdata.detectors[i] == det
                assert tv.hetdata.freq_factors[0] == 2
                assert tv.hetdata[det][0].par["F"][0] == self.f0
                assert np.allclose(tv.hetdata[det][0].data.real, data[:, 1])
                assert np.allclose(tv.hetdata[det][0].data.imag, data[:, 2])
                assert np.allclose(tv.hetdata[det][0].times.value, self.times)
                assert PriorDict(tv.prior) == self.priorbilby

        # pass data at 1f
        datafile = self.H1file[0]
        t3kw1 = pe(
            par_file=self.parfile,
            data_file_1f=datafile,
            detector="H1",
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file string)
        t3kw2 = pe(
            par_file=self.parfile,
            data_file_1f="{}:{}".format("H1", datafile),
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file dict)
        t3kw3 = pe(
            par_file=self.parfile, data_file_1f={"H1": datafile}, prior=self.priorbilby
        )

        # pass as config file
        config = "par-file = {}\n" "data-file-1f = {}\n" "prior = {}\n" "detector = H1"
        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, datafile, self.priorfile))
        t3c1 = pe(config=configfile)

        # perform consistency checks
        for tv in [t3kw1, t3kw2, t3kw3, t3c1]:
            assert len(tv.hetdata) == 1
            assert tv.hetdata.detectors[0] == "H1"
            assert tv.hetdata.freq_factors[0] == 1
            assert tv.hetdata["H1"][0].par["F"][0] == self.f0
            assert np.allclose(tv.hetdata["H1"][0].data.real, self.H1data[0][:, 1])
            assert np.allclose(tv.hetdata["H1"][0].data.imag, self.H1data[0][:, 2])
            assert np.allclose(tv.hetdata["H1"][0].times.value, self.times)
            assert PriorDict(tv.prior) == self.priorbilby

        # test with two detectors and two frequencies
        # pass as keyword arguments (detector as keyword)
        t4kw1 = pe(
            par_file=self.parfile,
            data_file_1f=[self.H1file[0], self.L1file[0]],
            data_file_2f=[self.H1file[1], self.L1file[1]],
            detector=["H1", "L1"],
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file string)
        t4kw2 = pe(
            par_file=self.parfile,
            data_file_1f=[
                "{}:{}".format("H1", self.H1file[0]),
                "{}:{}".format("L1", self.L1file[0]),
            ],
            data_file_2f=[
                "{}:{}".format("H1", self.H1file[1]),
                "{}:{}".format("L1", self.L1file[1]),
            ],
            prior=self.priorbilby,
        )

        # pass as keyword arguments (detector in data file dict)
        t4kw3 = pe(
            par_file=self.parfile,
            data_file_1f={"H1": self.H1file[0], "L1": self.L1file[0]},
            data_file_2f={"H1": self.H1file[1], "L1": self.L1file[1]},
            prior=self.priorbilby,
        )

        # pass as config file
        config = (
            "par-file = {}\n"
            "data-file-1f = [{}, {}]\n"
            "data-file-2f = [{}, {}]\n"
            "prior = {}\n"
            "detector = [H1, L1]"
        )
        with open(configfile, "w") as fp:
            fp.write(
                config.format(
                    self.parfile,
                    self.H1file[0],
                    self.L1file[0],
                    self.H1file[1],
                    self.L1file[1],
                    self.priorfile,
                )
            )
        t4c1 = pe(config=configfile)

        # perform consistency checks
        for tv in [t4kw1, t4kw2, t4kw3, t4c1]:
            assert len(tv.hetdata) == 4
            for i, det, data1f, data2f in zip(
                range(2),
                ["H1", "L1"],
                [self.H1data[0], self.L1data[0]],
                [self.H1data[1], self.L1data[1]],
            ):
                assert tv.hetdata.detectors[i] == det
                assert tv.hetdata[det][0].freq_factor == 1.0
                assert tv.hetdata[det][1].freq_factor == 2.0
                assert tv.hetdata[det][0].par["F"][0] == self.f0
                assert np.allclose(tv.hetdata[det][0].data.real, data1f[:, 1])
                assert np.allclose(tv.hetdata[det][0].data.imag, data1f[:, 2])
                assert np.allclose(tv.hetdata[det][0].times.value, self.times)
                assert tv.hetdata[det][1].par["F"][0] == self.f0
                assert np.allclose(tv.hetdata[det][1].data.real, data2f[:, 1])
                assert np.allclose(tv.hetdata[det][1].data.imag, data2f[:, 2])
                assert np.allclose(tv.hetdata[det][1].times.value, self.times)
                assert PriorDict(tv.prior) == self.priorbilby
        os.remove(configfile)

    def test_fake_data_exceptions(self):
        """
        Test the exceptions when creating fake data using pe.
        """

        # pass as config file (with incompatible injection times)
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = H1\n"
            "inj-times = 1"
        )

        configfile = "config_test.ini"

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile))

        with pytest.raises(TypeError):
            pe(config=configfile)

        # create fake data in one detector with no signal
        # First test error for an inconsistent number of start times
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = H1\n"
            "fake-start = [1000000000, 1000000000]\n"
            "fake-end = 1000086400\n"
            "fake-dt = 60\n"
            "fake-asd = 1e-24"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile))

        with pytest.raises(ValueError):
            pe(config=configfile)

        # Test inconsistent detector and start time
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = H1\n"
            "fake-start = L1:1000000000\n"
            "fake-end = 1000086400\n"
            "fake-dt = 60\n"
            "fake-asd = 1e-24"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile))

        with pytest.raises(ValueError):
            pe(config=configfile)

        # Test inconsistent detector and end time
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = H1\n"
            "fake-start = 1000000000\n"
            "fake-end = L1:1000086400\n"
            "fake-dt = 60\n"
            "fake-asd = 1e-24"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile))

        with pytest.raises(ValueError):
            pe(config=configfile)

        # Test inconsistent detector and time step
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = H1\n"
            "fake-start = 1000000000\n"
            "fake-end = 1000086400\n"
            "fake-dt = L1:60\n"
            "fake-asd = 1e-24"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile))

        with pytest.raises(ValueError):
            pe(config=configfile)

        os.remove(configfile)

    def test_fake_data_1det_1harm(self):
        """
        Test generation of fake data for one detector and one harmonic.
        """

        # Test seeded data is the same when called two different ways
        configfile = "config_test.ini"
        seed = 178203
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = H1\n"
            "fake-asd = 1e-24\n"
            "fake-seed = {}"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile, seed))

        fd1 = pe(config=configfile)

        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "fake-asd = H1:1e-24\n"
            "fake-seed = {}"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile, seed))

        fd2 = pe(config=configfile)

        assert (fd1.hetdata.detectors == ["H1"]) and (fd2.hetdata.detectors == ["H1"])
        assert np.array_equal(
            fd1.hetdata["H1"][0].times.value, np.arange(1000000000, 1000086400, 60)
        )
        assert np.array_equal(fd1.hetdata["H1"][0].data, fd2.hetdata["H1"][0].data)
        assert len(fd1.hetdata.freq_factors) == 1
        assert fd1.hetdata.freq_factors[0] == 2
        assert fd2.hetdata.freq_factors[0] == 2

        # Check that using fake-asd and fake-asd-2f are equivalent
        fd3 = pe(
            par_file=self.parfile,
            prior=self.priorfile,
            fake_asd_2f={"H1": 1e-24},
            fake_seed=seed,
        )

        assert (fd1.hetdata.detectors == ["H1"]) and (fd3.hetdata.detectors == ["H1"])
        assert np.array_equal(
            fd3.hetdata["H1"][0].times.value, np.arange(1000000000, 1000086400, 60)
        )
        assert np.array_equal(fd1.hetdata["H1"][0].data, fd3.hetdata["H1"][0].data)
        assert fd1.hetdata.freq_factors == fd3.hetdata.freq_factors

        del fd1
        del fd2
        del fd3

        os.remove(configfile)

    def test_fake_data_1det_2harm(self):
        """
        Test generation of fake data for one detector and two harmonics.
        """

        configfile = "config_test.ini"
        seed = 178203

        # Test creating fake noise for one detector at 1f and 2f
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = H1\n"
            "fake-asd-1f = 1e-24\n"
            "fake-asd-2f = 2e-24\n"
            "fake-seed = {}"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile, seed))

        fd1 = pe(config=configfile)

        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "fake-asd-1f = H1:1e-24\n"
            "fake-asd-2f = H1:2e-24\n"
            "fake-seed = {}"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile, seed))

        fd2 = pe(config=configfile)

        assert (fd1.hetdata.detectors == ["H1"]) and (fd2.hetdata.detectors == ["H1"])
        assert len(fd1.hetdata["H1"]) == 2 and len(fd2.hetdata["H1"]) == 2
        assert np.array_equal(
            fd1.hetdata["H1"][0].times.value, np.arange(1000000000, 1000086400, 60)
        )
        assert np.array_equal(fd1.hetdata["H1"][0].times, fd1.hetdata["H1"][1].times)
        assert np.array_equal(fd2.hetdata["H1"][0].times, fd1.hetdata["H1"][0].times)
        assert np.array_equal(fd2.hetdata["H1"][0].times, fd2.hetdata["H1"][1].times)
        assert np.array_equal(fd1.hetdata["H1"][0].data, fd2.hetdata["H1"][0].data)
        assert np.array_equal(fd1.hetdata["H1"][1].data, fd2.hetdata["H1"][1].data)
        assert len(fd1.hetdata.freq_factors) == 2
        assert sorted(fd1.hetdata.freq_factors) == [1, 2]
        assert sorted(fd2.hetdata.freq_factors) == [1, 2]

        del fd1
        del fd2

        os.remove(configfile)

    def test_fake_data_2det_1harm(self):
        """
        Test generation of fake data for two detectors and one harmonic.
        """

        configfile = "config_test.ini"
        seed = 178203

        # Test creating fake noise for two detectors
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = [H1, L1]\n"
            "fake-asd = [1e-24, 2e-24]\n"
            "fake-start = [1000000000, 1000000100]\n"
            "fake-end = [1000086400, 1000086500]\n"
            "fake-seed = {}"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile, seed))

        fd1 = pe(config=configfile)

        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "fake-asd = [H1:1e-24, L1:2e-24]\n"
            "fake-start = [H1:1000000000, L1:1000000100]\n"
            "fake-end = [H1:1000086400, L1:1000086500]\n"
            "fake-seed = {}"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile, seed))

        fd2 = pe(config=configfile)

        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "fake-asd = [H1:1e-24, L1:2e-24]\n"
            "fake-start = [H1:1000000000, L1:1000000100]\n"
            "fake-end = [H1:1000086400, L1:1000086500]"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile))

        fd3 = pe(config=configfile)

        assert (
            (len(fd1.hetdata.detectors) == 2)
            and (len(fd2.hetdata.detectors) == 2)
            and (len(fd3.hetdata.detectors) == 2)
        )
        assert fd1.hetdata.detectors == fd2.hetdata.detectors
        assert fd3.hetdata.detectors == fd1.hetdata.detectors
        assert "L1" in fd1.hetdata.detectors and "H1" in fd1.hetdata.detectors
        assert np.array_equal(
            fd1.hetdata["H1"][0].times.value, np.arange(1000000000, 1000086400, 60)
        )
        assert np.array_equal(fd1.hetdata["H1"][0].times, fd2.hetdata["H1"][0].times)
        assert np.array_equal(
            fd1.hetdata["L1"][0].times.value, np.arange(1000000100, 1000086500, 60)
        )
        assert np.array_equal(fd1.hetdata["L1"][0].times, fd2.hetdata["L1"][0].times)
        assert np.array_equal(fd1.hetdata["H1"][0].data, fd2.hetdata["H1"][0].data)
        assert np.array_equal(fd1.hetdata["L1"][0].data, fd2.hetdata["L1"][0].data)
        assert not np.array_equal(fd1.hetdata["H1"][0].data, fd3.hetdata["H1"][0].data)
        assert not np.array_equal(fd1.hetdata["L1"][0].data, fd3.hetdata["L1"][0].data)

        os.remove(configfile)

    def test_fake_data_2det_2harm(self):
        """
        Test generation of fake data for two detectors and two harmonics.
        """

        configfile = "config_test.ini"
        seed = 178203

        # Test creating fake noise for two detectors at 1f and 2f
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = [H1, L1]\n"
            "fake-asd-1f = [1e-24, 2e-24]\n"
            "fake-asd-2f = [2e-24, 4e-24]\n"
            "fake-seed = {}"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile, seed))

        fd1 = pe(config=configfile)

        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "fake-asd-1f = [H1:1e-24, L1:2e-24]\n"
            "fake-asd-2f = [H1:2e-24, L1:4e-24]\n"
            "fake-seed = {}"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfile, self.parfile, self.priorfile, seed))

        fd2 = pe(config=configfile)

        assert (len(fd1.hetdata.detectors) == 2) and (len(fd2.hetdata.detectors) == 2)
        assert fd1.hetdata.detectors == fd2.hetdata.detectors
        assert "L1" in fd1.hetdata.detectors and "H1" in fd1.hetdata.detectors
        assert np.array_equal(
            fd1.hetdata["H1"][0].times.value, np.arange(1000000000, 1000086400, 60)
        )
        assert np.array_equal(fd1.hetdata["H1"][0].times, fd2.hetdata["H1"][0].times)
        assert np.array_equal(fd1.hetdata["H1"][0].times, fd1.hetdata["H1"][1].times)
        assert np.array_equal(fd1.hetdata["L1"][0].times, fd2.hetdata["L1"][0].times)
        assert np.array_equal(fd1.hetdata["L1"][0].times, fd1.hetdata["L1"][1].times)
        assert np.array_equal(fd1.hetdata["H1"][0].data, fd2.hetdata["H1"][0].data)
        assert np.array_equal(fd1.hetdata["H1"][1].data, fd2.hetdata["H1"][1].data)
        assert np.array_equal(fd1.hetdata["L1"][0].data, fd2.hetdata["L1"][0].data)
        assert np.array_equal(fd1.hetdata["L1"][1].data, fd2.hetdata["L1"][1].data)
        assert len(fd1.hetdata.freq_factors) == 4
        assert fd1.hetdata.freq_factors == [1, 2, 1, 2]
        assert fd2.hetdata.freq_factors == [1, 2, 1, 2]

        os.remove(configfile)

    def test_fake_signal_2det_2harm(self):
        """
        Test generation of fake signal for two detectors and two harmonics.
        """

        configfile = "config_test.ini"

        # Test creating an injected signal for two detectors at 1f and 2f
        config = (
            "par-file = {}\n"
            "inj-par = {}\n"
            "prior = {}\n"
            "detector = [H1, L1]\n"
            "fake-asd-1f = [1e-24, 2e-24]\n"
            "fake-asd-2f = [2e-24, 4e-24]\n"
            "fake-start = [1000000000, 1000100000]\n"
            "fake-end = [1000086400, 1000186400]\n"
            "fake-dt = [1800, 1800]"
        )

        with open(configfile, "w") as fp:
            fp.write(config.format(self.parfilesig, self.parfilesig, self.priorfile))

        fd1 = pe(config=configfile)

        assert len(fd1.hetdata.detectors) == 2
        assert "L1" in fd1.hetdata.detectors and "H1" in fd1.hetdata.detectors
        assert np.array_equal(fd1.hetdata["H1"][0].times.value, self.sigH11f[:, 0])
        assert np.array_equal(fd1.hetdata["H1"][1].times.value, self.sigH12f[:, 0])
        assert np.array_equal(fd1.hetdata["L1"][0].times.value, self.sigL11f[:, 0])
        assert np.array_equal(fd1.hetdata["L1"][1].times.value, self.sigL12f[:, 0])
        assert np.allclose(
            fd1.hetdata["H1"][0].injection_data,
            self.sigH11f[:, 1] + 1j * self.sigH11f[:, 2],
            atol=0.0,
        )
        assert np.allclose(
            fd1.hetdata["H1"][1].injection_data,
            self.sigH12f[:, 1] + 1j * self.sigH12f[:, 2],
            atol=0.0,
        )
        assert np.allclose(
            fd1.hetdata["L1"][0].injection_data,
            self.sigL11f[:, 1] + 1j * self.sigL11f[:, 2],
            atol=0.0,
        )
        assert np.allclose(
            fd1.hetdata["L1"][1].injection_data,
            self.sigL12f[:, 1] + 1j * self.sigL12f[:, 2],
            atol=0.0,
        )

        os.remove(configfile)
