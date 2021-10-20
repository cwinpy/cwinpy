"""
Test script for data.py classes.
"""

import os

import cwinpy
import lal
import numpy as np
import pytest
from cwinpy import HeterodynedData, MultiHeterodynedData, PulsarParameters
from cwinpy.data import PSDwrapper
from matplotlib.figure import Figure


class TestHeterodynedData(object):
    """
    Tests for the HeterodynedData and MultiHeterodynedData objects.
    """

    def test_no_data(self):
        """
        Test exception occurs if passing no data and no time stamps.
        """

        # test exception if no data or times are passed
        with pytest.raises(ValueError):
            HeterodynedData()

    def test_broken_data(self):
        """
        Test reading of data fails during to a "broken" input file
        """

        # create a "broken" input file (not the spurious "H")
        brokendata = """\
# times       real      imaginary
1000000000.0  -2.3e-25   4.3e-26
1000000060.0   H.2e-26   1.2e-25
1000000120.0  -1.7e-25  -2.8e-25
1000000180.0  -7.6e-26  -8.9e-26
"""
        brokenfile = "brokendata.txt"
        with open(brokenfile, "w") as fp:
            fp.write(brokendata)

        with pytest.raises(IOError):
            HeterodynedData(brokenfile)

        # run through MultiHeterodynedData
        with pytest.raises(ValueError):
            MultiHeterodynedData(brokenfile)

        with pytest.raises(IOError):
            MultiHeterodynedData({"H1": brokenfile})

        os.remove(brokenfile)  # clean up file

    def test_multi_data(self):
        """
        Test various ways of generating data for multiple detectors.
        """

        # create four datasets
        times1 = np.linspace(1000000000.0, 1000086340.0, 1440)
        data1 = np.random.normal(0.0, 1e-25, size=(1440, 2)).tolist()
        detector1 = "H1"

        times2 = np.linspace(1000000000.0, 1000086340.0, 1440)
        data2 = np.random.normal(0.0, 1e-25, size=(1440, 2))
        detector2 = "L1"

        times3 = np.linspace(1000000000.0, 1000086340.0, 1440)
        data3 = np.random.normal(0.0, 1e-25, size=(1440, 2))
        detector3 = "G1"

        times4 = np.linspace(1000000000.0, 1000086340.0, 1440)
        data4 = np.random.normal(0.0, 1e-25, size=(1440, 2))
        detector4 = "K1"

        # add first dataset as precreated HeterodynedData object
        het1 = HeterodynedData(data1, times=times1, detector=detector1)

        mhet = MultiHeterodynedData(het1)

        # add second dataset as a dictionary
        ddic = {detector2: data2}
        tdic = {"XX": times2}  # set to fail

        # add second dataset
        with pytest.raises(KeyError):
            mhet.add_data(ddic, tdic, detector2)

        # fix tdic
        tdic = {detector2: times2}
        mhet.add_data(ddic, tdic, detector2)

        # add third data set as a dictionary of HeterodynedData
        het3 = HeterodynedData(data3, times=times3, detector=detector3)
        ddic = {detector3: het3}
        mhet.add_data(ddic)

        # add fourth data set by just passing the data
        tdic = {detector4: times4}  # fail with dictionary of times

        with pytest.raises(TypeError):
            mhet.add_data(data4, tdic, detector4)

        # just add with times
        mhet.add_data(data4, times4, detector4)

        assert len(mhet) == 4
        assert len(mhet.detectors) == 4
        assert len(mhet.to_list) == 4

        # test looping over MultiHeterodynedData
        dets = [detector1, detector2, detector3, detector4]
        for data, det in zip(mhet, dets):
            assert det == data.detector

    def test_too_many_columns(self):
        """
        Test for failure if there are too many columns in the data file.
        """

        # create a "broken" input file (not the spurious "H")
        brokendata = """\
# times       real      imaginary  std    extra
1000000000.0  -2.3e-25   4.3e-26   1e-26  1
1000000060.0   3.2e-26   1.2e-25   1e-26  2
1000000120.0  -1.7e-25  -2.8e-25   1e-26  3
1000000180.0  -7.6e-26  -8.9e-26   1e-26  4
"""
        brokenfile = "brokendata.txt"
        with open(brokenfile, "w") as fp:
            fp.write(brokendata)

        with pytest.raises(IOError):
            HeterodynedData(brokenfile)

        os.remove(brokenfile)  # clean up file

    def test_too_few_columns(self):
        """
        Test for failure if there are too few columns in the data file.
        """

        # create a "broken" input file (not the spurious "H")
        brokendata = """\
# times       real
1000000000.0  -2.3e-25
1000000060.0   3.2e-26
1000000120.0  -1.7e-25
1000000180.0  -7.6e-26
"""
        brokenfile = "brokendata.txt"
        with open(brokenfile, "w") as fp:
            fp.write(brokendata)

        with pytest.raises(IOError):
            HeterodynedData(brokenfile)

        os.remove(brokenfile)  # clean up file

    def test_nonuniform_data(self):
        """
        Test that non-uniform data times stamps are correctly retained.
        """

        # create four datasets
        times = np.linspace(1000000000.0, 1000086340.0, 1440)

        # remove some times to create non-uniform sampling
        times = np.delete(times, [20, 897, 1200])

        data = np.random.normal(0.0, 1e-25, size=(len(times), 2))
        detector = "H1"

        het = HeterodynedData(data=data, times=times, detector=detector)

        assert np.all(times == het.times.value)
        assert het.dt.value == np.min(np.diff(times))

    def test_read_text_data(self):
        """
        Test that a valid input file is read in correctly.
        """

        # create a data file to output
        hetdata = """\
# times       real      imaginary
1000000000.0  -2.3e-25   4.3e-26
1000000060.0   3.2e-26   1.2e-25
1000000120.0  -1.7e-25  -2.8e-25
1000000180.0  -7.6e-26  -8.9e-26
"""
        datafile = "testdata.txt"
        with open("testdata.txt", "w") as fp:
            fp.write(hetdata)

        het = HeterodynedData(datafile)

        assert len(het) == 4
        assert (het.data.real[0] == -2.3e-25) and (het.data.real[-1] == -7.6e-26)
        assert (het.data.imag[0] == 4.3e-26) and (het.data.imag[-1] == -8.9e-26)
        assert (het.times[0].value == 1000000000.0) and (
            het.times[-1].value == 1000000180.0
        )
        assert het.dt.value == 60.0
        assert het.sample_rate.value == 1.0 / 60.0

        assert het.cwinpy_heterodyne_dag_config is None
        assert het.heterodyne_arguments is None

        os.remove(datafile)  # clean up file

    def test_read_text_data_std(self):
        """
        Test that a valid file with standard deviations is read in correctly.
        """

        # create a data file to output
        hetdata = """\
# times       real      imaginary  std
1000000000.0  -2.3e-25   4.3e-26   1.1e-26
1000000060.0   3.2e-26   1.2e-25   2.1e-26
1000000120.0  -1.7e-25  -2.8e-25   1.5e-26
1000000180.0  -7.6e-26  -8.9e-26   1.3e-26
"""
        datafile = "testdata.txt"
        with open("testdata.txt", "w") as fp:
            fp.write(hetdata)

        het = HeterodynedData(datafile)

        assert len(het) == 4
        assert (het.data.real[0] == -2.3e-25) and (het.data.real[-1] == -7.6e-26)
        assert (het.data.imag[0] == 4.3e-26) and (het.data.imag[-1] == -8.9e-26)
        assert (het.stds[0] == 1.1e-26) and (het.stds[-1] == 1.3e-26)
        assert (het.vars[0] == (1.1e-26) ** 2) and (het.vars[-1] == (1.3e-26) ** 2)
        assert (het.times[0].value == 1000000000.0) and (
            het.times[-1].value == 1000000180.0
        )
        assert het.dt.value == 60.0
        assert het.sample_rate.value == 1.0 / 60.0

        os.remove(datafile)  # clean up file

    def test_remove_duplicate_data(self):
        """
        Test that duplicate data time stamps are removed.
        """

        # create a data file to output
        hetdata = """\
# times       real      imaginary  std
1000000000.0  -2.3e-25   4.3e-26   1.1e-26
1000000060.0   3.2e-26   1.2e-25   2.1e-26
1000000060.0   3.2e-26   1.2e-25   2.1e-26
1000000060.0   3.2e-26   1.2e-25   2.1e-26
1000000120.0  -1.7e-25  -2.8e-25   1.5e-26
1000000180.0  -7.6e-26  -8.9e-26   1.3e-26
1000000180.0  -7.6e-26  -8.9e-26   1.3e-26
"""
        datafile = "testdata.txt"
        with open("testdata.txt", "w") as fp:
            fp.write(hetdata)

        het = HeterodynedData(datafile)

        assert len(het) == 4
        assert (het.data.real[0] == -2.3e-25) and (het.data.real[-1] == -7.6e-26)
        assert (het.data.imag[0] == 4.3e-26) and (het.data.imag[-1] == -8.9e-26)
        assert (het.stds[0] == 1.1e-26) and (het.stds[-1] == 1.3e-26)
        assert (het.vars[0] == (1.1e-26) ** 2) and (het.vars[-1] == (1.3e-26) ** 2)
        assert (het.times[0].value == 1000000000.0) and (
            het.times[-1].value == 1000000180.0
        )
        assert het.dt.value == 60.0
        assert het.sample_rate.value == 1.0 / 60.0

        os.remove(datafile)  # clean up file

    def test_write_text_data(self):
        """
        Test that data can be correctly written (and re-read) from a text file.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=(1440, 2))

        het = HeterodynedData(data, times=times)

        for suffix in ["txt", "txt.gz"]:
            datafile = "testdata.{}".format(suffix)
            het.write(datafile)

            # read in data
            hetnew = HeterodynedData.read(datafile)

            assert np.array_equal(het.data, hetnew.data)
            assert np.array_equal(het.times, hetnew.times)

            # check things that the read-in data should not contain
            assert hetnew.detector is None
            assert hetnew.par is None

            os.remove(datafile)  # clean up file

    def test_write_text_data_std(self):
        """
        Test that data can be correctly written (and re-read) from a text file
        with the standard deviations also output.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=(1440, 2))
        stds = 1e-25 * np.ones_like(times)
        data = np.column_stack((data, stds))

        het = HeterodynedData(data, times=times)

        for suffix in ["txt", "txt.gz"]:
            datafile = "testdata.{}".format(suffix)
            het.write(datafile)

            # read in data
            hetnew = HeterodynedData.read(datafile)

            assert np.array_equal(het.data, hetnew.data)
            assert np.array_equal(het.times, hetnew.times)
            assert np.array_equal(het.stds, hetnew.stds)

            # check things that the read-in data should not contain
            assert hetnew.detector is None
            assert hetnew.par is None

            os.remove(datafile)  # clean up file

    def test_read_csv_data(self):
        """
        Test that a valid input CSV file is read in correctly.
        """

        # create a data file to output
        hetdata = """\
# times        real       imaginary
1000000000.0,  -2.3e-25,   4.3e-26
1000000060.0,   3.2e-26,   1.2e-25
1000000120.0,  -1.7e-25,  -2.8e-25
1000000180.0,  -7.6e-26,  -8.9e-26
"""
        datafile = "testdata.csv"
        with open("testdata.csv", "w") as fp:
            fp.write(hetdata)

        het = HeterodynedData(datafile)

        assert len(het) == 4
        assert (het.data.real[0] == -2.3e-25) and (het.data.real[-1] == -7.6e-26)
        assert (het.data.imag[0] == 4.3e-26) and (het.data.imag[-1] == -8.9e-26)
        assert (het.times[0].value == 1000000000.0) and (
            het.times[-1].value == 1000000180.0
        )
        assert het.dt.value == 60.0
        assert het.sample_rate.value == 1.0 / 60.0

        os.remove(datafile)  # clean up file

    def test_read_csv_data_std(self):
        """
        Test that a valid file with standard deviations is read in correctly.
        """

        # create a data file to output
        hetdata = """\
# times        real       imaginary   std
1000000000.0,  -2.3e-25,   4.3e-26,   1.1e-26
1000000060.0,   3.2e-26,   1.2e-25,   2.1e-26
1000000120.0,  -1.7e-25,  -2.8e-25,   1.5e-26
1000000180.0,  -7.6e-26,  -8.9e-26,   1.3e-26
"""
        datafile = "testdata.csv"
        with open("testdata.csv", "w") as fp:
            fp.write(hetdata)

        het = HeterodynedData(datafile)

        assert len(het) == 4
        assert (het.data.real[0] == -2.3e-25) and (het.data.real[-1] == -7.6e-26)
        assert (het.data.imag[0] == 4.3e-26) and (het.data.imag[-1] == -8.9e-26)
        assert (het.stds[0] == 1.1e-26) and (het.stds[-1] == 1.3e-26)
        assert (het.vars[0] == (1.1e-26) ** 2) and (het.vars[-1] == (1.3e-26) ** 2)
        assert (het.times[0].value == 1000000000.0) and (
            het.times[-1].value == 1000000180.0
        )
        assert het.dt.value == 60.0
        assert het.sample_rate.value == 1.0 / 60.0

        os.remove(datafile)  # clean up file

    def test_write_csv_data(self):
        """
        Test that data can be correctly written (and re-read) from a CSV file.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=(1440, 2))

        het = HeterodynedData(data, times=times)

        datafile = "testdata.csv"
        het.write(datafile)

        # read in data
        hetnew = HeterodynedData.read(datafile)

        assert np.array_equal(het.data, hetnew.data)
        assert np.array_equal(het.times, hetnew.times)

        # check things that the read-in data should not contain
        assert hetnew.detector is None
        assert hetnew.par is None

        os.remove(datafile)  # clean up file

    def test_write_csv_data_std(self):
        """
        Test that data can be correctly written (and re-read) from a CSV file
        with the standard deviations also output.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=(1440, 2))
        stds = 1e-25 * np.ones_like(times)
        data = np.column_stack((data, stds))

        het = HeterodynedData(data, times=times)

        datafile = "testdata.csv"
        het.write(datafile)

        # read in data
        hetnew = HeterodynedData.read(datafile)

        assert np.array_equal(het.data, hetnew.data)
        assert np.array_equal(het.times, hetnew.times)
        assert np.array_equal(het.stds, hetnew.stds)

        # check things that the read-in data should not contain
        assert hetnew.detector is None
        assert hetnew.par is None

        os.remove(datafile)  # clean up file

    def test_write_hdf_data(self):
        """
        Test that data can be correctly written (and re-read) from a HDF5 file.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=(1440, 2))
        det = "H1"

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

        parfile = "J0123+3456.par"

        # add content to the par file
        with open(parfile, "w") as fp:
            fp.write(parcontent)

        het = HeterodynedData(data, times=times, detector=det, par=parfile)

        for suffix in ["hdf5", "hdf", "h5"]:
            datafile = "testdata.{}".format(suffix)
            het.write(datafile, overwrite=True)

            # read in data
            hetnew = HeterodynedData.read(datafile)

            assert np.array_equal(het.data, hetnew.data)
            assert np.array_equal(het.times, hetnew.times)

            # check that detector and par file are read in correctly
            assert hetnew.detector == det
            for key in het.par.as_dict():
                if isinstance(hetnew.par[key], str):
                    assert hetnew.par[key] == het.par[key]
                else:
                    assert np.allclose(hetnew.par[key], het.par[key])

            # check version information is stored
            assert het.cwinpy_version == hetnew.cwinpy_version
            assert het.cwinpy_version == cwinpy.__version__

            os.remove(datafile)  # clean up file

        os.remove(parfile)

    def test_write_hdf_data_std(self):
        """
        Test that data can be correctly written (and re-read) from a HDF5 file
        with the standard deviations also output. Also, add an injection!
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=(1440, 2))
        stds = 1e-25 * np.ones_like(times)
        data = np.column_stack((data, stds))
        det = "H1"

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

        parfile = "J0123+3456.par"

        # add content to the par file
        with open(parfile, "w") as fp:
            fp.write(parcontent)

        het = HeterodynedData(data, times=times, detector=det, par=parfile, inject=True)

        for suffix in ["hdf5", "hdf", "h5"]:
            datafile = "testdata.{}".format(suffix)
            het.write(datafile, overwrite=True)

            # read in data
            hetnew = HeterodynedData.read(datafile)

            assert np.array_equal(het.data, hetnew.data)
            assert np.array_equal(het.times, hetnew.times)
            assert np.array_equal(het.stds, hetnew.stds)
            assert hetnew.injection is True
            assert np.array_equal(het.injection_data, hetnew.injection_data)

            # check that detector and par file are read in correctly
            assert hetnew.detector == det
            for key in het.par.as_dict():
                if key in hetnew.par.as_dict():
                    if isinstance(hetnew.par[key], str):
                        assert hetnew.par[key] == het.par[key]
                        assert hetnew.injpar[key] == het.injpar[key]
                    else:
                        assert np.allclose(hetnew.par[key], het.par[key])
                        assert np.allclose(hetnew.injpar[key], het.injpar[key])

            os.remove(datafile)  # clean up file

        os.remove(parfile)

    def test_merge_data(self):
        """
        Test merging multiple data sets during reading.
        """

        # create three sets of data
        times1 = np.linspace(1000000000.0, 1000086340.0, 1440)
        data1 = np.random.normal(0.0, 1e-25, size=(len(times1), 2))
        stds = 1e-25 * np.ones_like(times1)
        data1 = np.column_stack((data1, stds))

        times2 = np.linspace(999913600.0, 999999940.0, 1440)
        data2 = np.random.normal(0.0, 1e-25, size=(len(times2), 2))
        stds = 1e-25 * np.ones_like(times2)
        data2 = np.column_stack((data2, stds))

        # don't add standard deviations to third data set for now
        times3 = np.linspace(1000186400.0, 1000359140.0, 2880)
        data3 = np.random.normal(0.0, 1e-25, size=(len(times3), 2))

        parcontent1 = """\
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

        parfile1 = "J0123+3456.par"

        parcontent2 = """\
PSRJ     J0123+3457
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

        parfile2 = "J0123+3457.par"

        # add content to the par file
        for parfile, parcontent in zip(
            [parfile1, parfile2], [parcontent1, parcontent2]
        ):
            with open(parfile, "w") as fp:
                fp.write(parcontent)

        # test for overlapping times
        datafiles = []
        datalist = [data1, data2, data1]
        timeslist = [times1, times2, times1]

        N = len(datalist)

        for i in range(N):
            datafile = "testdata_H1_{}.hdf5".format(i)
            datafiles.append(datafile)

            # write out data
            het = HeterodynedData(
                datalist[i], times=timeslist[i], detector="H1", par=parfile1
            )
            het.write(datafile, overwrite=True)

        # read in data
        with pytest.raises(ValueError) as e:
            _ = HeterodynedData.read(datafiles)
        assert "Cannot merge overlapping data" in str(e)

        datalist = [data1, data2, data3]
        timeslist = [times1, times2, times3]

        # test for inconsistent detectors when merging
        for i, det in enumerate(["H1", "H1", "L1"]):
            # write out data
            het = HeterodynedData(
                datalist[i], times=timeslist[i], detector=det, par=parfile1
            )
            het.write(datafiles[i], overwrite=True)

        # read in data
        with pytest.raises(ValueError) as e:
            _ = HeterodynedData.read(datafiles)
        assert "Incompatible detectors" in str(e)

        # test for inconsistent pulsars
        for i in range(N):
            # write out data
            het = HeterodynedData(
                datalist[i],
                times=timeslist[i],
                detector="H1",
                par=(parfile1 if i == 0 else parfile2),
            )
            het.write(datafiles[i], overwrite=True)

        # read in data
        with pytest.raises(ValueError) as e:
            _ = HeterodynedData.read(datafiles)
        assert "Incompatible pulsars" in str(e)

        # test for inconsistent frequency scale factors
        for i in range(N):
            # write out data
            het = HeterodynedData(
                datalist[i],
                times=timeslist[i],
                detector="H1",
                par=parfile1,
                freqfactor=(2 if i == 0 else 1),
            )
            het.write(datafiles[i], overwrite=True)

        # read in data
        with pytest.raises(ValueError) as e:
            _ = HeterodynedData.read(datafiles)
        assert "Incompatible frequency factors" in str(e)

        # check for inconsistencies in whether variances were set or not
        for i in range(N):
            # write out data
            het = HeterodynedData(
                datalist[i],
                times=timeslist[i],
                detector="H1",
                par=parfile1,
                freqfactor=2,
            )
            het.write(datafiles[i], overwrite=True)

        # read in data
        with pytest.raises(ValueError) as e:
            _ = HeterodynedData.read(datafiles)
        assert "Incompatible setting of variances" in str(e)

        # make data sets have compatible variances settings
        stds = 1e-25 * np.ones_like(times3)
        data3 = np.column_stack((data3, stds))
        datalist[-1] = data3

        # check for inconsistent injection of a signal
        for i in range(N):
            # write out data
            het = HeterodynedData(
                datalist[i],
                times=timeslist[i],
                detector="H1",
                par=parfile1,
                freqfactor=2,
                inject=(True if i < (N - 1) else False),
            )
            het.write(datafiles[i], overwrite=True)

        # read in data
        with pytest.raises(ValueError) as e:
            _ = HeterodynedData.read(datafiles)
        assert "Incompatible injection times" in str(e)

        # create consistent files for merging and check the output
        hets = []
        for i in range(N):
            # write out data
            het = HeterodynedData(
                datalist[i],
                times=timeslist[i],
                detector="H1",
                par=parfile1,
                freqfactor=2,
                inject=True,
            )

            # add dummy heterodyne_arguments for testing
            het.heterodyne_arguments = {"dummy": "argument"}

            het.write(datafiles[i], overwrite=True)
            hets.append(het)  # store for comparisons

        # read in data
        newhet = HeterodynedData.read(datafiles)

        # test times are correct and sorted
        times = np.concatenate((times2, times1, times3))  # correct time order
        assert len(newhet) == len(times)
        assert np.array_equal(times, newhet.times.value)
        assert newhet.dt.value == np.min(np.diff(times))

        # test data is correct
        assert np.array_equal(
            newhet.data, np.concatenate([hets[i].data for i in [1, 0, 2]])
        )

        # test injection data
        assert newhet.injtimes.shape == (N, 2)
        assert np.allclose(
            newhet.injection_data,
            np.concatenate([hets[i].injection_data for i in [1, 0, 2]]),
        )

        # test heterodyne arguments
        assert len(newhet.heterodyne_arguments) == N
        assert all(
            [
                hetargs == {"dummy": "argument"}
                for hetargs in newhet.heterodyne_arguments
            ]
        )

        # remove par files
        for parfile in [parfile1, parfile2]:
            os.remove(parfile)

        # remove data files
        for datafile in datafiles:
            os.remove(datafile)

    def test_zero_data(self):
        """
        Test that data containing zeros is produced if only time stamps are
        provided.
        """

        # create "zero" data by only passing a set of times
        times = np.linspace(1000000000.0, 1000086340.0, 1440)

        het = HeterodynedData(times=times)

        assert len(het) == len(times)
        assert np.all(het.data == 0.0)

    def test_array_data(self):
        """
        Test passing the data as arrays containing times and data.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=(1440, 2))

        het = HeterodynedData(data, times=times)

        assert np.all(het.times.value == times)
        assert np.all(het.data.real == data[:, 0])
        assert np.all(het.data.imag == data[:, 1])
        assert het.dt.value == (times[1] - times[0])

    def test_bad_ephemeris_files(self):
        """
        Test passing bad solar system ephemeris files values.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=(1440, 2))

        with pytest.raises(IOError):
            HeterodynedData(data, times=times, ephemearth="kagskdgd")

        with pytest.raises(IOError):
            HeterodynedData(data, times=times, ephemsun="kagskdgd")

        with pytest.raises(TypeError):
            HeterodynedData(data, times=times, ephemearth=1.2)

        with pytest.raises(TypeError):
            HeterodynedData(data, times=times, ephemsun=13)

    def test_array_data_complex(self):
        """
        Test passing the data as arrays containing times and complex data.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=1440) + 1j * np.random.normal(
            0.0, 1e-25, size=1440
        )

        het = HeterodynedData(data, times=times)

        assert np.all(het.times.value == times)
        assert np.all(het.data.real == data.real)
        assert np.all(het.data.imag == data.imag)
        assert het.dt.value == (times[1] - times[0])

    def test_array_data_broken_lengths(self):
        """
        Test that failure occurs if the number of time stamps is different from the
        number of data points.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1439)
        data = np.random.normal(0.0, 1e-25, size=1440) + 1j * np.random.normal(
            0.0, 1e-25, size=1440
        )

        with pytest.raises(ValueError):
            HeterodynedData(data, times=times)

    def test_array_no_times(self):
        """
        Test that failure occurs if no time steps are passed.
        """

        data = np.random.normal(0.0, 1e-25, size=1440) + 1j * np.random.normal(
            0.0, 1e-25, size=1440
        )

        with pytest.raises(ValueError):
            HeterodynedData(data)

    def test_parse_detector(self):
        """
        Test parsing a detector name and a lal.Detector
        """

        from lal import Detector
        from lalpulsar import GetSiteInfo

        det = "BK"  # "bad" detector

        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=1440) + 1j * np.random.normal(
            0.0, 1e-25, size=1440
        )

        with pytest.raises(ValueError):
            HeterodynedData(data, times=times, detector=det)

        det = "H1"  # good detector
        laldet = GetSiteInfo("H1")

        het = HeterodynedData(data, times=times, detector=det)

        assert het.detector == det
        assert isinstance(het.laldetector, Detector)
        assert het.laldetector.frDetector.prefix == laldet.frDetector.prefix
        assert np.all((het.laldetector.response == laldet.response).flatten())

        # try passing the lal.Detector itself
        del het
        het = HeterodynedData(data, times=times, detector=laldet)
        assert het.detector == det
        assert isinstance(het.laldetector, Detector)
        assert het.laldetector.frDetector.prefix == laldet.frDetector.prefix
        assert np.all((het.laldetector.response == laldet.response).flatten())

    def test_parse_parfile(self):
        """
        Test parsing of a pulsar '.par' parameter file.
        """

        # set data
        times = np.linspace(1000000000.0, 1000086340.0, 1440)
        data = np.random.normal(0.0, 1e-25, size=1440) + 1j * np.random.normal(
            0.0, 1e-25, size=1440
        )

        # set detector
        det = "H1"

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

        # try passing a none str or PulsarParameterPy object
        parfile = 1
        with pytest.raises(TypeError):
            HeterodynedData(data, times=times, detector=det, par=parfile)

        parfile = "J0123+3456.par"

        # try reading parfile that doesn't exists
        with pytest.raises(IOError):
            HeterodynedData(data, times=times, detector=det, par=parfile)

        # add content to the par file
        with open(parfile, "w") as fp:
            fp.write(parcontent)

        het = HeterodynedData(data, times=times, detector=det, par=parfile)

        assert isinstance(het.par, PulsarParameters)
        assert len(het.par["F"]) == 2
        assert (het.par["F"][0] == 567.89) and (het.par["F"][1] == -1.2e-12)
        assert (
            (het.par["H0"] == 9.87e-26)
            and (het.par["COSIOTA"] == 0.3)
            and (het.par["PSI"] == 1.1)
            and (het.par["PHI0"] == 2.4)
        )
        assert het.par["RAJ"] == lal.TranslateHMStoRAD("01:23:45.6789")
        assert het.par["DECJ"] == lal.TranslateDMStoRAD("34:56:54.321")
        pepoch = lal.TranslateStringMJDTTtoGPS("56789")
        assert het.par["PEPOCH"] == (pepoch.gpsSeconds + 1e-9 * pepoch.gpsNanoSeconds)

        # pass parameters as PulsarParameters object
        del het

        par = PulsarParameters(parfile)
        het = HeterodynedData(data, times=times, detector=det, par=par)

        assert isinstance(het.par, PulsarParameters)
        assert len(het.par["F"]) == len(par["F"])
        assert (het.par["F"][0] == par["F"][0]) and (het.par["F"][1] == par["F"][1])
        assert (
            (het.par["H0"] == par["H0"])
            and (het.par["COSIOTA"] == par["COSIOTA"])
            and (het.par["PSI"] == par["PSI"])
            and (het.par["PHI0"] == par["PHI0"])
        )
        assert het.par["RAJ"] == par["RAJ"]
        assert het.par["DECJ"] == par["DECJ"]
        assert het.par["PEPOCH"] == par["PEPOCH"]

        os.remove(parfile)

    def test_running_median(self):
        """
        Test the running median calculation.
        """

        # set data
        times = np.linspace(1000000000.0, 1000001740, 30)
        data = np.random.normal(0.0, 1e-25, size=30) + 1j * np.random.normal(
            0.0, 1e-25, size=30
        )

        window = 1  # window is too short
        with pytest.raises(ValueError):
            het = HeterodynedData(data, times=times, window=window)

        window = 1.5  # window is not an integer
        with pytest.raises(TypeError):
            het = HeterodynedData(data, times=times, window=window)

        window = 30

        het = HeterodynedData(data, times=times, window=window)

        assert len(het.running_median) == len(het)
        assert het.running_median.real[0] == np.median(data.real[: (window // 2) + 1])
        assert het.running_median.imag[0] == np.median(data.imag[: (window // 2) + 1])
        assert het.running_median.real[len(data) // 2 - 1] == np.median(data.real)
        assert het.running_median.imag[len(data) // 2 - 1] == np.median(data.imag)
        assert het.running_median.real[-1] == np.median(data.real[-(window // 2) :])
        assert het.running_median.imag[-1] == np.median(data.imag[-(window // 2) :])
        assert len(het.subtract_running_median()) == len(het)
        assert het.subtract_running_median()[0] == (
            data[0]
            - (
                np.median(data.real[: (window // 2) + 1])
                + 1j * np.median(data.imag[: (window // 2) + 1])
            )
        )

    def test_outlier_removal(self):
        """
        Test the outlier removal algorithm.
        """

        # set data
        times = np.linspace(1000000000.0, 1000001740, 30)
        data = np.random.normal(0.0, 1.0, size=30) + 1j * np.random.normal(
            0.0, 1.0, size=30
        )

        # add outliers (one in the real part and one in the imaginary)
        data[10] = 20.0 + data.imag[10] * 1j
        data[20] = data.real[20] - 20.0 * 1j

        het = HeterodynedData(data, times=times)

        # try finding the outlier (and testing exceptions)
        thresh = "a"
        with pytest.raises(TypeError):
            _ = het.find_outliers(thresh=thresh)

        thresh = -1.0
        with pytest.raises(ValueError):
            _ = het.find_outliers(thresh=thresh)

        idxs = het.find_outliers()

        assert len(np.where(idxs == True)[0]) >= 2  # noqa: E712
        assert (10 in np.where(idxs == True)[0]) and (
            20 in np.where(idxs == True)[0]
        )  # noqa: E712

        # test removing the outlier automatically
        newhet = HeterodynedData(data, times=times, remove_outliers=True)
        assert len(newhet) == (len(data) - len(np.where(idxs == True)[0]))  # noqa: E712

    def test_bayesian_blocks(self):
        """
        Test Bayesian Blocks splitting.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)

        # create data from two obviously different Gaussian distributions
        sigma1 = 1.0
        data1 = np.random.normal(0.0, sigma1, size=720) + 1j * np.random.normal(
            0.0, sigma1, size=720
        )
        sigma2 = 50.0
        data2 = np.random.normal(0.0, sigma2, size=720) + 1j * np.random.normal(
            0.0, sigma2, size=720
        )

        data = np.concatenate((data1, data2))

        with pytest.raises(ValueError):
            # check error raise if "random" threshold is used
            het = HeterodynedData(data, times, bbthreshold="sdkgasnm")

        # check different thresholds
        for thresh in ["default", "trials", 1000.0]:
            het = HeterodynedData(data, times=times, bbthreshold=thresh)

            # check that a change point was found
            assert len(het.change_point_indices) > 0

            # check for a change point within +/- 2 of 720
            found = False
            for cp in het.change_point_indices:
                if 718 <= cp <= 722:
                    found = True
                    break

            assert found

        # check errors if re-running Bayesian blocks
        minlength = 5.2  # must be an integer
        with pytest.raises(TypeError):
            het.bbminlength = None
            het.bayesian_blocks(minlength=minlength)

        minlength = 0  # must be greater than 1
        with pytest.raises(ValueError):
            het.bbminlength = None
            het.bayesian_blocks(minlength=minlength)

        minlength = 5
        maxlength = 4  # maxlength must be greater than minlength
        with pytest.raises(ValueError):
            het.bbminlength = None
            het.bbmaxlength = None
            het.bayesian_blocks(minlength=minlength, maxlength=maxlength)

        # test re-splitting
        maxlength = 360
        het.bbmaxlength = None
        het.bayesian_blocks(maxlength=maxlength)
        for cl in het.chunk_lengths:
            assert cl <= maxlength

        del het

        # test no-splitting
        minlength = np.inf
        het = HeterodynedData(
            data, times=times, bbthreshold="default", bbminlength=minlength
        )

        assert len(het.chunk_lengths) == 1
        assert het.chunk_lengths[0] == len(het)

    def test_spectrum_plots(self):
        """
        Test the spectrogram, periodogram and power spectrum plots.
        """

        times1 = np.linspace(1000000000.0, 1000172740.0, 2 * 1440)
        data1 = np.random.normal(0.0, 1.0, size=2 * 1440) + 1j * np.random.normal(
            0.0, 1.0, size=2 * 1440
        )
        detector1 = "H1"

        times2 = np.linspace(1000000000.0, 1000172740.0, 2 * 1440)
        data2 = np.random.normal(0.0, 1.0, size=2 * 1440) + 1j * np.random.normal(
            0.0, 1.0, size=2 * 1440
        )
        detector2 = "L1"

        data = {
            detector1: HeterodynedData(data1, times=times1),
            detector2: HeterodynedData(data2, times=times2),
        }

        mhd = MultiHeterodynedData(data)

        # test errors
        with pytest.raises(ValueError):
            _ = mhd.spectrogram(dt="a")

        with pytest.raises(ValueError):
            _ = mhd.spectrogram(dt=200000)

        # with pytest.raises(TypeError):
        #    _ = mhd.spectrogram(overlap='a')

        with pytest.raises(ValueError):
            _ = mhd.spectrogram(overlap=-1)

        with pytest.raises(ValueError):
            _ = mhd.power_spectrum(average="a")

        # create a spectrogram
        freqs, power, stimes, fig = data[detector1].spectrogram(dt=3600)

        assert isinstance(fig, Figure)
        assert freqs.shape[0] == 60
        assert power.shape[0] == 60 and power.shape[1] == 95
        assert stimes.shape[0] == power.shape[1]

        # create a power spectrum
        freqs, power, fig = data[detector1].power_spectrum(dt=86400)

        assert isinstance(fig, Figure)
        assert power.shape[0] == len(data1) // 2
        assert freqs.shape[0] == power.shape[0]

        # create a periodogram
        freqs, power, fig = data[detector1].periodogram()

        assert isinstance(fig, Figure)
        assert power.shape[0] == len(data1)
        assert freqs.shape[0] == power.shape[0]

        # do the same, but with some data removed to test zero padding
        newdata = np.delete(data1, [10, 51, 780])
        newtimes = np.delete(times1, [10, 51, 780])

        newhet = HeterodynedData(newdata, times=newtimes, detector="H1")

        # create a power spectrum
        freqs, power, fig = newhet.power_spectrum(dt=86400)

        assert isinstance(fig, Figure)
        assert power.shape[0] == len(data1) // 2
        assert freqs.shape[0] == power.shape[0]

        # add a DCC signal and check it's at 0 Hz
        datadc = np.random.normal(5.0, 1.0, size=2 * 1440) + 1j * np.random.normal(
            5.0, 1.0, size=2 * 1440
        )

        newhet2 = HeterodynedData(datadc, times=times1, detector="H1")

        # create a power spectrum
        freqs, power, fig = newhet2.power_spectrum(dt=86400)

        assert freqs[np.argmax(power)] == 0.0

    def test_simulated_data(self):
        """
        Test that simulated data is produced as expected.
        """

        times = np.linspace(1000000000.0, 1000086340.0, 1440)

        # use invalid random seed
        with pytest.raises(ValueError):
            het1 = HeterodynedData(
                times=times, fakeasd=1e-24, detector="H1", fakeseed=-1
            )

        # create data using the same random seed and check it is identical
        het1 = HeterodynedData(
            times=times, fakeasd=1e-24, detector="H1", fakeseed=16273
        )
        het2 = HeterodynedData(
            times=times, fakeasd=1e-24, detector="H1", fakeseed=16273
        )

        assert np.array_equal(het1.data, het2.data)

        # create data using a different seed and check it is not the same
        het3 = HeterodynedData(
            times=times, fakeasd=1e-24, detector="H1", fakeseed=788532
        )

        assert not np.array_equal(het1.data, het3.data)

        del het1
        del het2
        del het3

        # do the same tests using a numpy RandomSeed
        seed1 = np.random.default_rng(875329)
        het1 = HeterodynedData(
            times=times, fakeasd=1e-24, detector="H1", fakeseed=seed1
        )
        seed2 = np.random.default_rng(875329)
        het2 = HeterodynedData(
            times=times, fakeasd=1e-24, detector="H1", fakeseed=seed2
        )

        assert np.array_equal(het1.data, het2.data)

        # check that passing the same RandomState again produces different data
        het3 = HeterodynedData(
            times=times, fakeasd=1e-24, detector="H1", fakeseed=seed1
        )

        assert not np.array_equal(het1.data, het3.data)

        del het1
        del het2
        del het3

        # check that using issigma returns data with the expected standard
        # deviation and mean
        sigma = 1e-25
        svar = sigma ** 2

        het1 = HeterodynedData(times=times, fakeasd=sigma, detector="H1", issigma=True)

        # standard error on the variance
        sevar = np.sqrt(2.0 * sigma ** 4 / (len(het1.data) - 1.0))

        # check variance is within +/- 3.5 sigma
        assert (svar - 3.5 * sevar < np.var(het1.data.real) < svar + 3.5 * sevar) and (
            svar - 3.5 * sevar < np.var(het1.data.imag) < svar + 3.5 * sevar
        )

        # standard error of the mean
        semean = sigma / np.sqrt(len(het1.data))

        assert (-3.5 * semean < np.mean(het1.data.real) < 3.5 * semean) and (
            -3.5 * semean < np.mean(het1.data.imag) < 3.5 * semean
        )

    def test_plot(self):
        """
        Test plotting function (and at the same time test fake noise generation)
        """

        # create an injection parameter file
        parcontent = """\
PSRJ    J0000+0000
RAJ     00:00:00.0
DECJ    00:00:00.0
F0      123.45
F1      1.2e-11
PEPOCH  56789.0
H0      1.5e-22
"""

        parfile = "test.par"
        with open(parfile, "w") as fp:
            fp.write(parcontent)

        # one point per 10 mins
        times = np.linspace(1000000000.0, 1000085800.0, 144)

        with pytest.raises(AttributeError):
            # if no parameter file is given, then generating fake data for a
            # particular detector should fail
            het = HeterodynedData(times=times, fakeasd="H1")

        # set the asd explicitly
        het = HeterodynedData(
            times=times, fakeasd=1e-24, detector="H1", par=parfile, inject=True
        )
        mhd = MultiHeterodynedData(het)

        # not allowed argument
        with pytest.raises(ValueError):
            fig = mhd.plot(which="blah")

        # test different plot types
        for which in ["abs", "REAL", "im", "Both"]:
            fig = mhd.plot(which=which)
            assert isinstance(fig[0], Figure)
            del fig

        # remove the par file
        os.remove(parfile)


def test_psd_wrapper():
    """
    Test PSDwrapper class.
    """

    import lalsimulation as lalsim

    psd = PSDwrapper(lalsim.SimNoisePSDAdvVirgo)

    with pytest.raises(ValueError):
        # no frequency supplied
        psd()

    with pytest.raises(RuntimeError):
        # PSD not the write format
        psd(100.0)

    psd = PSDwrapper(lalsim.SimNoisePSDaLIGOaLIGODesignSensitivityT1800044, f0=100.0)

    assert psd() == psd(100.0)
