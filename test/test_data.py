"""
Test script for data.py classes.
"""

import os
import pytest
import numpy as np
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy


from cwinpy import HeterodynedData


def test_no_data():
    """
    Test exception occurs if passing no data and no time stamps.
    """

    # test exception if no data or times are passed
    with pytest.raises(ValueError):
        het = HeterodynedData()
    

def test_broken_data():
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
    brokenfile = 'brokendata.txt'
    with open(brokenfile, 'w') as fp:
        fp.write(brokendata)

    with pytest.raises(IOError):
        het = HeterodynedData(brokenfile)

    os.remove(brokenfile)  # clean up file


def test_too_many_columns():
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
    brokenfile = 'brokendata.txt'
    with open(brokenfile, 'w') as fp:
        fp.write(brokendata)

    with pytest.raises(ValueError):
        het = HeterodynedData(brokenfile)

    os.remove(brokenfile)  # clean up file


def test_too_few_columns():
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
    brokenfile = 'brokendata.txt'
    with open(brokenfile, 'w') as fp:
        fp.write(brokendata)

    with pytest.raises(ValueError):
        het = HeterodynedData(brokenfile)

    os.remove(brokenfile)  # clean up file


def test_read_data():
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
    datafile = 'testdata.txt'
    with open('testdata.txt', 'w') as fp:
        fp.write(hetdata)

    het = HeterodynedData(datafile)

    assert len(het) == 4
    assert (het.data.real[0] == -2.3e-25) and (het.data.real[-1] == -7.6e-26)
    assert (het.data.imag[0] == 4.3e-26) and (het.data.imag[-1] == -8.9e-26)
    assert (het.times[0] == 1000000000.0) and (het.times[-1] == 1000000180.0)
    assert het.dt == 60.
    assert het.fs == 1./60.

    os.remove(datafile)  # clean up file


def test_read_data_std():
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
    datafile = 'testdata.txt'
    with open('testdata.txt', 'w') as fp:
        fp.write(hetdata)

    het = HeterodynedData(datafile)

    assert len(het) == 4
    assert (het.data.real[0] == -2.3e-25) and (het.data.real[-1] == -7.6e-26)
    assert (het.data.imag[0] == 4.3e-26) and (het.data.imag[-1] == -8.9e-26)
    assert (het.stds[0] == 1.1e-26) and (het.stds[-1] == 1.3e-26)
    assert (het.vars[0] == (1.1e-26)**2) and (het.vars[-1] == (1.3e-26)**2)
    assert (het.times[0] == 1000000000.0) and (het.times[-1] == 1000000180.0)
    assert het.dt == 60.
    assert het.fs == 1./60.

    os.remove(datafile)  # clean up file


def test_zero_data():
    """
    Test that data containing zeros is produced if only time stamps are
    provided.
    """

    # create "zero" data by only passing a set of times
    times = np.linspace(1000000000., 1000086340., 1440)

    het = HeterodynedData(times=times)

    assert len(het) == len(times)
    assert np.all(het.data == 0.0)


def test_array_data():
    """
    Test passing the data as arrays containing times and data.
    """

    times = np.linspace(1000000000., 1000086340., 1440)
    data = np.random.normal(0., 1e-25, size=(1440, 2))

    het = HeterodynedData(data, times=times)

    assert np.all(het.times == times)
    assert np.all(het.data.real == data[:, 0])
    assert np.all(het.data.imag == data[:, 1])
    assert het.dt == (times[1] - times[0])


def test_array_data_complex():
    """
    Test passing the data as arrays containing times and complex data.
    """

    times = np.linspace(1000000000., 1000086340., 1440)
    data = (np.random.normal(0., 1e-25, size=1440) +
            1j*np.random.normal(0., 1e-25, size=1440))

    het = HeterodynedData(data, times=times)

    assert np.all(het.times == times)
    assert np.all(het.data.real == data.real)
    assert np.all(het.data.imag == data.imag)
    assert het.dt == (times[1] - times[0])


def test_array_data_broken_lengths():
    """
    Test that failure occurs if the number of time stamps is different from the
    number of data points.
    """

    times = np.linspace(1000000000., 1000086340., 1439)
    data = (np.random.normal(0., 1e-25, size=1440) +
            1j*np.random.normal(0., 1e-25, size=1440))

    with pytest.raises(ValueError):
        het = HeterodynedData(data, times=times)


def test_array_no_times():
    """
    Test that failure occurs if no time steps are passed.
    """

    data = (np.random.normal(0., 1e-25, size=1440) +
            1j*np.random.normal(0., 1e-25, size=1440))

    with pytest.raises(ValueError):
        het = HeterodynedData(data)


def test_parse_detector():
    """
    Test parsing a detector name and a lal.Detector
    """

    from lal import Detector
    from lalpulsar import GetSiteInfo

    det = 'BK'  # "bad" detector

    times = np.linspace(1000000000., 1000086340., 1440)
    data = (np.random.normal(0., 1e-25, size=1440) +
            1j*np.random.normal(0., 1e-25, size=1440))

    with pytest.raises(ValueError):
        het = HeterodynedData(data, times=times, detector=det)

    det = 'H1'  # good detector
    laldet = GetSiteInfo('H1')

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
