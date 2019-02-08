"""
Test script for data.py classes.
"""

import os
import pytest
import numpy as np
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy
import lal

from cwinpy import HeterodynedData, MultiHeterodynedData


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

    # run through MultiHeterodynedData
    with pytest.raises(ValueError):
        mhet = MultiHeterodynedData(brokenfile)

    with pytest.raises(IOError):
        mhet = MultiHeterodynedData({'H1': brokenfile})

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


def test_parse_parfile():
    """
    Test parsing of a pulsar '.par' parameter file.
    """

    # set data
    times = np.linspace(1000000000., 1000086340., 1440)
    data = (np.random.normal(0., 1e-25, size=1440) +
            1j*np.random.normal(0., 1e-25, size=1440))

    # set detector
    det = 'H1'

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
        het = HeterodynedData(data, times=times, detector=det, par=parfile)

    parfile = 'J0123+3456.par'

    # try reading parfile that doesn't exists
    with pytest.raises(IOError):
        het = HeterodynedData(data, times=times, detector=det, par=parfile)

    # add content to the par file
    with open(parfile, 'w') as fp:
        fp.write(parcontent)

    het = HeterodynedData(data, times=times, detector=det, par=parfile)

    assert isinstance(het.par, PulsarParametersPy)
    assert len(het.par['F']) == 2
    assert (het.par['F'][0] == 567.89) and (het.par['F'][1] == -1.2e-12)
    assert ((het.par['H0'] == 9.87e-26) and (het.par['COSIOTA'] == 0.3) and
            (het.par['PSI'] == 1.1) and (het.par['PHI0'] == 2.4))
    assert het.par['RAJ'] == lal.TranslateHMStoRAD('01:23:45.6789')
    assert het.par['DECJ'] == lal.TranslateDMStoRAD('34:56:54.321')
    pepoch = lal.TranslateStringMJDTTtoGPS('56789')
    assert (het.par['PEPOCH'] == (pepoch.gpsSeconds +
                                  1e-9*pepoch.gpsNanoSeconds))

    # pass parameters as PulsarParametersPy object
    del het

    par = PulsarParametersPy(parfile)
    het = HeterodynedData(data, times=times, detector=det, par=par)

    assert isinstance(het.par, PulsarParametersPy)
    assert len(het.par['F']) == len(par['F'])
    assert ((het.par['F'][0] == par['F'][0]) and
            (het.par['F'][1] == par['F'][1]))
    assert ((het.par['H0'] == par['H0']) and
            (het.par['COSIOTA'] == par['COSIOTA']) and
            (het.par['PSI'] == par['PSI']) and
            (het.par['PHI0'] == par['PHI0']))
    assert het.par['RAJ'] == par['RAJ']
    assert het.par['DECJ'] == par['DECJ']
    assert het.par['PEPOCH'] == par['PEPOCH']

    os.remove(parfile)


def test_running_median():
    """
    Test the running median calculation.
    """

    # set data
    times = np.linspace(1000000000., 1000001740, 30)
    data = (np.random.normal(0., 1e-25, size=30) +
            1j*np.random.normal(0., 1e-25, size=30))

    window = 1  # window is too short
    with pytest.raises(ValueError):
        het = HeterodynedData(data, times=times, window=window)

    window = 30

    het = HeterodynedData(data, times=times, window=window)

    assert len(het.running_median) == len(het)
    assert het.running_median.real[0] == np.median(data.real[:(window//2)+1])
    assert het.running_median.imag[0] == np.median(data.imag[:(window//2)+1])
    assert het.running_median.real[len(data)//2-1] == np.median(data.real)
    assert het.running_median.imag[len(data)//2-1] == np.median(data.imag)
    assert het.running_median.real[-1] == np.median(data.real[-(window//2):])
    assert het.running_median.imag[-1] == np.median(data.imag[-(window//2):])
    assert len(het.subtract_running_median()) == len(het)
    assert (het.subtract_running_median()[0] ==
            (data[0] - (np.median(data.real[:(window//2)+1]) + 
                        1j*np.median(data.imag[:(window//2)+1]))))
