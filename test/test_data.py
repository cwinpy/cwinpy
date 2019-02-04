"""
Test script for data.py classes.
"""

import os
import pytest
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy


from cwinpy import HeterodynedData


def test_data():
    """
    Test reading/parsing of data.
    """

    # test exception if no data or times are passed
    with pytest.raises(ValueError):
        het = HeterodynedData()

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
