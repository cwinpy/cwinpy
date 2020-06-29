"""
Test code for Heterodyne class.
"""

import lal
import pytest
from cwinpy import Heterodyne


class TestHeterodyne(object):
    """
    Test the Heterodyne object.
    """

    def test_start_end(self):
        """
        Test for valid start and end times.
        """

        starttime = "blah"
        endtime = "blah"

        with pytest.raises(TypeError):
            Heterodyne(starttime, endtime)

        starttime = 100000000.0
        with pytest.raises(TypeError):
            Heterodyne(starttime, endtime)

        endtime = 100
        with pytest.raises(ValueError):
            Heterodyne(starttime, endtime)

        endtime = starttime + 86400
        het = Heterodyne(starttime, endtime)

        assert type(het.starttime) is int
        assert type(het.endtime) is int
        assert int(starttime) == het.starttime
        assert int(endtime) == het.endtime

        het = Heterodyne()
        assert het.starttime is None
        assert het.endtime is None

        # try setter
        het.starttime = 1000000001.1
        assert het.starttime == int(1000000001.1)

        # try setter
        het.endtime = 1000000002.1
        assert het.endtime == int(1000000002.1)

    def test_detector(self):
        """
        Test for valid detector.
        """

        detector = "lhsdlgda"
        with pytest.raises(ValueError):
            Heterodyne(detector=detector)

        detector = 348.23
        with pytest.raises(TypeError):
            Heterodyne(detector=detector)

        het = Heterodyne()

        assert het.detector is None
        assert het.laldetector is None

        # try setter
        detector = "H1"
        het.detector = detector

        assert het.detector == detector
        assert type(het.laldetector) is lal.Detector
        assert het.laldetector.frDetector.prefix == detector

    def test_frtype(self):
        """
        Test for valid frame type.
        """

        frtype = 1.0
        with pytest.raises(TypeError):
            Heterodyne(frtype=frtype)

        het = Heterodyne()
        assert het.frtype is None

        frtype = "H1_R"
        het.frtype = frtype
        assert het.frtype == frtype

    def test_channel(self):
        """
        Test for valid channel name.
        """

        channel = 1
        with pytest.raises(TypeError):
            Heterodyne(channel=channel)

        channel = "lsgdkgks"
        with pytest.raises(ValueError):
            Heterodyne(channel=channel)

        het = Heterodyne()

        assert het.channel is None
        assert het.detector is None

        detector = "L1"
        channel = "{}:GWOSC-4KHZ_R1_STRAIN".format(detector)

        het.detector = "H1"
        with pytest.raises(ValueError):
            het.channel = channel

        het.detector = None
        het.channel = channel

        assert het.channel == channel
        assert het.detector == detector
