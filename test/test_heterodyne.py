"""
Test code for Heterodyne class.
"""

import os
import shutil

import lal
import pytest
from cwinpy import Heterodyne


class TestHeterodyne(object):
    """
    Test the Heterodyne object.
    """

    @classmethod
    def setup_class(cls):
        # create dummy frame cache files
        cls.dummydir = "testing_frame_cache"
        os.makedirs(cls.dummydir, exist_ok=True)
        cls.dummy_cache_files = []
        for i in range(0, 5):
            dummyfile = os.path.join(
                cls.dummydir, "frame_cache_{0:01d}.cache".format(i)
            )
            cls.dummy_cache_files.append(dummyfile)
            with open(dummyfile, "w") as fp:
                fp.write("")

    @classmethod
    def teardown_class(cls):
        """
        Remove test simulation directory.
        """

        shutil.rmtree(cls.dummydir)

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

        # test the stride (default value)
        assert het.stride == 3600

        with pytest.raises(TypeError):
            het.stride = "kgsdg"

        with pytest.raises(TypeError):
            het.stride = 1.5

        with pytest.raises(ValueError):
            het.stride = 0

        stride = 1
        het.stride = stride
        assert het.stride == stride

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

    def test_frcache(self):
        """
        Test frame cache file setting.
        """

        with pytest.raises(TypeError):
            Heterodyne(frcache=1.2)

        with pytest.raises(ValueError):
            Heterodyne(frcache="lsgdfklg")

        with pytest.raises(TypeError):
            Heterodyne(frcache=[1, 2])

        het = Heterodyne(frcache=self.dummy_cache_files[0])
        assert het.frcache == self.dummy_cache_files[0]

        het.frcache = self.dummy_cache_files
        assert len(het.frcache) == len(self.dummy_cache_files)
        for i, df in enumerate(self.dummy_cache_files):
            assert df == het.frcache[i]

    def test_host(self):
        """
        Test host name.
        """

        with pytest.raises(TypeError):
            Heterodyne(host=1.2)

        with pytest.raises(RuntimeError):
            Heterodyne(host="+0--23oiyds")

        het = Heterodyne()

        assert het.host is None

        host = "https://www.google.com"
        het.host = host

        assert het.host == host

        host = "www.google.com"
        het.host = host
        assert het.host == host
