"""
Test code for Heterodyne class.
"""

import os
import shutil
import subprocess as sp

import lal
import pytest
from astropy.utils.data import download_file
from cwinpy import Heterodyne
from gwosc.api import DEFAULT_URL as GWOSC_DEFAULT_HOST
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy
from lalpulsar.simulateHeterodynedCW import DOWNLOAD_URL


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
                fp.write("blah\n")

        # create some fake data frames using lalapps_Makefakedata_v5
        mfd = shutil.which("lalapps_Makefakedata_v5")

        cls.fakedatadir = "testing_fake_frame_cache"
        cls.fakedatadetectors = ["H1", "L1"]
        cls.fakedatachannels = [
            "{}:FAKE_DATA".format(det) for det in cls.fakedatadetectors
        ]
        cls.fakedatastarts = [1000000000, 1000000000 + 86400]
        cls.fakedataduration = 86400

        os.makedirs(cls.fakedatadir, exist_ok=True)

        cls.fakedatabandwidth = 2  # Hz
        sqrtSn = 1e-24  # noise amplitude spectral density
        cls.fakedataname = "FAKEGLITCH"

        f0 = 1.23456 / 2.0  # source rotation frequency (Hz)
        f1 = 9.87654e-11 / 2.0  # source rotational frequency derivative (Hz/s)
        alpha = 0.0  # source right ascension (rads)
        delta = 0.5  # source declination (rads)
        pepoch = 1000000000  # frequency epoch (GPS)

        # GW parameters
        h0 = 3.0e-24  # GW amplitude
        phi0 = 1.0  # GW initial phase (rads)
        cosiota = 0.1  # cosine of inclination angle
        psi = 0.5  # GW polarisation angle (rads)

        inj = "{{Alpha={}; Delta={}; Freq={}; f1dot={}; refTime={}; h0={}; cosi={}; psi={}; phi0={};}}".format(
            alpha, delta, f0 * 2, f1 * 2, pepoch, h0, cosiota, psi, phi0
        )

        cls.fakepulsarpar = PulsarParametersPy()
        cls.fakepulsarpar["H0"] = h0
        cls.fakepulsarpar["PHI0"] = phi0 / 2.0
        cls.fakepulsarpar["PSI"] = psi
        cls.fakepulsarpar["COSIOTA"] = cosiota
        cls.fakepulsarpar["F"] = [f0, f1]
        cls.fakepulsarpar["RAJ"] = alpha
        cls.fakepulsarpar["DECJ"] = delta
        cls.fakepulsarpar["PEPOCH"] = pepoch

        # set ephemeris files
        efile = download_file(
            DOWNLOAD_URL.format("earth00-40-DE405.dat.gz"), cache=True
        )
        sfile = download_file(DOWNLOAD_URL.format("sun00-40-DE405.dat.gz"), cache=True)

        for datastart in cls.fakedatastarts:
            for i in range(len(cls.fakedatachannels)):
                cmds = [
                    "-F",
                    cls.fakedatadir,
                    "--outFrChannels={}".format(cls.fakedatachannels[i]),
                    "-I",
                    cls.fakedatadetectors[i],
                    "--sqrtSX={0:.1e}".format(sqrtSn),
                    "-G",
                    str(datastart),
                    "--duration={}".format(cls.fakedataduration),
                    "--Band={}".format(cls.fakedatabandwidth),
                    "--fmin",
                    "0",
                    '--injectionSources="{}"'.format(inj),
                    "--outLabel={}".format(cls.fakedataname),
                    '--ephemEarth="{}"'.format(efile),
                    '--ephemSun="{}"'.format(sfile),
                ]

                # run makefakedata
                sp.run([mfd] + cmds)

        # create dummy segment file
        cls.dummysegments = [(1000000000, 1000000600), (1000000800, 1000000900)]
        cls.dummysegmentfile = os.path.join(cls.fakedatadir, "fakesegments.txt")
        with open(cls.dummysegmentfile, "w") as fp:
            for segs in cls.dummysegments:
                fp.write("{} {}\n".format(segs[0], segs[1]))

    @classmethod
    def teardown_class(cls):
        """
        Remove test simulation directory.
        """

        shutil.rmtree(cls.dummydir)
        shutil.rmtree(cls.fakedatadir)

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

    def test_frametype(self):
        """
        Test for valid frame type.
        """

        frametype = 1.0
        with pytest.raises(TypeError):
            Heterodyne(frametype=frametype)

        het = Heterodyne()
        assert het.frametype is None

        frametype = "H1_R"
        het.frametype = frametype
        assert het.frametype == frametype

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

    def test_framecache(self):
        """
        Test frame cache file setting.
        """

        with pytest.raises(TypeError):
            Heterodyne(framecache=1.2)

        with pytest.raises(ValueError):
            Heterodyne(framecache="lsgdfklg")

        with pytest.raises(TypeError):
            Heterodyne(framecache=[1, 2])

        het = Heterodyne(framecache=self.dummy_cache_files[0])
        assert het.framecache == self.dummy_cache_files[0]

        het.framecache = self.dummy_cache_files
        assert len(het.framecache) == len(self.dummy_cache_files)
        for i, df in enumerate(self.dummy_cache_files):
            assert df == het.framecache[i]

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

    def test_get_frame_data(self):
        """
        Test reading of local frame data.
        """

        het = Heterodyne()

        with pytest.raises(ValueError):
            # no start or end time set
            het.get_frame_data()

        with pytest.raises(ValueError):
            # no start time set
            het.get_frame_data(endtime=1000000000)

        with pytest.raises(ValueError):
            # no channel set
            het.get_frame_data(starttime=1000000000, endtime=1000084600)

        with pytest.raises(IOError):
            # invalid file name
            het.get_frame_data(
                starttime=1000000000,
                endtime=1000084600,
                channel=self.fakedatachannels[0],
                framecache="jhsdklgdks.txt",
            )

        with pytest.raises(IOError):
            # cache file contains invalid frames
            het.get_frame_data(
                starttime=1000000000,
                endtime=1000000000 + 2 * 86400,
                framecache=self.dummy_cache_files[0],
                site="H1",
                channel=self.fakedatachannels[0],
            )

        # test reading files/generating a local cache list (all files)
        cachefile = os.path.join(self.fakedatadir, "frcache.txt")
        data = het.get_frame_data(
            starttime=1000000000,
            endtime=1000000000 + 2 * 86400,
            framecache=self.fakedatadir,
            site="H1",
            outputframecache=cachefile,
            channel=self.fakedatachannels[0],
        )

        assert int(data.t0.value) == self.fakedatastarts[0]
        assert data.dt.value == 1 / (2 * (self.fakedatabandwidth))

        with open(cachefile, "r") as fp:
            cachedata = [fl.strip() for fl in fp.readlines()]

        assert len(cachedata) == 2
        for i in range(len(cachedata)):
            assert "{}-{}_{}-{}-{}.gwf".format(
                self.fakedatadetectors[0][0],
                self.fakedatadetectors[0],
                self.fakedataname,
                self.fakedatastarts[i],
                self.fakedataduration,
            ) == os.path.basename(cachedata[i])

        # test reading files from cache file
        data = het.get_frame_data(
            starttime=1000000000,
            endtime=1000000000 + 2 * 86400,
            framecache=cachefile,
            site="H1",
            channel=self.fakedatachannels[0],
        )

        assert int(data.t0.value) == self.fakedatastarts[0]
        assert data.dt.value == 1 / (2 * (self.fakedatabandwidth))

        with pytest.raises(IOError):
            # try reading data outside of range
            het.get_frame_data(
                starttime=900000000,
                endtime=900000000 + 2 * 86400,
                framecache=cachefile,
                site="H1",
                channel=self.fakedatachannels[0],
            )

        with pytest.raises(IOError):
            # try reading data from the wrong channel
            het.get_frame_data(
                starttime=1000000000,
                endtime=1000000000 + 2 * 86400,
                framecache=cachefile,
                site="H1",
                channel=self.fakedatachannels[1],
            )

        del het
        del data

        # test reading from GWOSC
        het = Heterodyne()
        data = het.get_frame_data(
            starttime=1126259460, endtime=1126259464, host=GWOSC_DEFAULT_HOST, site="H1"
        )

        assert int(data.t0.value) == 1126259460
        assert data.dt.value == 1 / 4096
        assert len(data) == 16384

    @pytest.mark.disable_socket
    def test_get_frame_data_no_internet(self):
        # test exception if not able to access GWOSC data
        het = Heterodyne()
        with pytest.raises(IOError):
            het.get_frame_data(
                site="H1",
                starttime=1126259460,
                endtime=1126259464,
                host=GWOSC_DEFAULT_HOST,
            )

    def test_get_segments(self):
        """
        Test reading segment list file.
        """

        het = Heterodyne()

        segments = het.get_segment_list(segmentfile=self.dummysegmentfile)

        assert len(segments) == len(self.dummysegments)
        for sega, segb in zip(segments, self.dummysegments):
            for sa, sb in zip(sega, segb):
                assert sa == sb

        het = Heterodyne(segmentlist=self.dummysegments)

        assert len(segments) == len(self.dummysegments)
        for sega, segb in zip(segments, self.dummysegments):
            for sa, sb in zip(sega, segb):
                assert sa == sb

        with pytest.raises(IOError):
            # no existent segment file
            Heterodyne(segmentlist="klsghdfkdgskd")

        with pytest.raises(TypeError):
            Heterodyne(outputsegmentlist=1)

        with pytest.raises(TypeError):
            Heterodyne(appendsegmentlist=1.1)

        with pytest.raises(TypeError):
            Heterodyne(includeflags=1)

        with pytest.raises(TypeError):
            Heterodyne(excludeflags=1)

        with pytest.raises(TypeError):
            Heterodyne(segmentserver=1)

        het = Heterodyne(
            starttime=900000000, endtime=910000000, segmentlist=self.dummysegmentfile
        )

        assert len(het.segments) == 0

        het = Heterodyne(
            starttime=1000000850, endtime=1000001000, segmentlist=self.dummysegmentfile
        )

        assert len(het.segments) == 1
        assert het.segments[0][0] == 1000000850
        assert het.segments[0][1] == 1000000900

        het = Heterodyne(
            starttime=1000000100, endtime=1000000700, segmentlist=self.dummysegmentfile
        )

        assert len(het.segments) == 1
        assert het.segments[0][0] == 1000000100
        assert het.segments[0][1] == 1000000600
