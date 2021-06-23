"""
Test code for Heterodyne class.
"""

import os
import shutil
import subprocess as sp

import lal
import numpy as np
import pytest
from astropy.utils.data import download_file
from cwinpy import HeterodynedData
from cwinpy.heterodyne import Heterodyne, heterodyne
from cwinpy.signal import HeterodynedCWSimulator
from cwinpy.utils import LAL_EPHEMERIS_URL
from gwosc.api import DEFAULT_URL as GWOSC_DEFAULT_HOST
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy


def relative_difference(data, model):
    """
    Compute the relative difference between data and model.
    """

    return np.abs(data - model) / np.abs(model)


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
        cls.fakedatastarts = [
            1000000000,
            1000000000 + 86400 + 234,
            1000000000 + 86400 * 2 + 11923,
            1000000000 + 86400 * 3 + 32955,
        ]
        cls.fakedataduration = [86400, 34, 49731, 5004]

        os.makedirs(cls.fakedatadir, exist_ok=True)

        cls.fakedatabandwidth = 8  # Hz
        sqrtSn = 1e-29  # noise amplitude spectral density
        cls.fakedataname = "FAKEDATA"

        # Create two pulsars to inject: one isolated and one binary
        cls.fakepulsarpar = []

        # requirements for Makefakedata pulsar input files
        isolatedstr = """\
Alpha = {alpha}
Delta = {delta}
Freq = {f0}
f1dot = {f1}
f2dot = {f2}
refTime = {pepoch}
h0 = {h0}
cosi = {cosi}
psi = {psi}
phi0 = {phi0}
"""

        binarystr = """\
orbitasini = {asini}
orbitPeriod = {period}
orbitTp = {Tp}
orbitArgp = {argp}
orbitEcc = {ecc}
"""

        transientstr = """\
transientWindowType = {wintype}
transientStartTime = {tstart}
transientTau = {tau}
"""

        # FIRST PULSAR (ISOLATED)
        f0 = 6.9456 / 2.0  # source rotation frequency (Hz)
        f1 = -9.87654e-11 / 2.0  # source rotational frequency derivative (Hz/s)
        f2 = 2.34134e-18 / 2.0  # second frequency derivative (Hz/s^2)
        alpha = 0.0  # source right ascension (rads)
        delta = 0.5  # source declination (rads)
        pepoch = 1000000000  # frequency epoch (GPS)

        # GW parameters
        h0 = 3.0e-24  # GW amplitude
        phi0 = 1.0  # GW initial phase (rads)
        cosiota = 0.1  # cosine of inclination angle
        psi = 0.5  # GW polarisation angle (rads)

        mfddic = {
            "alpha": alpha,
            "delta": delta,
            "f0": 2 * f0,
            "f1": 2 * f1,
            "f2": 2 * f2,
            "pepoch": pepoch,
            "h0": h0,
            "cosi": cosiota,
            "psi": psi,
            "phi0": phi0,
        }

        cls.fakepulsarpar.append(PulsarParametersPy())
        cls.fakepulsarpar[0]["PSRJ"] = "J0000+0000"
        cls.fakepulsarpar[0]["H0"] = h0
        cls.fakepulsarpar[0]["PHI0"] = phi0 / 2.0
        cls.fakepulsarpar[0]["PSI"] = psi
        cls.fakepulsarpar[0]["COSIOTA"] = cosiota
        cls.fakepulsarpar[0]["F"] = [f0, f1, f2]
        cls.fakepulsarpar[0]["RAJ"] = alpha
        cls.fakepulsarpar[0]["DECJ"] = delta
        cls.fakepulsarpar[0]["PEPOCH"] = pepoch
        cls.fakepulsarpar[0]["EPHEM"] = "DE405"
        cls.fakepulsarpar[0]["UNITS"] = "TDB"

        cls.fakepardir = "testing_fake_par_dir"
        os.makedirs(cls.fakepardir, exist_ok=True)
        cls.fakeparfile = []
        cls.fakeparfile.append(os.path.join(cls.fakepardir, "J0000+0000.par"))
        cls.fakepulsarpar[0].pp_to_par(cls.fakeparfile[-1])

        injfile = os.path.join(cls.fakepardir, "inj.dat")
        with open(injfile, "w") as fp:
            fp.write("[Pulsar 1]\n")
            fp.write(isolatedstr.format(**mfddic))
            fp.write("\n")

        # SECOND PULSAR (BINARY SYSTEM)
        f0 = 3.8654321 / 2.0  # source rotation frequency (Hz)
        f1 = 9.87654e-13 / 2.0  # source rotational frequency derivative (Hz/s)
        f2 = -1.34134e-20 / 2.0  # second frequency derivative (Hz/s^2)
        alpha = 1.3  # source right ascension (rads)
        delta = -0.4  # source declination (rads)
        pepoch = 1000086400  # frequency epoch (GPS)

        # GW parameters
        h0 = 7.5e-25  # GW amplitude
        phi0 = 0.7  # GW initial phase (rads)
        cosiota = 0.6  # cosine of inclination angle
        psi = 1.1  # GW polarisation angle (rads)

        # binary parameters
        asini = 1.4  # projected semi-major axis (ls)
        period = 0.1 * 86400  # orbital period (s)
        Tp = 999992083  # time of periastron (GPS)
        argp = 0.0  # argument of perisatron (rad)
        ecc = 0.09  # the orbital eccentricity

        mfddic = {
            "alpha": alpha,
            "delta": delta,
            "f0": 2 * f0,
            "f1": 2 * f1,
            "f2": 2 * f2,
            "pepoch": pepoch,
            "h0": h0,
            "cosi": cosiota,
            "psi": psi,
            "phi0": phi0,
        }

        mfdbindic = {
            "asini": asini,
            "Tp": Tp,
            "period": period,
            "argp": argp,
            "ecc": ecc,
        }

        cls.fakepulsarpar.append(PulsarParametersPy())
        cls.fakepulsarpar[1]["PSRJ"] = "J1111+1111"
        cls.fakepulsarpar[1]["H0"] = h0
        cls.fakepulsarpar[1]["PHI0"] = phi0 / 2.0
        cls.fakepulsarpar[1]["PSI"] = psi
        cls.fakepulsarpar[1]["COSIOTA"] = cosiota
        cls.fakepulsarpar[1]["F"] = [f0, f1, f2]
        cls.fakepulsarpar[1]["RAJ"] = alpha
        cls.fakepulsarpar[1]["DECJ"] = delta
        cls.fakepulsarpar[1]["PEPOCH"] = pepoch
        cls.fakepulsarpar[1]["BINARY"] = "BT"
        cls.fakepulsarpar[1]["E"] = ecc
        cls.fakepulsarpar[1]["A1"] = asini
        cls.fakepulsarpar[1]["T0"] = Tp
        cls.fakepulsarpar[1]["OM"] = argp
        cls.fakepulsarpar[1]["PB"] = period
        cls.fakepulsarpar[1]["EPHEM"] = "DE405"
        cls.fakepulsarpar[1]["UNITS"] = "TDB"

        cls.fakeparfile.append(os.path.join(cls.fakepardir, "J1111+1111.par"))
        cls.fakepulsarpar[1].pp_to_par(cls.fakeparfile[-1])

        with open(injfile, "a") as fp:
            fp.write("[Pulsar 2]\n")
            fp.write(isolatedstr.format(**mfddic))
            fp.write(binarystr.format(**mfdbindic))
            fp.write("\n")

        # THIRD PULSAR (GLITCHING PULSAR)
        f0 = 5.3654321 / 2.0  # source rotation frequency (Hz)
        f1 = -4.57654e-10 / 2.0  # source rotational frequency derivative (Hz/s)
        f2 = 1.34134e-18 / 2.0  # second frequency derivative (Hz/s^2)
        alpha = 4.6  # source right ascension (rads)
        delta = -0.9  # source declination (rads)
        pepoch = 1000000000 + 1.5 * 86400  # frequency epoch (GPS)

        # glitch parameters
        df0 = 0.0001  # EM glitch frequency jump
        df1 = 1.2e-11  # EM glitch frequency derivative jump
        df2 = -4.5e-19  # EM glitch frequency second derivative jump
        dphi = 1.1  # EM glitch phase offset
        glepoch = pepoch  # glitch epoch

        # GW parameters
        h0 = 8.7e-25  # GW amplitude
        phi0 = 0.142  # GW initial phase (rads)
        cosiota = -0.3  # cosine of inclination angle
        psi = 0.52  # GW polarisation angle (rads)

        # binary parameters
        asini = 2.9  # projected semi-major axis (ls)
        period = 0.3 * 86400  # orbital period (s)
        Tp = 999995083  # time of periastron (GPS)
        argp = 0.5  # argument of perisatron (rad)
        ecc = 0.09  # the orbital eccentricity

        # for MFD I need to create this as two transient pulsars using a
        # rectangular window cutting before and after the glitch
        mfddic = {
            "alpha": alpha,
            "delta": delta,
            "f0": 2 * f0,
            "f1": 2 * f1,
            "f2": 2 * f2,
            "pepoch": pepoch,
            "h0": h0,
            "cosi": cosiota,
            "psi": psi,
            "phi0": phi0,
        }

        mfdbindic = {
            "asini": asini,
            "Tp": Tp,
            "period": period,
            "argp": argp,
            "ecc": ecc,
        }

        mfdtransientdic = {
            "wintype": "rect",
            "tstart": cls.fakedatastarts[0],
            "tau": cls.fakedatastarts[1]
            + cls.fakedataduration[1]
            - cls.fakedatastarts[0],
        }

        # signal before the glitch
        with open(injfile, "a") as fp:
            fp.write("[Pulsar 3]\n")
            fp.write(isolatedstr.format(**mfddic))
            fp.write(binarystr.format(**mfdbindic))
            fp.write(transientstr.format(**mfdtransientdic))
            fp.write("\n")

        mfddic["f0"] = 2 * (f0 + df0)
        mfddic["f1"] = 2 * (f1 + df1)
        mfddic["f2"] = 2 * (f2 + df2)
        mfddic["phi0"] = phi0 + 2 * dphi

        mfdtransientdic["tstart"] = cls.fakedatastarts[-2]
        mfdtransientdic["tau"] = (
            cls.fakedatastarts[-1] + cls.fakedataduration[-1] - cls.fakedatastarts[-2]
        )

        # signal after the glitch
        with open(injfile, "a") as fp:
            fp.write("[Pulsar 4]\n")
            fp.write(isolatedstr.format(**mfddic))
            fp.write(binarystr.format(**mfdbindic))
            fp.write(transientstr.format(**mfdtransientdic))

        cls.fakepulsarpar.append(PulsarParametersPy())
        cls.fakepulsarpar[2]["PSRJ"] = "J2222+2222"
        cls.fakepulsarpar[2]["H0"] = h0
        cls.fakepulsarpar[2]["PHI0"] = phi0 / 2.0
        cls.fakepulsarpar[2]["PSI"] = psi
        cls.fakepulsarpar[2]["COSIOTA"] = cosiota
        cls.fakepulsarpar[2]["F"] = [f0, f1, f2]
        cls.fakepulsarpar[2]["RAJ"] = alpha
        cls.fakepulsarpar[2]["DECJ"] = delta
        cls.fakepulsarpar[2]["PEPOCH"] = pepoch
        cls.fakepulsarpar[2]["BINARY"] = "BT"
        cls.fakepulsarpar[2]["E"] = ecc
        cls.fakepulsarpar[2]["A1"] = asini
        cls.fakepulsarpar[2]["T0"] = Tp
        cls.fakepulsarpar[2]["OM"] = argp
        cls.fakepulsarpar[2]["PB"] = period
        cls.fakepulsarpar[2]["EPHEM"] = "DE405"
        cls.fakepulsarpar[2]["UNITS"] = "TDB"
        cls.fakepulsarpar[2]["GLEP"] = [glepoch]
        cls.fakepulsarpar[2]["GLF0"] = [df0]
        cls.fakepulsarpar[2]["GLF1"] = [df1]
        cls.fakepulsarpar[2]["GLF2"] = [df2]
        cls.fakepulsarpar[2]["GLPH"] = [dphi / (2 * np.pi)]

        cls.fakeparfile.append(os.path.join(cls.fakepardir, "J2222+2222.par"))
        cls.fakepulsarpar[2].pp_to_par(cls.fakeparfile[-1])

        # set ephemeris files
        efile = download_file(
            LAL_EPHEMERIS_URL.format("earth00-40-DE405.dat.gz"), cache=True
        )
        sfile = download_file(
            LAL_EPHEMERIS_URL.format("sun00-40-DE405.dat.gz"), cache=True
        )

        for j, datastart in enumerate(cls.fakedatastarts):
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
                    "--duration={}".format(cls.fakedataduration[j]),
                    "--Band={}".format(cls.fakedatabandwidth),
                    "--fmin",
                    "0",
                    '--injectionSources="{}"'.format(injfile),
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
        shutil.rmtree(cls.fakepardir)
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
                endtime=1000000000 + 3 * 86400,
                framecache=self.dummy_cache_files[0],
                site="H1",
                channel=self.fakedatachannels[0],
            )

        # test reading files/generating a local cache list (all files)
        cachefile = os.path.join(self.fakedatadir, "frcache.txt")
        data = het.get_frame_data(
            starttime=1000000000,
            endtime=1000000000 + 86400,
            framecache=self.fakedatadir,
            site="H1",
            outputframecache=cachefile,
            channel=self.fakedatachannels[0],
        )

        assert int(data.t0.value) == self.fakedatastarts[0]
        assert data.dt.value == 1 / (2 * (self.fakedatabandwidth))

        with open(cachefile, "r") as fp:
            cachedata = [fl.strip() for fl in fp.readlines()]

        assert len(cachedata) == 1
        for i in range(len(cachedata)):
            assert (
                "{}-{}_{}-{}-{}.gwf".format(
                    self.fakedatadetectors[0][0],
                    self.fakedatadetectors[0],
                    self.fakedataname,
                    self.fakedatastarts[i],
                    self.fakedataduration[i],
                )
                == os.path.basename(cachedata[i])
            )

        # test reading files from cache file
        data = het.get_frame_data(
            starttime=1000000000,
            endtime=1000000000 + 86400,
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
                endtime=1000000000 + 86400,
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
        # test None if not able to access GWOSC data
        het = Heterodyne()
        data = het.get_frame_data(
            site="H1", starttime=1126259460, endtime=1126259464, host=GWOSC_DEFAULT_HOST
        )

        assert data is None

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

    def test_set_pulsars(self, capsys):
        """
        Test setting of pulsar parameter files.
        """

        with pytest.raises(TypeError):
            Heterodyne(pulsarfiles=1.2)

        het = Heterodyne(pulsarfiles=self.fakepardir)

        assert sorted(list(het.pulsarfiles.keys())) == [
            "J0000+0000",
            "J1111+1111",
            "J2222+2222",
        ]
        assert sorted(list(het.pulsarfiles.values())) == sorted(
            [
                os.path.realpath(self.fakeparfile[0]),
                os.path.realpath(self.fakeparfile[1]),
                os.path.realpath(self.fakeparfile[2]),
            ]
        )
        assert sorted(het.pulsars) == ["J0000+0000", "J1111+1111", "J2222+2222"]

        het = Heterodyne(pulsarfiles=self.fakeparfile[0])
        assert het.pulsarfiles == {"J0000+0000": self.fakeparfile[0]}
        assert het.pulsars == ["J0000+0000"]

        het = Heterodyne(pulsarfiles=self.fakeparfile[1])
        assert het.pulsarfiles == {"J1111+1111": self.fakeparfile[1]}
        assert het.pulsars == ["J1111+1111"]

        het = Heterodyne(pulsarfiles=self.fakeparfile[2])
        assert het.pulsarfiles == {"J2222+2222": self.fakeparfile[2]}
        assert het.pulsars == ["J2222+2222"]

        het = Heterodyne(pulsarfiles=[self.fakeparfile[0]])

        assert het.pulsarfiles == {"J0000+0000": self.fakeparfile[0]}
        assert het.pulsars == ["J0000+0000"]

        het = Heterodyne(pulsarfiles=self.fakeparfile)

        assert sorted(list(het.pulsarfiles.keys())) == [
            "J0000+0000",
            "J1111+1111",
            "J2222+2222",
        ]
        assert list(het.pulsarfiles.values()) == [
            self.fakeparfile[0],
            self.fakeparfile[1],
            self.fakeparfile[2],
        ]
        assert sorted(het.pulsars) == ["J0000+0000", "J1111+1111", "J2222+2222"]

        with pytest.raises(TypeError):
            Heterodyne(pulsarfiles=self.fakeparfile, pulsars=3.4)

        # check that ValueError is raised if pulsar does not exist in supplied par files
        with pytest.raises(ValueError):
            het = Heterodyne(pulsarfiles=self.fakeparfile, pulsars="J0328+5323")

        het = Heterodyne(pulsarfiles=[self.fakeparfile[0]], pulsars=["J0000+0000"])

        assert het.pulsarfiles == {"J0000+0000": self.fakeparfile[0]}
        assert het.pulsars == ["J0000+0000"]

        with pytest.raises(TypeError):
            het.pulsars = 453

        pulsarfiles = {}
        pulsarfiles["J0000+0000"] = "kgsdkgf"
        het = Heterodyne(pulsarfiles=pulsarfiles)
        captured = capsys.readouterr()
        assert len(het.pulsarfiles) == 0
        assert (
            captured.out
            == "Pulsar file 'kgsdkgf' could not be read. This pulsar will be ignored.\n"
        )

        pulsarfiles = {}
        pulsarfiles["J0000+0001"] = os.path.realpath(self.fakeparfile[0])
        het = Heterodyne(pulsarfiles=pulsarfiles)
        captured = capsys.readouterr()
        assert len(het.pulsarfiles) == 1
        assert (
            captured.out
            == "Inconsistent naming in pulsarfile dictionary. Using pulsar name 'J0000+0000' from parameter file\n"
        )

        pulsarfiles = {}
        pulsarfiles["J0000+0000"] = os.path.realpath(self.fakeparfile[0])
        het = Heterodyne(pulsarfiles=pulsarfiles)
        assert het.pulsarfiles == {"J0000+0000": os.path.realpath(self.fakeparfile[0])}
        assert het.pulsars == ["J0000+0000"]

    def test_download_pulsar(self, capsys):
        """
        Test downloading a pulsar from the ATNF pulsar catalogue.
        """

        psr = "J0534+2200"
        het = Heterodyne(pulsarfiles=psr)
        captured = capsys.readouterr()

        assert (
            captured.out
            == f"Ephemeris for '{psr}' has been obtained from the ATNF pulsar catalogue\n"
        )
        assert len(het.pulsarfiles) == 1
        assert het.pulsars == ["J0534+2200"]

    def test_crop(self):
        """
        Test setting of crop.
        """

        with pytest.raises(ValueError):
            Heterodyne(crop=-1)

        with pytest.raises(TypeError):
            Heterodyne(crop=0.5)

        crop = 50
        het = Heterodyne(crop=crop)

        assert het.crop == crop

    def test_heterodyne(self):
        """
        Test heterodyning on fake data.
        """

        segments = [
            (self.fakedatastarts[i], self.fakedatastarts[i] + self.fakedataduration[i])
            for i in range(len(self.fakedatastarts))
        ]

        het = Heterodyne(
            pulsarfiles=self.fakeparfile,
            pulsars=["J0000+0000"],
            segmentlist=segments,
            framecache=self.fakedatadir,
            channel=self.fakedatachannels[0],
        )

        with pytest.raises(TypeError):
            het.stride = "ksgdk"

        with pytest.raises(TypeError):
            het.stride = 4.5

        with pytest.raises(ValueError):
            het.stride = -1

        with pytest.raises(TypeError):
            het.filterknee = "lshdl"

        with pytest.raises(ValueError):
            het.filterknee = 0

        with pytest.raises(TypeError):
            het.freqfactor = "ldkme"

        with pytest.raises(ValueError):
            het.freqfactor = -2.3

        # test that output directory has defaulted to cwd
        assert os.path.split(list(het.outputfiles.values())[0])[0] == os.getcwd()

        # test setting an output directory
        outdir = os.path.join(self.fakedatadir, "heterodyne_output")
        het.outputfiles = outdir

        assert len(het.outputfiles) == 1
        assert list(het.outputfiles.keys()) == ["J0000+0000"]
        assert list(het.outputfiles.values()) == [os.path.join(outdir, het.label)]

        with pytest.raises(ValueError):
            # attempt to include glitch evolution without setting includessb to True
            het.heterodyne(includeglitch=True)

        # perform first stage heterodyne
        het = heterodyne(
            starttime=segments[0][0],
            endtime=segments[-1][-1],
            pulsarfiles=self.fakeparfile,
            segmentlist=segments,
            framecache=self.fakedatadir,
            channel=self.fakedatachannels[0],
            freqfactor=2,
            stride=86400 // 2,
            output=outdir,
            resamplerate=1,
        )

        labeldict = {
            "det": het.detector,
            "gpsstart": int(het.starttime),
            "gpsend": int(het.endtime),
            "freqfactor": int(het.freqfactor),
        }

        # expected length (after cropping)
        uncroppedsegs = [seg for seg in segments if (seg[1] - seg[0]) > het.crop]
        length = (
            het.resamplerate * np.diff(uncroppedsegs).sum()
            - 2 * len(uncroppedsegs) * het.crop
        )

        # expected start time (after cropping)
        t0 = segments[0][0] + het.crop + 0.5 / het.resamplerate

        # expected end time (after croppping)
        tend = segments[-1][-1] - het.crop - 0.5 / het.resamplerate

        # check output
        for psr in ["J0000+0000", "J1111+1111", "J2222+2222"]:
            assert os.path.isfile(het.outputfiles[psr].format(**labeldict, psr=psr))

            hetdata = HeterodynedData.read(
                het.outputfiles[psr].format(**labeldict, psr=psr)
            )

            assert len(hetdata) == length
            assert het.resamplerate == hetdata.dt.value
            assert t0 == hetdata.times.value[0]
            assert tend == hetdata.times.value[-1]
            assert het.detector == hetdata.detector

        # perform second stage of heterodyne
        with pytest.raises(TypeError):
            Heterodyne(heterodyneddata=0)

        fineoutdir = os.path.join(self.fakedatadir, "fine_heterodyne_output")

        # first heterodyne without SSB
        het2 = heterodyne(
            detector=self.fakedatadetectors[0],
            heterodyneddata=outdir,  # pass previous output directory
            pulsarfiles=self.fakeparfile,
            freqfactor=2,
            resamplerate=1 / 60,
            includessb=False,
            output=fineoutdir,
            label="heterodyne_{psr}_{det}_{freqfactor}.hdf5",
        )

        models = []
        lengthnew = int(
            np.sum(
                [
                    np.floor(((seg[1] - seg[0]) - 2 * het2.crop) * het2.resamplerate)
                    for seg in uncroppedsegs
                ]
            )
        )
        for i, psr in enumerate(["J0000+0000", "J1111+1111", "J2222+2222"]):
            # load data
            hetdata = HeterodynedData.read(
                het2.outputfiles[psr].format(**labeldict, psr=psr)
            )

            assert het2.resamplerate == 1 / hetdata.dt.value
            assert len(hetdata) == lengthnew

            # set expected model
            sim = HeterodynedCWSimulator(
                hetdata.par,
                hetdata.detector,
                times=hetdata.times.value,
                earth_ephem=hetdata.ephemearth,
                sun_ephem=hetdata.ephemsun,
            )

            # due to how the HeterodynedCWSimulator works we need to set
            # updateglphase = True for the glitching signal to generate a
            # signal without the glitch phase included!
            models.append(
                sim.model(
                    usephase=True,
                    freqfactor=hetdata.freq_factor,
                    updateglphase=(True if psr == "J2222+2222" else False),
                )
            )

            # without inclusion of SSB model should not match
            assert np.any(relative_difference(hetdata.data, models[i]) > 5e-3)

        # now heterodyne with SSB
        del het2
        het2 = heterodyne(
            detector=self.fakedatadetectors[0],
            heterodyneddata=outdir,  # pass previous output directory
            pulsarfiles=self.fakeparfile,
            freqfactor=2,
            resamplerate=1 / 60,
            includessb=True,
            output=fineoutdir,
            label="heterodyne_{psr}_{det}_{freqfactor}.hdf5",
            overwrite=True,
        )

        for i, psr in enumerate(["J0000+0000", "J1111+1111", "J2222+2222"]):
            # load data
            hetdata = HeterodynedData.read(
                het2.outputfiles[psr].format(**labeldict, psr=psr)
            )

            assert het2.resamplerate == 1 / hetdata.dt.value
            assert len(hetdata) == lengthnew

            # check output matches model to within 2%
            if psr == "J0000+0000":  # isolated pulsar
                assert np.all(relative_difference(hetdata.data, models[i]) < 0.02)
            else:
                # without inclusion of BSB/glitch phase model should not match
                assert np.any(relative_difference(hetdata.data, models[i]) > 0.02)

        # now heterodyne with SSB and BSB
        del het2
        het2 = heterodyne(
            detector=self.fakedatadetectors[0],
            heterodyneddata={
                psr: het.outputfiles[psr].format(**labeldict, psr=psr)
                for psr in ["J0000+0000", "J1111+1111", "J2222+2222"]
            },  # test using dictionary
            pulsarfiles=self.fakeparfile,
            freqfactor=2,
            resamplerate=1 / 60,
            includessb=True,
            includebsb=True,
            output=fineoutdir,
            label="heterodyne_{psr}_{det}_{freqfactor}.hdf5",
            overwrite=True,
        )

        for i, psr in enumerate(["J0000+0000", "J1111+1111", "J2222+2222"]):
            # load data
            hetdata = HeterodynedData.read(
                het2.outputfiles[psr].format(**labeldict, psr=psr)
            )

            assert het2.resamplerate == 1 / hetdata.dt.value
            assert len(hetdata) == lengthnew

            if psr in [
                "J0000+0000",
                "J1111+1111",
            ]:  # isolated and binary pulsar (non-glitching)
                assert np.all(relative_difference(hetdata.data, models[i]) < 0.02)
            else:
                # without inclusion glitch phase model should not match
                assert np.any(relative_difference(hetdata.data, models[i]) > 0.02)

        # now heterodyne with SSB, BSB and glitch phase
        del het2
        het2 = heterodyne(
            detector=self.fakedatadetectors[0],
            heterodyneddata={
                psr: het.outputfiles[psr].format(**labeldict, psr=psr)
                for psr in ["J0000+0000", "J1111+1111", "J2222+2222"]
            },  # test using dictionary
            pulsarfiles=self.fakeparfile,
            freqfactor=2,
            resamplerate=1 / 60,
            includessb=True,
            includebsb=True,
            includeglitch=True,
            output=fineoutdir,
            label="heterodyne_{psr}_{det}_{freqfactor}.hdf5",
            overwrite=True,
        )

        for i, psr in enumerate(["J0000+0000", "J1111+1111", "J2222+2222"]):
            # load data
            hetdata = HeterodynedData.read(
                het2.outputfiles[psr].format(**labeldict, psr=psr)
            )

            assert het2.resamplerate == 1 / hetdata.dt.value
            assert len(hetdata) == lengthnew
            assert np.all(relative_difference(hetdata.data, models[i]) < 0.02)

    def test_full_heterodyne(self):
        """
        Test heterodyning on fake data, performing the heterodyne in one step.
        """

        segments = [
            (self.fakedatastarts[i], self.fakedatastarts[i] + self.fakedataduration[i])
            for i in range(len(self.fakedatastarts))
        ]

        # perform heterodyne in one step
        fulloutdir = os.path.join(self.fakedatadir, "full_heterodyne_output")

        inputkwargs = dict(
            starttime=segments[0][0],
            endtime=segments[-1][-1],
            pulsarfiles=self.fakeparfile,
            segmentlist=segments,
            framecache=self.fakedatadir,
            channel=self.fakedatachannels[0],
            freqfactor=2,
            stride=86400 // 2,
            resamplerate=1 / 60,
            includessb=True,
            includebsb=True,
            includeglitch=True,
            output=fulloutdir,
            label="heterodyne_{psr}_{det}_{freqfactor}.hdf5",
        )

        het = heterodyne(**inputkwargs)

        labeldict = {
            "det": het.detector,
            "gpsstart": int(het.starttime),
            "gpsend": int(het.endtime),
            "freqfactor": int(het.freqfactor),
        }

        # compare against model
        for i, psr in enumerate(["J0000+0000", "J1111+1111", "J2222+2222"]):
            # load data
            hetdata = HeterodynedData.read(
                het.outputfiles[psr].format(**labeldict, psr=psr)
            )

            assert het.resamplerate == 1 / hetdata.dt.value

            # check heterodyne_arguments were stored and retrieved correctly
            assert isinstance(hetdata.heterodyne_arguments, dict)
            for param in inputkwargs:
                if param == "pulsarfiles":
                    assert inputkwargs[param][i] == hetdata.heterodyne_arguments[param]
                    assert hetdata.heterodyne_arguments["pulsars"] == psr
                else:
                    assert inputkwargs[param] == hetdata.heterodyne_arguments[param]

            # set expected model
            sim = HeterodynedCWSimulator(
                hetdata.par,
                hetdata.detector,
                times=hetdata.times.value,
                earth_ephem=hetdata.ephemearth,
                sun_ephem=hetdata.ephemsun,
            )

            # due to how the HeterodynedCWSimulator works we need to set
            # updateglphase = True for the glitching signal to generate a
            # signal without the glitch phase included!
            model = sim.model(
                usephase=True,
                freqfactor=hetdata.freq_factor,
                updateglphase=(True if psr == "J2222+2222" else False),
            )

            # increase tolerance for acceptance due to small outliers (still
            # equivalent at the ~2% level)
            assert np.all(relative_difference(hetdata.data, model) < 0.02)
