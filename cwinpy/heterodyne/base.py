"""
Classes for heterodyning strain data.
"""

import inspect
import os
import re
import signal
import tempfile
from pathlib import Path

import lal
import lalpulsar
import numpy as np
from gwosc.api import DEFAULT_URL as GWOSC_DEFAULT_HOST
from gwosc.timeline import get_segments
from gwpy.io.cache import is_cache, read_cache
from gwpy.segments import DataQualityFlag, SegmentList
from gwpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy
from scipy.interpolate import splev, splrep

from ..data import HeterodynedData
from ..utils import get_psr_name, initialise_ephemeris, is_par_file
from .fastheterodyne import fast_heterodyne


class Heterodyne(object):
    """
    Heterodyne gravitational-wave strain data based on a source's phase evolution.

    Parameters
    ----------
    starttime: int, float
        The integer start time of the data to be heterodyned in GPS seconds.
    endtime: int, float
        The integer end time of the data to be heterodyned in GPS seconds.
    stride: int
        The number of seconds to stride through the data, i.e., loop through
        the data reading in ``stride`` seconds each time. Defaults to 3600.
    detector: str
        A string given the name of the detector for which the data is to be
        heterodyned.
    frametype: str
        The ``frametype`` name of the data to be heterodyned. See, e.g., the
        GWPy documentation
        `here <https://gwpy.github.io/docs/stable/timeseries/datafind.html#available-datasets>`_
        for information on frame types. If this is not given the correct data
        set will be attempted to be found using the ``channel`` name.
    channel: str
        The "channel" within the gravitational-wave data file(s) (either a GW
        frame ``.gwf``, or HDF5 file) containing the strain data to be
        heterodyned. The channel name should contain the detector name prefix
        as the first two characters followed by a colon, e.g.,
        ``L1:GWOSC-4KHZ_R1_STRAIN``.
    host: str
        The server name for finding the gravitational-wave data files. Use
        ``datafind.ligo.org:443`` for open data available via CVMFS. To use
        open data available from the `GWOSC <https://www.gw-openscience.org>`_
        use ``https://www.gw-openscience.org``. See also
        :func:`gwpy.timeseries.TimeSeries.get`.
    outputframecache: str
        If a string is given it should give a file path to which a list of
        gravitational-wave data file paths, as found by the code, will be
        written. If not given then the file list will not be output.
    appendframecache: bool
        If writing out the frame cache to a file, set this to True to append
        to the file rather than overwriting. Default is False.
    framecache: str, list
        If you have a pregenerated cache of gravitational-wave file paths
        (either in a file or as a list of files) you can use them.
        Alternatively, you can supply a directory that will be searched
        recursively for frame files (first it will try to find ".gwf" file
        extension files then ".hdf5" files). This should be used in conjunction
        with the ``frametype`` argument.
    segmentlist: str, list
        A list of data segment start and end times (stored a tuple pairs) in
        the list. Or, an ascii text file containing segment start and end times
        in two columns.
    includeflags: str, list
        A string, or list of string giving data DQ flags to use to generate a
        segment list if not provided in ``segmentlist``. See, e.g., the GWPy
        documentation
        `here <https://gwpy.github.io/docs/stable/segments/index.html>`_ and
        the :func:`cwinpy.heterodyne.generate_segments` function. Use, e.g.,
        "H1_DATA" (or e.g., "H1_CBC_CAT2") to get segments from GWOSC for open
        data.
    excludeflags: str, list
        A string, or list of string giving data DQ flags to exclude when
        generating a segment list if not provided in ``segmentlist``. See
        the :func:`cwinpy.heterodyne.generate_segments` function.
    outputsegmentlist: str
        If generating a segment list it will be output to a file specified by
        this argument if given.
    appendsegmentlist: bool
        If generating a segment list it will be appended to the file given with
        ``outputsegmentlist`` if supplied.
    segmentserver: str
        If querying the segment database a URL of the database can be supplied.
    output: str, dict
        The base directory into which the heterodyned results will be output.
        To specify explicit directory paths for individual pulsars this can be
        a dictionary of directory paths keyed to the pulsar name (in which
        case the ``label`` argument will be used to set the file name), or full
        file paths, which will be used in place of the ``label`` argument. If
        not given this defaults to the current working directory.
    label: str
        The output format for the heterodyned data files. These can be format
        strings containing the keywords ``psr`` for the pulsar name, ``det``
        for the detector, ``freqfactor`` for the rotation frequency scale
        factor used, ``gpsstart`` for the GPS start time, and ``gpsend`` for
        the GPS end time. The extension should be given as ".hdf", ".h5", or
        ".hdf5". E.g., the default is
        ``"heterodyne_{psr}_{det}_{freqfactor}_{gpsstart}-{gpsend}.hdf"``.
    pulsarfiles: str, list, dict
        This specifies the pulsars for which to heterodyne the data. It can be
        either i) a string giving the path to an individual pulsar
        TEMPO(2)-style parameter file, ii) a string giving the path to a
        directory containing multiple TEMPO(2)-style parameter files (the path
        will be recursively searched for any file with the extension ".par"),
        iii) a list of paths to individual pulsar parameter files, iv) a
        dictionary containing paths to individual pulsars parameter files keyed
        to their names. If ``pulsarfiles`` contains pulsar names rather than
        files it will attempt to extract an ephemeris for that pulsar from the
        ATNF pulsar catalogue. If such an ephemeris is available then that will
        be used (notification will be given when this is the case).
    pulsars: str, list
        You can analyse only particular pulsars from those specified by
        parameter files found through the ``pulsarfiles`` argument by passing a
        string, or list of strings, with particular pulsars names to use.
    filterknee: float
        The knee frequency (Hz) of the low-pass filter applied after
        heterodyning the data. This should only be given when heterodying raw
        strain data and not if re-heterodyning processed data. Default is 0.5
        Hz.
    heterodyneddata: str, dict
        A string or dictionary of strings/lists containing the full file path,
        or directory path, pointing the the location of pre-heterodyned data.
        For a single pulsar a file path can be given. For multiple pulsars a
        directory containing heterodyned files (in HDF5 or txt format) can be
        given provided that within it the file names contain the pulsar names
        as supplied in the file input with ``pulsarfiles``. Alternatively, a
        dictionary can be supplied, keyed on the pulsar name, containing a
        single file path, a directory path as above, or a list of file paths.
        If supplying a directory, it can contain multiple heterodyned files for
        a each pulsar and all will be used.
    resamplerate: float
        The rate in Hz at which to resample the data (via averaging) after
        application of the heterodyne (and filter if applied).
    freqfactor: float
        The factor applied to the pulsars rotational parameters when defining
        the gravitational-wave phase evolution. For example, the default value
        of 2 multiplies the phase evolution by 2 under the assumption of a
        signal emitted from the l=m=2 quadrupole mode of a rigidly rotating
        triaxial neutron star.
    crop: int
        The number of seconds to crop from the start and end of data segments
        to remove filter impulse effects and issues prior to lock-loss.
        Default is 60 seconds.
    includessb: bool
        Set this to True to include removing the modulation of the signal due
        to Solar System motion and relativistic effects (e.g., Roemer,
        Einstein, and Shapiro delay) during the heterodyne. Default is False.
    includebsb: bool
        Set this to True to include removing the modulation of the signal due
        to binary system motion and relativistic effects during the heterodyne.
        To use this ``includessb`` must also be True. Default is False.
    includeglitch: bool
        Set this to True to include removing the effects on the phase evolution
        of any modelled pulsar glitches during the heterodyne. Default is False.
    includefitwaves: bool
        Set this to True to include removing the phase evolution of a series of
        sinusoids designed to model low-frequency timing noise in the pulsar
        signal during the heterodyne. Default is False.
    interpolationstep: int
        If performing the full heterodyne in one step it is quicker to
        interpolate the solar system barycentre time delay and binary
        barycentre time delay. This sets the step size, in seconds, between
        the points in these calculations that will be the interpolation nodes.
        The default is 60 seconds.
    earthephemeris: dict
        A dictionary, keyed to ephemeris names, e.g., "DE405", pointing to the
        location of a file containing that ephemeris for the Earth. If a pulsar
        requires a specific ephemeris that is not provided in this dictionary,
        then the code will automatically attempt to find or download the
        required file if available.
    sunephemeris: dict
        A dictionary, keyed to ephemeris names, e.g., "DE405", pointing to the
        location of a file containing that ephemeris for the Sun. If a pulsar
        requires a specific ephemeris that is not provided in this dictionary,
        then the code will automatically attempt to find or download the
        required file if available.
    timeephemeris: dict
        A dictionary, keyed to time system name, which can be either "TCB" or
        "TDB", pointing to the location of a file containing that ephemeris for
        that time system. If a pulsar requires a specific ephemeris that is not
        provided in this dictionary, then the code will automatically attempt
        to find or download the required file if available.
    resume: bool
        Set to True to resume heterodyning in case not all pulsars completed.
        This checks whether output files (as set using ``output`` and
        ``label`` arguments) already exist and does not repeat the analysis
        if that is the case. If wanting to overwrite existing files make sure
        this is False. Defaults to False.
    cwinpy_heterodyne_dag_config_file: str
        If Heterodyne is being called by a job in a HTCondor DAG, then this can
        provide the path to the configuration file that was used to setup the
        DAG. Defaults to None.
    """

    # allowed file extension
    extensions = [".hdf", ".h5", ".hdf5", ".txt"]

    def __init__(
        self,
        starttime=None,
        endtime=None,
        stride=3600,
        detector=None,
        frametype=None,
        channel=None,
        host=None,
        outputframecache=None,
        appendframecache=False,
        framecache=None,
        segmentlist=None,
        includeflags=None,
        excludeflags=None,
        outputsegmentlist=None,
        appendsegmentlist=False,
        segmentserver=None,
        heterodyneddata=None,
        pulsarfiles=None,
        pulsars=None,
        output=None,
        label=None,
        filterknee=0.5,
        resamplerate=1.0,
        freqfactor=2,
        crop=60,
        includessb=False,
        includebsb=False,
        includeglitch=False,
        includefitwaves=False,
        interpolationstep=60,
        earthephemeris=None,
        sunephemeris=None,
        timeephemeris=None,
        resume=False,
        cwinpy_heterodyne_dag_config_file=None,
    ):
        # set analysis times
        self.starttime = starttime
        self.endtime = endtime
        self.stride = stride

        # set detector
        self.detector = detector

        # set frame type and channel
        self.channel = channel
        if framecache is None:
            self.frametype = frametype
            self.host = host
            self.outputframecache = outputframecache
            self.appendframecache = appendframecache
        else:
            self.framecache = framecache

        # set segment list information
        self.outputsegmentlist = outputsegmentlist
        self.appendsegmentlist = appendsegmentlist
        self.segments = segmentlist

        if segmentlist is None:
            self.includeflags = includeflags
            self.excludeflags = excludeflags
            self.segmentserver = segmentserver

        # set the pulsars
        self.pulsarfiles = pulsarfiles
        if pulsars is not None:
            self.pulsars = pulsars

        # set the output directory
        self.label = label
        self.outputfiles = output
        self.resume = resume

        # set previously heterodyned data
        self.heterodyneddata = heterodyneddata

        # set heterodyne parameters
        self.resamplerate = resamplerate
        self.filterknee = filterknee
        self.freqfactor = freqfactor
        self.crop = crop
        self.includessb = includessb
        self.includebsb = includebsb
        self.includeglitch = includeglitch
        self.includefitwaves = includefitwaves
        self.interpolationstep = interpolationstep

        # set ephemeris information
        self.set_ephemeris(earthephemeris, sunephemeris, timeephemeris)

        # set the name of any DAG configuration file
        self.cwinpy_heterodyne_dag_config_file = cwinpy_heterodyne_dag_config_file

        # set signal in case of termination of job
        signal.signal(signal.SIGTERM, self._write_current_pulsars_and_exit)
        signal.signal(signal.SIGINT, self._write_current_pulsars_and_exit)
        signal.signal(signal.SIGALRM, self._write_current_pulsars_and_exit)
        self.exit_code = 130  # exit code expected by HTCondor from eviction

    @property
    def starttime(self):
        """
        The start time of the heterodyned data in GPS seconds.
        """

        return self._starttime

    @starttime.setter
    def starttime(self, gpsstart):
        if gpsstart is not None:
            if not isinstance(gpsstart, (int, float)):
                raise TypeError("GPS start time must be a number")

            if self.endtime is not None:
                if gpsstart >= self.endtime:
                    raise ValueError("GPS start time must be before end time")

            self._starttime = int(gpsstart)
        else:
            self._starttime = None

    @property
    def endtime(self):
        """
        The end time of the heterodyned data in GPS seconds.
        """

        if hasattr(self, "_endtime"):
            return self._endtime
        else:
            return None

    @endtime.setter
    def endtime(self, gpsend):
        if gpsend is not None:
            if not isinstance(gpsend, (int, float)):
                raise TypeError("GPS end time must be a number")

            if self.starttime is not None:
                if gpsend <= self.starttime:
                    raise ValueError("GPS end time must be after start time")

            self._endtime = int(gpsend)
        else:
            self._endtime = None

    @property
    def stride(self):
        """
        The stride length to loop through the data.
        """

        return self._stride

    @stride.setter
    def stride(self, stride):
        if not isinstance(stride, (int, float)):
            raise TypeError("Stride must be an integer")
        else:
            if isinstance(stride, float):
                if not stride.is_integer():
                    raise TypeError("Stride must be an integer")

            if stride <= 0:
                raise ValueError("Stride must be a positive number")

            self._stride = int(stride)

    @property
    def detector(self):
        """
        The gravitational-wave detector name prefix.
        """

        return self._detector

    @detector.setter
    def detector(self, detector):
        if isinstance(detector, str):
            try:
                self._laldetector = lalpulsar.GetSiteInfo(detector)
            except RuntimeError:
                raise ValueError(
                    "Detector '{}' is not a known/valid detector name".format(detector)
                )
        elif type(detector) is lal.Detector:
            self._laldetector = detector
        elif detector is None:
            self._laldetector = None
        else:
            raise TypeError("Detector is unknown type")

        self._detector = (
            self._laldetector.frDetector.prefix
            if self._laldetector is not None
            else None
        )

    @property
    def laldetector(self):
        """
        A ``lal.Detector`` structure containing the detector information.
        """

        if hasattr(self, "_laldetector"):
            return self._laldetector
        else:
            return None

    @property
    def frametype(self):
        """
        The frame type of data to get.
        """

        if hasattr(self, "_frametype"):
            return self._frametype
        else:
            return None

    @frametype.setter
    def frametype(self, frametype):
        if frametype is None or isinstance(frametype, str):
            self._frametype = frametype
        else:
            raise TypeError("Frame type must be a string")

    @property
    def channel(self):
        """
        The data channel within a gravitational-wave data frame to use.
        """

        return self._channel

    @channel.setter
    def channel(self, channel):
        if channel is None or isinstance(channel, str):
            self._channel = channel

            if self.channel is not None:
                # get detector from channel name
                if ":" not in self.channel:
                    raise ValueError("Channel name must contain a detector prefix")

                # check detectors are consistent
                det = self.channel.split(":")[0]
                if self.detector is not None:
                    if det != self.detector:
                        raise ValueError(
                            "Given detector '{}' and detector from channel name '{}' do not match".format(
                                self.detector, det
                            )
                        )
                else:
                    # set detector from channel name
                    self.detector = det
        else:
            raise TypeError("Channel must be a string")

    @property
    def framecache(self):
        """
        A file name, or list of files, containing gravitational-wave strain
        data.
        """

        if hasattr(self, "_framecache"):
            return self._framecache
        else:
            return None

    @framecache.setter
    def framecache(self, framecache):
        if framecache is None:
            self._framecache = None
        elif isinstance(framecache, (str, list)):
            if isinstance(framecache, str):
                if not os.path.isfile(framecache) and not os.path.isdir(framecache):
                    raise ValueError(
                        "Frame cache file/directory '{}' does not exist".format(
                            framecache
                        )
                    )
            if isinstance(framecache, list):
                # check files exist
                for fr in framecache:
                    if not isinstance(fr, str):
                        raise TypeError("Frame cache list must contain strings")
                    if not os.path.isfile(fr):
                        raise ValueError(
                            "Frame cache file '{}' does not exist".format(fr)
                        )
            self._framecache = framecache
        else:
            raise TypeError("Frame cache must be a string or list")

    @property
    def outputframecache(self):
        """
        The path to a file into which to output a file containing a list of
        frame files.
        """

        if hasattr(self, "_outputframecache"):
            return self._outputframecache
        else:
            return None

    @outputframecache.setter
    def outputframecache(self, outputframecache):
        if isinstance(outputframecache, str) or outputframecache is None:
            self._outputframecache = outputframecache
        else:
            raise TypeError("outputframecache must be a string")

    @property
    def appendframecache(self):
        """
        A boolean stating whether to append a list of frame files to an
        existing file specified by ``outputframecache``.
        """

        if hasattr(self, "_appendframecache"):
            return self._appendframecache
        else:
            return False

    @appendframecache.setter
    def appendframecache(self, appendframecache):
        if isinstance(appendframecache, (bool, int)) or appendframecache is None:
            self._appendframecache = bool(appendframecache)
        else:
            raise TypeError("appendframecache must be a boolean")

    @property
    def host(self):
        """
        The data server hostname URL.
        """

        if hasattr(self, "_host"):
            return self._host
        else:
            return None

    @host.setter
    def host(self, host):
        if not isinstance(host, str) and host is not None:
            raise TypeError("Hostname server must be string")
        else:
            self._host = host

            if self.host is not None:
                # check for valid and exitsing URL
                import requests

                if "http" != self.host[0:4]:
                    for schema in ["http", "https"]:
                        try:
                            url = requests.get("{}://{}".format(schema, self.host))
                        except Exception:
                            url = None
                else:
                    try:
                        url = requests.get(self.host)
                    except Exception as e:
                        raise RuntimeError("Host URL was not valid: {}".format(e))

                if url is None:
                    raise RuntimeError("Host URL was not valid")
                else:
                    if url.status_code != 200:
                        raise RuntimeError("Host URL was not valid")

    def get_frame_data(
        self,
        starttime=None,
        endtime=None,
        channel=None,
        framecache=None,
        frametype=None,
        host=None,
        outputframecache=None,
        appendframecache=None,
        **kwargs,
    ):
        """
        Get gravitational-wave frame/hdf5 data between a given start and end
        time in GPS seconds.

        Parameters
        ----------
        starttime: int, float
            The start time of the data to extract in GPS seconds. If not set,
            the object's ``starttime`` attribute will be used.
        endtime: int, float
            The end time of the data to extract in GPS seconds. If not set,
            the object's ``endtime`` attribute will be used.
        channel: str
            The data channel with the gravitational-wave frame data to use. If
            not set, the object's ``channel`` attribute will be used.
        framecache: str, list
            The path to a file containing a list of frame files (or a single
            frame file), a list of frame files, or a directory containing frame
            files. If this is a directory the
            :func:`~cwinpy.heterodyne.local_frame_cache` function will be
            used to generate the frame list. If not set, the object's
            ``framecache`` attribute will be used.
        frametype: str
            The "type" of frame data to get. If not set, the object's
            ``frametype`` will be used. If no ``frametype`` is found the code
            will attempt to work it out from the supplied ``channel``.
        host: str
            The host server for the frame files.

        Returns
        -------
        data: TimeSeries
            A :class:`gwpy.timeseries.TimeSeries` containing the data.
        """

        starttime = starttime if isinstance(starttime, int) else self.starttime
        endtime = endtime if isinstance(endtime, int) else self.endtime
        channel = channel if isinstance(channel, str) else self.channel
        framecache = (
            framecache if isinstance(framecache, (str, list)) else self.framecache
        )
        frametype = frametype if isinstance(frametype, str) else self.frametype
        host = host if isinstance(host, str) else self.host
        outputframecache = (
            outputframecache
            if isinstance(outputframecache, str)
            else self.outputframecache
        )
        appendframecache = (
            appendframecache
            if isinstance(appendframecache, bool)
            else self.appendframecache
        )

        if starttime is None:
            raise ValueError("A start time is not set")

        if endtime is None:
            raise ValueError("An end time is not set")

        if channel is None and host != GWOSC_DEFAULT_HOST:
            raise ValueError("No channel name has been set")

        if framecache is not None:
            # read data from cache
            cache = framecache
            if isinstance(framecache, str):
                # check if cache is a directory
                if os.path.isdir(framecache):
                    # get list of frame files
                    extensions = (
                        ["gwf", "hdf5"]
                        if "extension" not in kwargs
                        else [kwargs["extension"]]
                    )

                    # try different extensions in turn
                    for extension in extensions:
                        cache = local_frame_cache(
                            framecache,
                            starttime=starttime,
                            endtime=endtime,
                            frametype=frametype,
                            recursive=kwargs.get("recursive", True),
                            site=kwargs.get("site", self.detector),
                            extension=extension,
                            write=outputframecache,
                            append=appendframecache,
                        )
                        if len(cache) > 0:
                            break
                elif os.path.isfile(framecache) and os.path.splitext(framecache)[1] in [
                    ".gwf",
                    ".hdf5",
                ]:
                    # check if cache is a single frame file
                    cache = [framecache]
                else:
                    # read in frame files from cache file
                    if is_cache(framecache):
                        cache = read_cache(framecache)
                    else:
                        raise IOError(
                            "Frame cache file '{}' could not be read".format(framecache)
                        )
            try:
                # TimeSeries.read can't read from multiple files, so read individually
                data = TimeSeriesDict()
                startread = starttime
                for frfile in cache:
                    _, _, frt0, frdur = frame_information(frfile)

                    if frt0 >= endtime:
                        # found frame files, so break out of loop
                        break
                    if starttime < frt0 and endtime >= frt0:
                        startread = frt0

                        if endtime > frt0 + frdur:
                            endread = frt0 + frdur
                        else:
                            endread = endtime
                    elif (frt0 <= starttime < frt0 + frdur) and endtime >= frt0:
                        startread = starttime

                        if endtime > frt0 + frdur:
                            endread = frt0 + frdur
                        else:
                            endread = endtime
                    else:
                        # move on to next file
                        continue

                    thisdata = TimeSeriesDict.read(
                        frfile,
                        [channel],
                        start=startread,
                        end=endread,
                        pad=kwargs.get("pad", 0.0),
                    )

                    # zero pad any gaps in the data
                    data.append(thisdata, pad=0.0, gap="pad")

                # extract channel from dictionary
                data = data[channel]
            except Exception as e:
                raise IOError("Could not read in frame data from cache: {}".format(e))
        else:
            # download data
            if host == GWOSC_DEFAULT_HOST:
                try:
                    # get GWOSC data
                    data = TimeSeries.fetch_open_data(
                        kwargs.get("site", self.detector),
                        starttime,
                        endtime,
                        host=host,
                        cache=kwargs.get("cache", False),
                    )
                except Exception as e:
                    # return None if no data could be obtained
                    print("Warning: {}".format(e.args[0]))
                    data = None
            else:  # pragma: no cover
                try:
                    if self.outputframecache:
                        # output cache list
                        cache = remote_frame_cache(
                            starttime,
                            endtime,
                            channel,
                            frametype=frametype,
                            write=outputframecache,
                            append=appendframecache,
                            host=host,
                        )
                        data = TimeSeries.read(
                            [item for key in cache for item in cache[key]],
                            start=starttime,
                            end=endtime,
                            pad=kwargs.get("pad", 0.0),  # fill gaps with zeros
                        )
                    else:
                        data = TimeSeries.get(
                            channel,
                            starttime,
                            endtime,
                            host=host,
                            frametype=frametype,
                            pad=kwargs.get("pad", 0.0),  # fill gaps with zeros
                        )
                except Exception as e:
                    raise IOError("Could not download frame data: {}".format(e))

        return data

    def get_segment_list(self, **kwargs):
        """
        Generate and return a list of time segments of data to analyse. See
        :func:`~cwinpy.heterodyne.generate_segments` for the arguments. Unless
        reading from a pre-generated segments file this requires access to the
        GW segment database, which is reserved for members of the
        LIGO-Virgo-KAGRA collaborations.

        For parameters that are not provided the values set in the object's
        attributes will be used if available.
        """

        kwargs.setdefault("starttime", self.starttime)
        kwargs.setdefault("endtime", self.endtime)

        if "segmentfile" not in kwargs:
            kwargs.setdefault("includeflags", self.includeflags)
            kwargs.setdefault("excludeflags", self.excludeflags)
            kwargs.setdefault("server", self.segmentserver)
            kwargs.setdefault("writesegments", self.outputsegmentlist)
            kwargs.setdefault("appendsegments", self.appendsegmentlist)

            # check whether include flags looks like it wants GWOSC data
            if len(self.includeflags) == 1:
                # GWOSC segments look like DET_DATA, DET_CW* or DET_*_CAT*
                if (
                    "{}_DATA".format(self.detector) == self.includeflags[0]
                    or "{}_CW".format(self.detector) in self.includeflags[0]
                    or "CBC_CAT" in self.includeflags[0]
                    or "BURST_CAT" in self.includeflags[0]
                ):
                    kwargs.setdefault("usegwosc", True)

        segments = generate_segments(**kwargs)
        self._segments = segments

        return segments

    @property
    def segments(self):
        """
        The list of data segments to analyse, within start and end times.
        """

        if self.starttime is None:
            st = -np.inf
        else:
            st = self.starttime

        if self.endtime is None:
            et = np.inf
        else:
            et = self.endtime

        if len(np.shape(self._segments)) > 1:
            if np.shape(self._segments)[1] == 2:
                if self._segments[0][0] >= st and self._segments[-1][-1] <= et:
                    return self._segments

        segments = []

        for segment in self._segments if self._segments is not None else [[st, et]]:
            if segment[1] < st or segment[0] > et:
                continue

            start = segment[0] if segment[0] > st else st
            end = segment[1] if segment[1] < et else et

            if start >= end:
                continue

            segments.append((start, end))

        return segments

    @segments.setter
    def segments(self, segments):
        if segments is None:
            self._segments = None
        elif isinstance(segments, list):
            self._segments = segments
        elif isinstance(segments, str):
            # try reading segments
            self.get_segment_list(segmentfile=segments)
        else:
            raise TypeError("segment must be a list or a string")

    @property
    def outputsegmentlist(self):
        """
        The path to a file into which to output a file containing a list of
        data segments.
        """

        return self._outputsegmentlist

    @outputsegmentlist.setter
    def outputsegmentlist(self, outputsegmentlist):
        if isinstance(outputsegmentlist, str) or outputsegmentlist is None:
            self._outputsegmentlist = outputsegmentlist
        else:
            raise TypeError("outputsegmentlist must be a string")

    @property
    def appendsegmentlist(self):
        """
        A boolean stating whether to append a list of frame files to an
        existing file specified by ``outputsegmentlist``.
        """

        return self._appendsegmentlist

    @appendsegmentlist.setter
    def appendsegmentlist(self, appendsegmentlist):
        if isinstance(appendsegmentlist, (bool, int)) or appendsegmentlist is None:
            self._appendsegmentlist = bool(appendsegmentlist)
        else:
            raise TypeError("appendsegmentlist must be a boolean")

    @property
    def includeflags(self):
        """
        The data quality segment flags to use to generate times of data to
        analyse.
        """

        if hasattr(self, "_includeflags"):
            return self._includeflags
        else:
            return None

    @includeflags.setter
    def includeflags(self, includeflags):
        if isinstance(includeflags, str):
            self._includeflags = includeflags.split(",")
        elif isinstance(includeflags, list):
            self._includeflags = [
                flag for incfl in includeflags for flag in incfl.split(",")
            ]
        elif includeflags is None:
            self._includeflags = None
        else:
            raise TypeError("includeflags must be a list or a string")

    @property
    def excludeflags(self):
        """
        The data quality segment flags to use to exclude times of data to
        analyse.
        """

        if hasattr(self, "_excludeflags"):
            return self._excludeflags
        else:
            return None

    @excludeflags.setter
    def excludeflags(self, excludeflags):
        if isinstance(excludeflags, str):
            self._excludeflags = excludeflags.split(",")
        elif isinstance(excludeflags, list):
            self._excludeflags = [
                flag for incfl in excludeflags for flag in incfl.split(",")
            ]
        elif excludeflags is None:
            self._excludeflags = None
        else:
            raise TypeError("excludeflags must be a list or a string")

    @property
    def segmentserver(self):
        """
        The server URL for a data quality segment database.
        """

        if hasattr(self, "_segmentserver"):
            return self._segmentserver
        else:
            return None

    @segmentserver.setter
    def segmentserver(self, server):
        if isinstance(server, str) or server is None:
            self._segmentserver = server
        else:
            raise TypeError("segmentserver must be a string")

    @property
    def pulsarfiles(self):
        """
        A dictionary of parameter files containing pulsars timing ephemeris
        keyed to the pulsar name.
        """

        return self._pulsars

    @pulsarfiles.setter
    def pulsarfiles(self, pfiles):
        self._pulsars = {}

        if isinstance(pfiles, (str, list, dict)):
            pfilelist = pfiles
            pnames = None
            if isinstance(pfiles, str):
                if os.path.isdir(pfiles):
                    pfilelist = [
                        str(pf.resolve()) for pf in Path(pfiles).rglob("*.par")
                    ]
                else:
                    pfilelist = [pfiles]
            elif isinstance(pfiles, dict):
                pfilelist = list(pfiles.values())
                pnames = list(pfiles.keys())
            else:
                pfilelist = []
                for pf in pfiles:
                    if os.path.isfile(pf):
                        pfilelist.append(pf)
                    elif os.path.isdir(pf):
                        pfilelist += [
                            str(pfp.resolve()) for pfp in Path(pf).rglob("*.par")
                        ]

            for i, pf in enumerate(pfilelist):
                if is_par_file(pf):
                    # read parameters
                    par = PulsarParametersPy(pf)

                    # get pulsar name
                    pname = get_psr_name(par)
                    self._pulsars[pname] = pf

                    # check for naming consistency
                    if pnames is not None:
                        if pnames[i] != pname:
                            print(
                                "Inconsistent naming in pulsarfile dictionary. "
                                "Using pulsar name '{}' from parameter file".format(
                                    pname
                                )
                            )
                else:
                    # try checking if a pulsar name has been given and if that
                    # is present in the ATNF catalogue. In that case use the
                    # ATNF ephemeris.
                    if not hasattr(self, "_atnf_query"):
                        from psrqpy import QueryATNF

                        self._atnf_query = QueryATNF()

                    par = self._atnf_query.get_ephemeris(psr=pf)

                    if par is None:
                        print(
                            f"Pulsar file '{pf}' could not be read. This pulsar will be ignored."
                        )
                        continue
                    else:
                        print(
                            f"Ephemeris for '{pf}' has been obtained from the ATNF pulsar catalogue"
                        )
                        # create temporary par file containing ATNF ephemeris
                        tmppar = tempfile.mkstemp(suffix=".par", prefix=pf)
                        with open(tmppar[1], "w") as fp:
                            fp.write(par)

                        # get pulsar name
                        readpar = PulsarParametersPy(tmppar[1])
                        pname = get_psr_name(readpar)
                        self._pulsars[pname] = tmppar[1]
        elif pfiles is not None:
            raise TypeError("pulsarfiles must be a string, list or dict")

    @property
    def pulsars(self):
        """
        A list of pulsar names to be analysed.
        """

        return list(self._pulsars.keys())

    @pulsars.setter
    def pulsars(self, pulsars):
        if isinstance(pulsars, (str, list, PulsarParametersPy)):
            pulsarlist = (
                [pulsars] if isinstance(pulsars, (str, PulsarParametersPy)) else pulsars
            )

            # check if any supplied pulsars have associated parameter files
            if len([pulsar for pulsar in pulsarlist if pulsar in self.pulsars]) == 0:
                raise ValueError("No parameters are provided for the supplied pulsars")

            for pulsar in self.pulsars:
                if pulsar not in list(pulsarlist):
                    del self._pulsars[pulsar]
                else:
                    pulsarlist.remove(pulsar)

            if len(pulsarlist) != 0:
                print(
                    "Pulsars '{}' not included as no parameter files have been given for them".format(
                        pulsarlist
                    )
                )
        else:
            raise TypeError("pulsars must be a list or string")

    @property
    def resume(self):
        """
        Resume an analysis that has failed part way through.
        """

        return self._resume

    @resume.setter
    def resume(self, resume):
        if isinstance(resume, (bool, int)):
            self._resume = bool(resume)
        else:
            raise TypeError("resume must be a boolean")

    @property
    def label(self):
        """
        File name label formatter.
        """

        if self._label is None:
            # return default formatter
            return "heterodyne_{psr}_{det}_{freqfactor}_{gpsstart}-{gpsend}.hdf5"
        else:
            return self._label

    @label.setter
    def label(self, label):
        if isinstance(label, str) or label is None:
            self._label = label
        else:
            raise TypeError("label must be a string")

    @property
    def outputfiles(self):
        """
        A dictionary of output file names for each pulsar.
        """

        return self._outputfiles

    @outputfiles.setter
    def outputfiles(self, outputfiles):
        self._outputfiles = {}

        if outputfiles is None:
            # set to current working directory
            outputfiles = os.getcwd()

        if isinstance(outputfiles, str):
            if not os.path.isdir(outputfiles):
                try:
                    os.makedirs(outputfiles, exist_ok=True)
                except Exception as e:
                    raise IOError(
                        "Couldn't create output directory '{}': {}".format(
                            outputfiles, e
                        )
                    )

            for pulsar in self.pulsars:
                self._outputfiles[pulsar] = os.path.join(outputfiles, self.label)
        elif isinstance(outputfiles, dict):
            self._outputfiles = outputfiles

            for pulsar in outputfiles:
                if pulsar not in self.pulsars:
                    raise KeyError(
                        "Trying to set output file for pulsar that does not exist"
                    )

                # check if a file (by testing for a hdf5-type or txt extension)
                if os.path.splitext(outputfiles[pulsar])[1] in self.extensions:
                    # try making directory
                    if not os.path.isdir(os.path.split(outputfiles[pulsar])[0]):
                        try:
                            os.makedirs(
                                os.path.split(outputfiles[pulsar])[0], exist_ok=True
                            )
                        except Exception as e:
                            raise IOError(
                                "Couldn't create output directory '{}': {}".format(
                                    outputfiles[pulsar], e
                                )
                            )
                else:
                    # a directory has been given
                    if not os.path.isdir(outputfiles[pulsar]):
                        try:
                            os.makedirs(outputfiles[pulsar], exist_ok=True)
                        except Exception as e:
                            raise IOError(
                                "Couldn't create output directory '{}':{}".format(
                                    outputfiles[pulsar], e
                                )
                            )

                    self._outputfiles[pulsar] = os.path.join(
                        outputfiles[pulsar], self.label
                    )

    @property
    def resamplerate(self):
        """
        The rate at which to resample (by averaging) the data after heteroydne.
        """

        return self._resamplerate

    @resamplerate.setter
    def resamplerate(self, rs):
        if isinstance(rs, (float, int)):
            if rs > 0:
                self._resamplerate = float(rs)
            else:
                raise ValueError("Resample rate must be positive")
        else:
            raise TypeError("resample rate must be a positive number")

    @property
    def filterknee(self):
        """
        The knee frequency for the low-pass filtering of the heterodyned data
        in Hz.
        """

        return self._filterknee

    @filterknee.setter
    def filterknee(self, fk):
        if isinstance(fk, (float, int)):
            if fk > 0:
                self._filterknee = float(fk)
            else:
                raise ValueError("Filter knee frequency rate must be positive")
        elif fk is None:
            self._filterknee = None
        else:
            raise TypeError("Filter knee must be a positive number")

    @property
    def freqfactor(self):
        """
        The mutiplicative factor applied to the pulsar rotational phase
        evolution to give the corresponding gravitational-wave phase evolution.
        """

        return self._freqfactor

    @freqfactor.setter
    def freqfactor(self, ff):
        if isinstance(ff, (float, int)):
            if ff > 0:
                self._freqfactor = float(ff)
            else:
                raise ValueError("Frequency factor must be positive")
        else:
            raise TypeError("freqfactor must be a positive number")

    @property
    def crop(self):
        """
        The number of seconds to crop from the data at the start and end of
        data segments.
        """

        return self._crop

    @crop.setter
    def crop(self, crop):
        if isinstance(crop, int):
            if crop >= 0:
                self._crop = crop
            else:
                raise ValueError("crop must be a positive integer or zero")
        else:
            raise TypeError("crop must be an integer")

    def heterodyne(self, **kwargs):
        """
        Heterodyne the data. This can be used to heterodyne raw
        gravitational-wave strain data, or re-heterodyne data that has been
        previously heterodyned. The heterodyning will be performed on all
        supplied pulsars.

        It performs the following main steps:

        * obtain a set of data segment to analyse (if not already supplied);
        * read in the data;
        * perform the heterodyne;
        * (if using raw data) low-pass filter the heterodyned data;
        * downsample the data (via averaging)
        * convert to a :class:`~cwinpy.data.HeterodynedData` object and write
          out to a file.

        It is recommended to use this function with no arguments and instead
        rely on the arguments supplied when initialising the
        :class:`~cwinpy.heterodyne.Heterodyne` object. However, any of the
        class keywords can be used and will overwrite those from
        initialisation. The function does not return any output as results are
        instead written to disk.
        """

        # time information
        self.starttime = kwargs.get("starttime", self.starttime)
        self.endtime = kwargs.get("endtime", self.endtime)
        self.stride = kwargs.get("stride", self.stride)

        # heterodyne information
        self.filterknee = kwargs.get("filterknee", self.filterknee)
        self.resamplerate = kwargs.get("resamplerate", self.resamplerate)
        self.crop = kwargs.get("crop", self.crop)
        self.freqfactor = kwargs.get("freqfactor", self.freqfactor)
        self.includessb = kwargs.get("includessb", self.includessb)
        self.includebsb = kwargs.get("includebsb", self.includebsb)
        self.includeglitch = kwargs.get("includeglitch", self.includeglitch)
        self.includefitwaves = kwargs.get("includefitwaves", self.includefitwaves)
        self.interpolationstep = kwargs.get("interpolationstep", self.interpolationstep)

        # solar system ephemeris information

        # update heterodyned data
        if kwargs.get("heterodyneddata", None) is not None:
            self.heterodyneddata = kwargs.get("heterodyneddata")

        # update pulsars if given
        if kwargs.get("pulsarfiles", None) is not None:
            self.pulsarfiles = kwargs.get("pulsarfiles")
        if kwargs.get("pulsars", None) is not None:
            self.pulsars = kwargs.get("pulsars")

        # update outputfiles if given
        if kwargs.get("label", None) is not None:
            self.label = kwargs.get("label")
        if kwargs.get("output", None) is not None:
            self.outputfiles = kwargs.get("output")
        elif len(self.outputfiles) == 0:
            raise ValueError("No output destination has been set!")

        # get segment list
        if (
            self._segments is None
            and not self.heterodyneddata
            and self.includeflags is not None
        ):
            self.get_segment_list(**kwargs)

        ttype = {
            "TDB": lalpulsar.TIMECORRECTION_TDB,
            "TCB": lalpulsar.TIMECORRECTION_TCB,
        }

        pulsarlist = self.pulsars
        self.resume = kwargs.get("resume", self.resume)

        # heterodyne data
        if self.heterodyneddata:
            # re-heterodyne data

            # loop over pulsars
            for pulsar in pulsarlist:
                labeldict = {
                    "det": self.detector,
                    "gpsstart": int(self.starttime)
                    if self.starttime is not None
                    else None,
                    "gpsend": int(self.endtime) if self.endtime is not None else None,
                    "freqfactor": int(self.freqfactor),
                    "psr": pulsar,
                }

                if self.resume:
                    if os.path.isfile(self.outputfiles[pulsar].format(**labeldict)):
                        # skip this pulsar as heterodyne has already been performed
                        continue

                hetdata = TimeSeriesList()

                for hetfile in self.heterodyneddata[pulsar]:
                    # read in data
                    thishet = HeterodynedData.read(hetfile)

                    # check for consistent freqfactor
                    if thishet.freq_factor != self.freqfactor:
                        raise ValueError(
                            "Inconsistent frequency factor between heterodyned data and input"
                        )

                    origparams = thishet.par
                    psr = PulsarParametersPy(self._pulsars[pulsar])

                    units = [
                        u for u in [origparams["UNITS"], psr["UNITS"]] if u is not None
                    ]
                    if len(units) == 0:
                        # default to TCB
                        units = ["TCB"]

                    ephems = [
                        e for e in [origparams["EPHEM"], psr["EPHEM"]] if e is not None
                    ]
                    if len(ephems) == 0:
                        # default to DE405
                        ephems = ["DE405"]

                    for ephem, unit in zip(ephems, units):
                        if ephem not in self._ephemerides or unit not in self._timecorr:
                            edat, tdat = initialise_ephemeris(ephem=ephem, units=unit)

                            if ephem not in self._ephemerides:
                                self._ephemerides[ephem] = edat

                            if unit not in self._timecorr:
                                self._timecorr[unit] = tdat

                    # convert times to GPS time vector
                    gpstimes = lalpulsar.CreateTimestampVector(thishet.size)
                    for i, time in enumerate(thishet.times.value):
                        gpstimes.data[i] = lal.LIGOTimeGPS(time)

                    if thishet.include_ssb:
                        ephem = origparams["EPHEM"]
                        units = (
                            origparams["UNITS"]
                            if origparams["UNITS"] is not None
                            else "TCB"
                        )

                        # get original barycentring time delay
                        ssb = lalpulsar.HeterodynedPulsarGetSSBDelay(
                            origparams.PulsarParameters(),
                            gpstimes,
                            self.laldetector,
                            self._ephemerides[ephem],
                            self._timecorr[units],
                            ttype[units],
                        )

                        if thishet.include_bsb:
                            bsb = lalpulsar.HeterodynedPulsarGetBSBDelay(
                                origparams.PulsarParameters(),
                                gpstimes,
                                ssb,
                                self._ephemerides[ephem],
                            )
                        else:
                            bsb = lal.CreateREAL8Vector(thishet.size)
                            for i in range(thishet.size):
                                bsb.data[i] = 0.0
                    else:
                        # set delays to zero
                        ssb = lal.CreateREAL8Vector(thishet.size)
                        bsb = lal.CreateREAL8Vector(thishet.size)
                        for i in range(thishet.size):
                            ssb.data[i] = 0.0
                            bsb.data[i] = 0.0

                    # get the heterodyne glitch phase (which should be zero)
                    if thishet.include_glitch:
                        glphase = lalpulsar.HeterodynedPulsarGetGlitchPhase(
                            origparams.PulsarParameters(), gpstimes, ssb, bsb
                        )
                    else:
                        glphase = None

                    # get fitwaves phase
                    if thishet.include_fitwaves:
                        fwphase = lalpulsar.HeterodynedPulsarGetFITWAVESPhase(
                            origparams.PulsarParameters(),
                            gpstimes,
                            ssb,
                            origparams["F0"],
                        )
                    else:
                        fwphase = None

                    ephem = psr["EPHEM"] if psr["EPHEM"] is not None else "DE405"
                    units = psr["UNITS"] if psr["UNITS"] is not None else "TCB"

                    # calculate phase
                    fullphase = lalpulsar.HeterodynedPulsarPhaseDifference(
                        psr.PulsarParameters(),
                        origparams.PulsarParameters(),
                        gpstimes,
                        self.freqfactor,
                        ssb,
                        int(
                            self.includessb
                        ),  # the SSB delay should be updated compared to hetSSBdelay
                        bsb,
                        int(
                            self.includebsb
                        ),  # the BSB delay should be updated compared to hetBSBdelay
                        glphase,
                        int(
                            self.includeglitch
                        ),  # the glitch phase should be updated compared to glphase
                        fwphase,
                        int(
                            self.includefitwaves
                        ),  # the FITWAVES phase should be updated compare to fwphase
                        self.laldetector,
                        self._ephemerides[ephem],
                        self._timecorr[units],
                        ttype[units],
                    )

                    # perform heterodyne and resample
                    hetts = thishet.heterodyne(
                        2.0 * np.pi * fullphase.data,
                        stride=(1 / self.resamplerate),
                        datasegments=self.segments,
                    )

                    if hetts is not None:
                        hetdata.append(hetts.as_timeseries())

                # output the data
                times = np.array(
                    [v for d in hetdata for v in d.times.value], dtype=float
                )
                data = hetdata.join(gap="ignore")
                data.times = times  # preserve uneven time stamps

                # convert to HeterodynedData
                het = HeterodynedData(
                    data=data,
                    detector=self.detector,
                    par=self._pulsars[pulsar],
                    freqfactor=self.freqfactor,
                )
                het.include_ssb = self.includessb
                het.include_bsb = self.includebsb
                het.include_glitch = self.includeglitch
                het.include_fitwaves = self.includefitwaves

                het.write(self.outputfiles[pulsar].format(**labeldict), overwrite=True)
        else:
            self._datadict = {}

            if not self.includessb:
                if self.includebsb or self.includeglitch or self.includefitwaves:
                    raise ValueError(
                        "includessb must be True if trying to include binary evolution, glitches or FITWAVES parameters"
                    )

            if self.crop > self.stride:
                raise ValueError("Stride length must be longer than cropping size")

            if self.resume:
                # check for existence of previous files and find where they got
                # up to, so we can resume at that point
                minend = np.inf
                self._filter_history = {}
                for pulsar in list(pulsarlist):
                    labeldict = {
                        "det": self.detector,
                        "gpsstart": int(self.starttime),
                        "gpsend": int(self.endtime),
                        "freqfactor": int(self.freqfactor),
                        "psr": pulsar,
                    }
                    pfile = self.outputfiles[pulsar].format(**labeldict)
                    if os.path.isfile(pfile) and os.path.getsize(pfile) > 0:
                        try:
                            prevdata = HeterodynedData.read(pfile)
                        except OSError:
                            # could not read file, so ignore this pulsar
                            minend = self.starttime
                            continue

                        endtime = prevdata.times.value[-1]

                        if endtime >= self.endtime - self.resamplerate / 2 - self.crop:
                            # pulsar already completed, so can be skipped
                            pulsarlist.remove(pulsar)
                        else:
                            self._datadict[pulsar] = TimeSeriesList()
                            self._datadict[pulsar].append(prevdata.as_timeseries())

                            # get the time at which to resume
                            if endtime < minend:
                                minend = endtime + self.resamplerate / 2

                            self._filter_history[pulsar] = prevdata.filter_history
                    else:
                        minend = self.starttime

                if len(pulsarlist) == 0:
                    # no pulsars to reanalyse
                    print("Heterodyne for all pulsars is complete")
                    return

                # crop any in case some pulsars run past minend
                for pulsar in list(pulsarlist):
                    if pulsar in self._datadict:
                        cropped = self._datadict[pulsar][0].crop(
                            end=minend + self.resamplerate / 2, copy=True
                        )
                        self._datadict[pulsar][0] = cropped

                # reset starttime (and store original)
                self._origstart = self.starttime
                self._origstarts = [seg[0] for seg in self.segments]
                self.starttime = minend

            # loop over segments
            samplerates = []
            for segment in self.segments:
                # skip any segments that are too short
                if not hasattr(self, "_origstarts"):
                    seglen = segment[1] - segment[0]
                    if seglen < (2 * self.crop + 1 / self.resamplerate):
                        continue
                else:
                    if segment[0] not in self._origstarts:
                        # we are part way through a segment (having resumed),
                        # so would only be cropping the end of the data
                        seglen = segment[1] - segment[0]
                        if seglen < (self.crop + 1 / self.resamplerate):
                            continue

                # loop within segment in steps of "stride"
                curendtime = segment[0]
                counter = 0
                while curendtime < (segment[1] - self.crop):
                    curstarttime = segment[0] + counter * self.stride
                    curendtime = segment[0] + (counter + 1) * self.stride

                    if curendtime >= (segment[1] - self.crop):
                        # last part of segment
                        curendtime = segment[1] - self.crop

                        # make sure segment is long enough
                        if (curendtime - curstarttime) < (1 / self.resamplerate):
                            break

                    # download/read data
                    datakwargs = kwargs.copy()
                    datakwargs.update(
                        dict(starttime=int(curstarttime), endtime=int(curendtime))
                    )

                    data = self.get_frame_data(**datakwargs)

                    # check for consistent sample rate
                    samplerates.append(data.sample_rate.value)
                    if samplerates[0] != samplerates[-1]:
                        raise ValueError("Inconsistent sample rates in data!")

                    # set up the filters (this will only happen on first pass)
                    if self.filterknee is not None:
                        self._setup_filters(self.filterknee, data.sample_rate.value)

                    # convert times to GPS time vector
                    gpstimes = lalpulsar.CreateTimestampVector(data.size)
                    for i, time in enumerate(data.times.value):
                        gpstimes.data[i] = lal.LIGOTimeGPS(time)

                    # get times for interpolation if required
                    if self.interpolationstep > 0 and self.includessb:
                        idxstep = int(data.sample_rate.value * self.interpolationstep)
                        ntimes = int(np.ceil(data.size / idxstep)) + 1
                        gpstimesint = lalpulsar.CreateTimestampVector(ntimes)
                        for i, time in enumerate(data.times.value[::idxstep]):
                            gpstimesint.data[i] = lal.LIGOTimeGPS(time)
                        # include final time value
                        gpstimesint.data[-1] = lal.LIGOTimeGPS(data.times.value[-1])
                        timesint = np.append(
                            data.times.value[::idxstep], [data.times.value[-1]]
                        )

                    # loop over pulsars
                    for pulsar in pulsarlist:
                        if pulsar not in self._datadict:
                            self._datadict[pulsar] = TimeSeriesList()

                        # read pulsar parameter file
                        psr = PulsarParametersPy(self._pulsars[pulsar])

                        # initialise ephemerides if required
                        edat = lalpulsar.EphemerisData()
                        tdat = None
                        units = "TCB"
                        if self.includessb:
                            ephem = (
                                psr["EPHEM"] if psr["EPHEM"] is not None else "DE405"
                            )
                            units = psr["UNITS"] if psr["UNITS"] is not None else "TCB"

                            if ephem is None:
                                raise ValueError(
                                    "Pulsar '{}' has no 'EPHEM' value set".format(
                                        pulsar
                                    )
                                )

                            if (
                                ephem not in self._ephemerides
                                or units not in self._timecorr
                            ):
                                edat, tdat = initialise_ephemeris(
                                    ephem=ephem, units=units
                                )

                                if ephem not in self._ephemerides:
                                    self._ephemerides[ephem] = edat

                                if units not in self._timecorr:
                                    self._timecorr[units] = tdat

                            if self.interpolationstep > 0:
                                # calculate SSB delay and BSB delay at interpolation nodes
                                ssbdelay = lalpulsar.HeterodynedPulsarGetSSBDelay(
                                    psr.PulsarParameters(),
                                    gpstimesint,
                                    self.laldetector,
                                    self._ephemerides[ephem],
                                    self._timecorr[units],
                                    ttype[units],
                                )

                                # create interpolation function
                                k = (len(timesint) - 1) if len(timesint) < 4 else 3
                                tckssb = splrep(timesint, ssbdelay.data, k=k)
                                ssbdelayint = lal.CreateREAL8Vector(data.size)
                                ssbdelayint.data = splev(data.times.value, tckssb)

                                if self.includebsb:
                                    # calculate BSB delay
                                    bsbdelay = lalpulsar.HeterodynedPulsarGetBSBDelay(
                                        psr.PulsarParameters(),
                                        gpstimesint,
                                        ssbdelay,
                                        self._ephemerides[ephem],
                                    )

                                    # create interpolation function
                                    tckbsb = splrep(timesint, bsbdelay.data, k=k)
                                    bsbdelayint = lal.CreateREAL8Vector(data.size)
                                    bsbdelayint.data = splev(data.times.value, tckbsb)
                                else:
                                    bsbdelayint = None

                                # get the heterodyne glitch phase
                                if self.includeglitch:
                                    glphase = lalpulsar.HeterodynedPulsarGetGlitchPhase(
                                        psr.PulsarParameters(),
                                        gpstimesint,
                                        ssbdelay,
                                        bsbdelay,
                                    )

                                    # create interpolation function (note due to the minus sign in
                                    # the heterodyne the glitch phase sign needs to be flipped)
                                    tckglph = splrep(timesint, -1.0 * glphase.data, k=k)
                                    glphaseint = lal.CreateREAL8Vector(data.size)
                                    glphaseint.data = splev(data.times.value, tckglph)
                                else:
                                    glphaseint = None

                                # get fitwaves phase
                                if self.includefitwaves:
                                    fwphase = (
                                        lalpulsar.HeterodynedPulsarGetFITWAVESPhase(
                                            psr.PulsarParameters(),
                                            gpstimesint,
                                            ssbdelay,
                                            psr["F0"],
                                        )
                                    )

                                    # create interpolation function (note due to the minus sign in
                                    # the heterodyne the fitwaves phase sign needs to be flipped)
                                    tckfwph = splrep(timesint, -1.0 * fwphase.data, k=k)
                                    fwphaseint = lal.CreateREAL8Vector(data.size)
                                    fwphaseint.data = splev(data.times.value, tckfwph)
                                else:
                                    fwphaseint = None

                        # get phase evolution
                        useint = self.interpolationstep > 0 and self.includessb
                        phase = lalpulsar.HeterodynedPulsarPhaseDifference(
                            psr.PulsarParameters(),
                            None,
                            gpstimes,
                            self.freqfactor,
                            ssbdelayint if useint else None,
                            0 if useint else int(self.includessb),
                            bsbdelayint if useint else None,
                            0 if useint else int(self.includebsb),
                            glphaseint if useint else None,
                            0 if useint else int(self.includeglitch),
                            fwphaseint if useint else None,
                            0 if useint else int(self.includefitwaves),
                            self.laldetector,
                            edat if not self.includessb else self._ephemerides[ephem],
                            tdat if not self.includessb else self._timecorr[units],
                            ttype[units],
                        )

                        # heterodyne data
                        datahet = fast_heterodyne(data, -phase.data)

                        # filter data
                        self._filter_data(pulsar, datahet)

                        # downsample data
                        stridesamp = int(
                            (1 / self.resamplerate) * datahet.sample_rate.value
                        )
                        nsteps = int(datahet.size // stridesamp)
                        datadown = TimeSeries(np.zeros(nsteps, dtype=complex))
                        datadown.__array_finalize__(datahet)

                        datadown.sample_rate = self.resamplerate
                        for step in range(nsteps):
                            istart = int(stridesamp * step)
                            idx = slice(istart, istart + stridesamp)
                            datadown.value[step] = datahet.value[idx].mean()

                        # crop filter response from data
                        if counter == 0:
                            docrop = True
                            if hasattr(self, "_origstarts"):
                                # don't crop if resuming within a segment
                                if data.t0.value not in self._origstarts:
                                    docrop = False

                            if self.crop > 0 and docrop:
                                datadown = datadown.crop(
                                    start=data.t0.value + self.crop, copy=True
                                )

                        # centre the time stamps to the average window
                        datadown.t0 = datadown.t0.value + 0.5 / self.resamplerate

                        # store the heterodyned and downsampled data
                        self._datadict[pulsar].append(datadown)

                    counter += 1

            # output data
            self._write_current_pulsars()

    def _write_current_pulsars(self):
        # get arguments passed to Heterodyne
        sig = inspect.signature(Heterodyne)
        hetargs = {}
        for parameter in sig.parameters:
            if hasattr(self, parameter):
                hetargs[parameter] = getattr(self, parameter)

        hetargs["segmentlist"] = self.segments

        # add DAG configuration file data if present
        cf = None
        if self.cwinpy_heterodyne_dag_config_file is not None:
            if os.path.isfile(self.cwinpy_heterodyne_dag_config_file):
                with open(self.cwinpy_heterodyne_dag_config_file) as fp:
                    cf = fp.read()

        # output heterodyned data
        for pulsar in self._datadict:
            # set hetargs to just contain information for the individual pulsar
            hetargs["pulsars"] = pulsar
            hetargs["pulsarfiles"] = self.pulsarfiles[pulsar]
            hetargs["output"] = os.path.split(self.outputfiles[pulsar])[0]

            # get time stamps
            times = np.empty((0,), dtype=float)
            for d in self._datadict[pulsar]:
                times = np.append(times, d.times.value)
                del d.xindex  # delete times (otherwise join has issues!)

            data = self._datadict[pulsar].join(gap="ignore")

            # convert to HeterodynedData
            het = HeterodynedData(
                data=data.value,
                times=times,  # preserve uneven time stamps
                detector=self.detector,
                par=self._pulsars[pulsar],
                freqfactor=self.freqfactor,
                bbminlength=data.size,  # don't perform Bayesian blocks
                window=0,  # don't compute a running median
            )
            het.include_ssb = self.includessb
            het.include_bsb = self.includebsb
            het.include_glitch = self.includeglitch
            het.include_fitwaves = self.includefitwaves
            het.heterodyne_arguments = hetargs

            if cf is not None:
                het.cwinpy_heterodyne_dag_config = cf

            # save filter history from the forward pass
            history = []
            for idx in range(len(self._filters[pulsar])):
                history.append(
                    (
                        self._filters[pulsar][idx][0].history.data,
                        self._filters[pulsar][idx][1].history.data,
                    )
                )
            het.filter_history = history

            labeldict = {
                "det": self.detector,
                "gpsstart": int(self.starttime)
                if not hasattr(self, "_origstart")
                else int(self._origstart),
                "gpsend": int(self.endtime),
                "freqfactor": int(self.freqfactor),
                "psr": pulsar,
            }
            het.write(self.outputfiles[pulsar].format(**labeldict), overwrite=True)

    def _write_current_pulsars_and_exit(self, signum=None, frame=None):
        """
        Output current heterodyned data and exit in case of unexpected
        termination.
        """

        # use try statement in case exit happens during function call
        try:
            self._write_current_pulsars()
        except Exception:
            pass
        os._exit(self.exit_code)

    @property
    def heterodyneddata(self):
        """
        A dictionary of file paths (in lists) to heterodyned data files, keyed
        to pulsar names.
        """

        return self._heterodyneddata

    @heterodyneddata.setter
    def heterodyneddata(self, hetdata):
        self._heterodyneddata = {}

        if isinstance(hetdata, str):
            # check for single file or directory
            filelist, isfile = self._heterodyned_data_file_check(hetdata)

            if isfile:
                # try reading the data
                het = HeterodynedData.read(filelist[0])
                pname = get_psr_name(het.par)
                self._heterodyneddata[pname] = filelist
                if pname not in self.pulsars:
                    # set pulsars parameters from file
                    self._pulsars[pname] = het.par
            else:
                # return a reverse sorted list so, for example, J0000+0000AA comes
                # before J0000+0000A
                for pulsar in sorted(self.pulsars, reverse=True):
                    # get any files for each pulsar (assuming they contain the pulsar name)
                    pulsarfiles = [
                        f for f in filelist if (pulsar in f) and (os.path.isfile(f))
                    ]

                    if len(pulsarfiles) > 0:
                        self._heterodyneddata[pulsar] = pulsarfiles
                    else:
                        raise RuntimeError(
                            "No files found for pulsar '{}'".format(pulsar)
                        )
        elif isinstance(hetdata, dict):
            # get dictionary
            for key in hetdata:
                filelist, isfile = self._heterodyned_data_file_check(hetdata[key])

                if not isfile:
                    # get files for specific pulsar (assuming they contain the pulsar name)
                    pulsarfiles = [
                        f for f in filelist if (key in f) and (os.path.isfile(f))
                    ]

                    if len(pulsarfiles) == 0:
                        raise RuntimeError(
                            "No files found for pulsar '{}'".format(pulsar)
                        )
                else:
                    pulsarfiles = filelist

                if key not in self.pulsars:
                    # get pulsar information from first file
                    het = HeterodynedData.read(pulsarfiles[0])

                    if key == get_psr_name(het.par):
                        self._pulsars[key] = het.par
                    else:
                        raise KeyError("Inconsistent pulsar keys")

                self._heterodyneddata[key] = pulsarfiles
        elif hetdata is not None:
            raise TypeError("Heterodyned data must be a string or dictionary")

    def _heterodyned_data_file_check(self, hetdata):
        """
        Check if heterodyned data has been passed as a file or directory.
        Return list of files (by recursively globbing the directory, if a
        directory is given).

        Parameters
        ----------
        hetdata: str
            A string with a single file path or a directory path.

        Returns
        -------
        hetfiles: list
            A list of heterodyned data files
        isfile: bool
            A boolean stating whether the input ``hetdata`` was a file or not.
        """

        if isinstance(hetdata, str):
            hetdata = [hetdata]
        elif not isinstance(hetdata, list):
            raise TypeError(
                "Heterodyneddata must be a string or list giving a file or directory path"
            )

        # check if a file by testing for a hdf5-type or txt extension
        hetfiles = []
        isfile = False
        for hetfile in hetdata:
            if os.path.splitext(hetfile)[1] in self.extensions:
                # try reading the data
                try:
                    het = HeterodynedData.read(hetfile)
                except Exception as e:
                    raise IOError(e.args[0])

                if het.par is None:
                    raise AttributeError(
                        "Heterodyned data '{}' contains no pulsar parameter file".format(
                            hetfile
                        )
                    )
                else:
                    hetfiles.append(hetfile)
                    isfile = True
            elif os.path.isdir(hetfile):
                # glob for file types
                curhetfiles = [
                    str(f.resolve())
                    for ext in self.extensions
                    for f in Path(hetfile).rglob("*{}".format(ext))
                ]

                if len(curhetfiles) > 0:
                    # try reading first file
                    try:
                        het = HeterodynedData.read(curhetfiles[0])
                    except Exception as e:
                        raise IOError(e.args[0])

                    hetfiles.extend(curhetfiles)
                    isfile = False
                else:
                    raise RuntimeError(
                        "No files found in directory '{}'".format(hetfile)
                    )
            else:
                raise ValueError("hetdata must be a file or directory path")

        return hetfiles, isfile

    def _setup_filters(self, filterknee, samplerate):
        """
        Set up the 9th order low-pass Butterworth filters to apply to the data
        for each pulsars. This is performed by applying a 3rd order filter
        three times.

        Parameters
        ----------
        filterknee: float
            The filter knee frequency in Hz.
        samplerate: float
            The data sample rate in Hz
        """

        if not hasattr(self, "_filters"):
            self._filters = {}
        else:
            return

        wc = np.tan(np.pi * filterknee / samplerate)

        # set zero pole gain values
        zpg = lal.CreateCOMPLEX16ZPGFilter(0, 3)
        zpg.poles.data[0] = (wc * np.sqrt(3.0) / 2.0) + 1j * (wc * 0.5)
        zpg.poles.data[1] = 1j * wc
        zpg.poles.data[2] = -(wc * np.sqrt(3.0) / 2.0) + 1j * (wc * 0.5)
        zpg.gain = 1j * wc * wc * wc
        lal.WToZCOMPLEX16ZPGFilter(zpg)

        for pulsar in self.pulsars:
            self._filters[pulsar] = []

            # create IIR filters
            for i in range(3):
                filterRe = lal.CreateREAL8IIRFilter(zpg)
                filterIm = lal.CreateREAL8IIRFilter(zpg)

                if hasattr(self, "_filter_history"):
                    if pulsar in self._filter_history:
                        # set previous history
                        filterRe.history.data = self._filter_history[pulsar][i][0]
                        filterIm.history.data = self._filter_history[pulsar][i][1]

                self._filters[pulsar].append((filterRe, filterIm))

    def _filter_data(self, pulsar, data, forwardsonly=False):
        """
        Apply the low pass filters to the data for a particular pulsar.

        Parameters
        ----------
        pulsar: str
            The name of the pulsar who's data is being filtered.
        data: array_like
            The array of complex heterodyned data to be filtered.
        forwardswonly: bool
            Set to True to only filter the data in the forwards direction. This
            means that the filter phase lag will still be present.
        """

        if pulsar not in self._filters:
            raise KeyError("No filter set for pulsar '{}'".format(pulsar))

        dr = lal.CreateREAL8Vector(len(data))
        dr.data = data.value.real
        di = lal.CreateREAL8Vector(len(data))
        di.data = data.value.imag

        # apply filters (3 passes of the third order Butterworth filters)
        for idx in range(3):
            # run filter forwards
            lal.IIRFilterREAL8Vector(dr, self._filters[pulsar][idx][0])
            lal.IIRFilterREAL8Vector(di, self._filters[pulsar][idx][1])

            if not forwardsonly:
                # run filter backwards
                historyr = self._filters[pulsar][idx][0].history.data.copy()
                historyi = self._filters[pulsar][idx][1].history.data.copy()

                lal.IIRFilterReverseREAL8Vector(dr, self._filters[pulsar][idx][0])
                lal.IIRFilterReverseREAL8Vector(di, self._filters[pulsar][idx][1])

                # restore the history to that from the forward pass
                self._filters[pulsar][idx][0].history.data = historyr
                self._filters[pulsar][idx][1].history.data = historyi

        data.value.real = dr.data
        data.value.imag = di.data

    @property
    def includessb(self):
        """
        A boolean stating whether the heterodyne includes Solar System
        barycentring.
        """

        try:
            return self._includessb
        except AttributeError:
            return False

    @includessb.setter
    def includessb(self, incl):
        self._includessb = bool(incl)

    @property
    def includebsb(self):
        """
        A boolean stating whether the heterodyne includes Binary System
        barycentring.
        """

        try:
            return self._includebsb
        except AttributeError:
            return False

    @includebsb.setter
    def includebsb(self, incl):
        self._includebsb = bool(incl)

    @property
    def includeglitch(self):
        """
        A boolean stating whether the heterodyne includes corrections for any
        glitch phase evolution.
        """

        try:
            return self._includeglitch
        except AttributeError:
            return False

    @includeglitch.setter
    def includeglitch(self, incl):
        self._includeglitch = bool(incl)

    @property
    def includefitwaves(self):
        """
        A boolean stating whether the heterodyne includes corrections for any
        red noise FITWAVES parameters.
        """

        try:
            return self._includefitwaves
        except AttributeError:
            return False

    @includefitwaves.setter
    def includefitwaves(self, incl):
        self._includefitwaves = bool(incl)

    def set_ephemeris(self, earthephemeris=None, sunephemeris=None, timeephemeris=None):
        """
        Initialise the solar system and time ephemeris data.

        Parameters
        ----------
        earthephemeris: dict
            A dictionary, keyed to ephemeris names, e.g., "DE405", pointing to
            the location of a file containing that ephemeris for the Earth.
        sunephemeris: dict
            A dictionary, keyed to ephemeris names, e.g., "DE405", pointing to
            the location of a file containing that ephemeris for the Sun.
        timeephemeris: dict
            A dictionary, keyed to time system name, which can be either "TCB"
            or "TDB", pointing to the location of a file containing that
            ephemeris for that time system.
        """

        if not hasattr(self, "_ephemerides"):
            self._ephemerides = {}

        if not hasattr(self, "_timecorr"):
            self._timecorr = {}

        if not hasattr(self, "_earthephemeris"):
            self._earthephemeris = earthephemeris

        if not hasattr(self, "_sunephemeris"):
            self._sunephemeris = sunephemeris

        if isinstance(earthephemeris, dict) and isinstance(sunephemeris, dict):
            if not hasattr(self, "_earthephemeris"):
                self._earthephemeris = earthephemeris
            else:
                self._earthephemeris.update(earthephemeris)

            if not hasattr(self, "_sunephemeris"):
                self._sunephemeris = sunephemeris
            else:
                self._sunephemeris.update(sunephemeris)

            for ephemtype in earthephemeris:
                if ephemtype not in sunephemeris:
                    raise KeyError(
                        "Earth and Sun ephemeris dictionaries must contain the same keys"
                    )

                self._ephemerides[ephemtype] = initialise_ephemeris(
                    earthfile=earthephemeris[ephemtype],
                    sunfile=sunephemeris[ephemtype],
                    ssonly=True,
                )

        if isinstance(timeephemeris, dict):
            if not hasattr(self, "_timeephemeris"):
                self._timeephemeris = timeephemeris
            else:
                self._timeephemeris.update(timeephemeris)

            for timetype in timeephemeris:
                self._timecorr[timetype] = initialise_ephemeris(
                    timefile=timeephemeris[timetype],
                    timeonly=True,
                )

    @property
    def earthephemeris(self):
        if hasattr(self, "_earthephemeris"):
            return self._earthephemeris
        else:
            return None

    @property
    def sunephemeris(self):
        if hasattr(self, "_sunephemeris"):
            return self._sunephemeris
        else:
            return None

    @property
    def timeephemeris(self):
        if hasattr(self, "_timeephemeris"):
            return self._timeephemeris
        else:
            return None


def remote_frame_cache(
    start,
    end,
    channels,
    frametype=None,
    verbose=False,
    write=None,
    append=False,
    **kwargs,
):
    """
    Generate a cache list of gravitational-wave data files using a remote
    server. This is based on the code from the :method:`~gwpy.timesseries.find`
    method.

    Examples
    --------

    Generate a cache list of frame files for two LIGO detectors for all O2
    data (this assumes you are accessing data on a LVC cluster, or have set
    the environment variable ``LIGO_DATAFIND_SERVER=datafind.ligo.org:443``
    for accessing data hosted via CVMFS):

    >>> start = 1164556817  # GPS for 2016-11-30 16:00:00 UTC
    >>> end = 1187733618    # GPS for 2017-08-25 22:00:00 UTC
    >>> channels = ['H1:DCH-CLEAN_STRAIN_C02', 'L1:DCH-CLEAN_STRAIN_C02']
    >>> frametype = ['H1_CLEANED_HOFT_C02', 'L1_CLEANED_HOFT_C02']
    >>> cache = remote_frame_cache(start, end, channels, frametype=frametype)

    Parameters
    ----------
    start: float, int
        The start time in GPS seconds for which to get the cache.
    end: float, int
        The end time in GPS seconds for which to get the cache.
    channels: str, list
        The channel, or list of channels (for example for different detectors),
        for which to get the cache.
    frametype: str, list
        The frametype or list of frame types, for which to get the cache. If
        both ``channels`` and ``frametype`` are lists, the channels and
        frame types should be ordered as pairs. If not given this is attempted
        to be determined based on the channel name.
    verbose: bool
        Output verbose information.
    host: str
        The server host. If not set this will be automatically determine if
        possible. To list access data available via CVMFS either set host to
        ``'datafind.ligo.org:443'`` or set the environment variable
        ``LIGO_DATAFIND_SERVER=datafind.ligo.org:443`.
    write: str
        A file path to write out the list of frame files to. Default is to not
        write out the frame list.
    append: bool
        If writing out to a file, this says to append to the file if it already
        exists. The default is False.
    **kwargs:
        See :meth:`gwpy.io.datafind.find_best_frametype` for additional
        keyword arguments.

    Returns
    -------
    cache: dict
        A dictionary of lists of frame files for each detector keyed to the
        detector prefix.
    """

    from gwpy.time import to_gps
    from gwpy.io import datafind as io_datafind
    from gwpy.detector import ChannelList
    from gwpy.utils import gprint

    start = to_gps(start)
    end = to_gps(end)

    if isinstance(channels, str):
        channels = [channels]
    elif not isinstance(channels, list):
        raise TypeError("Channel must be a string or list")

    # find frametype
    frametypes = {}
    if frametype is None:
        matched = io_datafind.find_best_frametype(channels, start, end, **kwargs)
        # flip dict to frametypes with a list of channels
        for name, ftype in matched.items():
            try:
                frametypes[ftype].append(name)
            except KeyError:
                frametypes[ftype] = [name]

        if verbose and len(frametypes) > 1:
            gprint("Determined {} frametypes to read".format(len(frametypes)))
        elif verbose:
            gprint("Determined best frametype as {}".format(list(frametypes.keys())[0]))
    else:
        if isinstance(frametype, list):
            if len(frametype) == len(channels):
                frametypes = {frametype[i]: [channels[i]] for i in range(len(channels))}
            else:
                raise ValueError("Number of frames types and channels must be equal")
        else:
            frametypes = {frametype: channels}

    # generate cache
    cache = {}
    for frametype, clist in frametypes.items():
        if verbose:
            verbose = "Reading {} frames".format(frametype)

        # parse as a ChannelList
        channellist = ChannelList.from_names(*clist)

        # find observatory for this group
        try:
            observatory = "".join(sorted(set(c.ifo[0] for c in channellist)))
            ifo = channellist[0].ifo
        except TypeError as exc:
            exc.args = ("Cannot parse list of IFOs from channel names",)
            raise

        if observatory not in cache:
            cache[ifo] = []

        # find frames
        subcache = io_datafind.find_urls(
            observatory,
            frametype,
            start,
            end,
            on_gaps=kwargs.get("on_gaps", "ignore"),
            host=kwargs.get("host", None),
        )

        if not subcache:
            print(
                "No {}-{} frame files found for [{}, {})".format(
                    observatory, frametype, start, end
                )
            )

        cache[ifo].extend(subcache)

    # write output to file
    if write is not None:
        try:
            # set whether to append to file or not
            format = "a" if append else "w"

            with open(write, format) as fp:
                for ifo in cache:
                    for frfile in cache[ifo]:
                        fp.write(frfile)
                        fp.write("\n")
        except Exception as e:
            raise IOError("Could not output cache file: {}".format(e))

    return cache


def local_frame_cache(
    path,
    recursive=True,
    starttime=None,
    endtime=None,
    extension="gwf",
    site=None,
    frametype=None,
    write=None,
    append=False,
):
    """
    Generate a list of locally stored frame file paths that are found within a
    given directory. The files can be restricted to those containing data
    between certain start and end times, certain detectors, and certain types,
    provided the file names are of the standard format as described in
    `LIGO-T010150 <https://dcc.ligo.org/LIGO-T010150/public>`_, e.g.:
    ``SITE-TYPE-GPSSTART-DURATION.gwf``. If the files are not of the standard
    format they can be returned, but the conditions cannot be checked.

    Parameters
    ----------
    path: str
        The directory path containing the frame files.
    recursive: bool
        The sets whether or not subdirectories within the main path are also
        searched for files. Defaults to True.
    starttime: int
        A GPS time giving the start time of the data required. If not set the
        earliest frame files found will be returned.
    endtime: int
        A GPS time giving the end time of the data required. If not set the
        latest frame files found will be returned.
    extension: str
        The frame file name extension. By default this will be "gwf".
    site: str
        A string giving the detector name. The first letter of this is assumed
        to give the ``SITE`` in a standard format file name. If not given,
        frames from any detector found within the path will be returned.
    frametype: str
        A string giving the frame type used for the ``TYPE`` value in the
        standard format file name. If not given, frames of any type found
        within the path will be returned.
    write: str
        A file path to write out the list of frame files to. Default is to not
        write out the frame list.
    append: bool
        If writing out to a file, this says to append to the file if it already
        exists. The default is False.

    Returns
    -------
    cache: list
        The list of frame files.
    """

    try:
        if recursive:
            files = Path(path).rglob("*.{}".format(extension))
        else:
            files = Path(path).glob("*.{}".format(extension))
    except Exception as e:
        raise IOError("Could not parse the path: {}".format(e))

    if starttime is None:
        starttime = -np.inf
    elif not isinstance(starttime, int):
        raise TypeError("starttime must be an integer")

    if endtime is None:
        endtime = np.inf
    elif not isinstance(endtime, int):
        raise TypeError("endtime must be an integer")

    cache = []
    filetimes = []
    for frfile in files:
        if not frfile.resolve().is_file():
            continue

        fileparts = frame_information(frfile)

        if len(fileparts) == 4:
            # file is in standard format
            filesite = fileparts[0][0]
            filetype = fileparts[1]

            if isinstance(site, str):
                if filesite != site[0]:
                    # sites do not match
                    continue

            if isinstance(frametype, str):
                if filetype != frametype:
                    # types do not match
                    continue

            filetime = fileparts[2]
            filedur = fileparts[3]

            if not (starttime < filetime + filedur and endtime > filetime):
                # times do not match
                continue

            # store file start times for sorting
            filetimes.append(filetime)
        else:
            # cannot check conditions
            raise IOError(
                "Frame file name '{}' is not the correct format".format(frfile)
            )

        cache.append("file://localhost{}".format(str(frfile.resolve())))

    cache = [x[1] for x in sorted(zip(filetimes, cache))]  # sort cache files om time

    # write output to file
    if write is not None:
        try:
            # set whether to append to file or not
            format = "a" if append else "w"

            with open(write, format) as fp:
                for frfile in cache:
                    fp.write(frfile)
                    fp.write("\n")
        except Exception as e:
            raise IOError("Could not output cache file: {}".format(e))

    return cache


def frame_information(framefile):
    """
    Get the site name, frame type, start time and duration of a given frame
    file. It is assumed that the provided the file names are of the standard
    format as described in
    `LIGO-T010150 <https://dcc.ligo.org/LIGO-T010150/public>`_, e.g.:
    ``SITE-TYPE-GPSSTART-DURATION.gwf``.

    Parameters
    ----------
    framefile: str
        The name of a frame file.

    Returns
    -------
    info: tuple
        A tuple containing the site, frame start time and duration
    """

    frameinfo = re.compile(
        r"(?P<site>[A-Z])-(?P<frtype>\S+)-(?P<frt0>[0-9]+)-(?P<frdur>[0-9]+).(gwf|hdf5|hdf|h5)"
    )
    match = frameinfo.match(os.path.basename(framefile))
    if match is None:
        raise IOError("Could not check frame file information")

    site = match.groupdict()["site"]
    frtype = match.groupdict()["frtype"]
    frt0 = int(match.groupdict()["frt0"])
    frdur = int(match.groupdict()["frdur"])

    return site, frtype, int(frt0), frdur


def generate_segments(
    starttime=None,
    endtime=None,
    includeflags=None,
    excludeflags=None,
    segmentfile=None,
    server=None,
    usegwosc=False,
    writesegments=None,
    appendsegments=False,
):
    """
    Generate a list of times to analysis based on data quality (DQ) segments.
    As mentioned in the `GWPy documentation
    <https://gwpy.github.io/docs/stable/segments/dqsegdb.html>`_ this requires
    access to the GW segment database, which is reserved for members of the
    LIGO-Virgo-KAGRA collaborations.

    To get segment lists for open data via GWOSC the "include_flags" argument
    should be, e.g., "DET_DATA" where "DET" is replaced by the detector prefix.
    The "usegwosc" argument should be set to True to be explicitly about using
    GWOSC. If looking at CW hardware injections in GWOSC data the exclude flag
    should be, e.g., "DET_NO_CW_HW_INJ", where again "DET" is replaced by the
    detector prefix. This will exclude times when the injections were not
    present.

    Parameters
    ----------
    starttime: int
        A GPS time giving the start time of the data required. If not set the
        earliest frame files found will be returned.
    endtime: int
        A GPS time giving the end time of the data required. If not set the
        latest frame files found will be returned.
    includeflags: str, list
        A string, or list, containing the DQ flags of segments to include.
        If this is a string it can be a single DQ flags, or a set of comma
        separated flags. Or, it can list each flag separately. The intersection
        of all included flags (logical AND) will be returned.
    excludeflags: str, list
        A string, or list, containing DQ flags of segments to exclude. If this
        is a string it can be a single DQ flags, or a set of comma separated
        flags. The intersection of times that are not excluded with those that
        are included flags will be returned.
    segmentfile: str
        If a file is given, containing two columns of segment start and end
        times, this will be read in and returned (using the start and end times
        to restrict the results).
    server: str
        The URL of the segment database server to use. If not given then the
        default server URL is used (see
        :func:`gwpy.segments.DataQualityFlag.query`).
    usegwosc: bool
        If querying for segments from GWOSC (see above) set this to True.
        Default is False.
    writesegments: str
        A string giving a file to output the segments to.
    appendsegments: bool
        If writing to a file set this to append to a pre-exiting file. Default
        is False.

    Returns
    -------
    segments: list
        A list of tuples pairs containing the start and end GPS times of the
        requested segments.
    """

    if not isinstance(starttime, (int, float)):
        if starttime is None and isinstance(segmentfile, str):
            starttime = -np.inf
        else:
            raise TypeError("starttime must be an integer or float")

    if not isinstance(endtime, (int, float)):
        if endtime is None and isinstance(segmentfile, str):
            endtime = np.inf
        else:
            raise TypeError("endtime must be an integer or float")

    if endtime <= starttime:
        raise ValueError("starttime must be before endtime")

    if isinstance(segmentfile, str):
        try:
            segmentsarray = np.loadtxt(segmentfile, comments=["#", "%"], dtype=float)
            segments = []
            for segment in segmentsarray:
                if segment[1] < starttime or segment[0] > endtime:
                    continue

                start = segment[0] if segment[0] > starttime else starttime
                end = segment[1] if segment[1] < endtime else endtime

                if start >= end:
                    continue

                segments.append((start, end))
        except Exception as e:
            raise IOError("Could not load segment file '{}': {}".format(segmentfile, e))

        return segments
    else:  # pragma: no cover
        if not isinstance(includeflags, (str, list)):
            raise TypeError("includeflags must be a string or list")
        else:
            if isinstance(includeflags, str):
                includeflags = [includeflags]

            includetypes = [flag for flags in includeflags for flag in flags.split(",")]

        if excludeflags is not None:
            if not isinstance(excludeflags, (str, list)):
                raise TypeError("excludeflags must be a string or list")
            else:
                if isinstance(excludeflags, str):
                    excludeflags = [excludeflags]

            excludetypes = [
                flag
                for flags in excludeflags
                if len(flags) > 0
                for flag in flags.split(",")
            ]

        segs = None
        serverkwargs = {}
        if isinstance(server, str):
            serverkwargs["url"] = server
        # get included segments
        for dqflag in includetypes:
            # create query
            if usegwosc:
                query = SegmentList(get_segments(dqflag, starttime, endtime))
            else:
                # use "active" segments
                query = DataQualityFlag.query(
                    dqflag, starttime, endtime, **serverkwargs
                ).active

            if segs is None:
                segs = SegmentList(query.copy())
            else:
                segs = segs & query

        # remove excluded segments
        if excludeflags is not None and segs is not None and len(excludetypes) > 0:
            for dqflag in excludetypes:
                if usegwosc:
                    query = SegmentList(get_segments(dqflag, starttime, endtime))
                else:
                    # use "active" segments
                    query = DataQualityFlag.query(
                        dqflag, starttime, endtime, **serverkwargs
                    ).active

                segs = segs & ~query

        # convert to list of tuples and output if required
        if isinstance(writesegments, str):
            format = "w" if not appendsegments else "a"

            try:
                fp = open(writesegments, format)
            except Exception as e:
                raise IOError(
                    "Could not open output file '{}': {}".format(writesegments, e)
                )

        # write out segment list
        segments = []
        if segs is not None:
            for thisseg in segs:
                segments.append((float(thisseg[0]), float(thisseg[1])))

                if isinstance(writesegments, str):
                    fp.write("{} {}\n".format(float(thisseg[0]), float(thisseg[1])))

        return segments
