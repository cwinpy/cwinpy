"""
Classes for heterodyning strain data.
"""

import glob
import os

import lal
import lalframe
import lalpulsar
import numpy as np
from gwosc.api import DEFAULT_URL as GWOSC_DEFAULT_HOST
from gwpy.io.cache import is_cache, read_cache
from gwpy.timeseries import TimeSeries, TimeSeriesDict

# from gwpy.segments import DataQualityFlag

# Things that this class should be able to do:
#  - find requested gravitational-wave data
#  - find requested segment lists
#  - coarse heterodyne a chunk of data for multiple pulsars
#  - fine heterodyne data (potentially reading in, sorting, and combining multiple coarse data chunks)
# Things that a pipeline should be able to do
#  - find missing data, i.e., if an analysis chunk has failed notice this


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
        the data reading in ``stride`` seconds each time. Defaults to 3600
    detector: str
        A string given the name of the detector for which the data is being
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
    segments: str, list
        A list or data segment start and end times (stored a tuple pairs) in
        the list. Or, an ascii text file containing segment start and end times
        in two columns. Or, a list of data DQ flags to use to generate a
        segment list. See, e.g., the GWPy documentation
        `here <https://gwpy.github.io/docs/stable/segments/index.html>`_.
    excludesegments: str, list
        A string, or list of strings, giving DQ flags to exclude from the
        segment list.
    """

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
        segments=None,
        excludesegments=None,
        outputsegments=None,
        heterodyne="coarse",
        heterodyneddata=None,
        pulsars=None,
        basedir=None,
        filterknee=0.25,
        resamplerate=1.0,
        include_ssb=True,
        include_bsb=True,
        include_glitch=True,
        include_fitwaves=True,
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

    @property
    def starttime(self):
        """
        The start time of the heterodyned data in GPS seconds.
        """

        if hasattr(self, "_starttime"):
            return self._starttime
        else:
            return None

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

        if hasattr(self, "_detector"):
            return self._detector
        else:
            return None

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
        A lal.Detector structure containing the detector information.
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

        if hasattr(self, "_channel"):
            return self._channel
        else:
            return None

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
                    cache = local_frame_cache(
                        framecache,
                        starttime=starttime,
                        endtime=endtime,
                        frametype=frametype,
                        recursive=kwargs.get("recursive", False),
                        site=kwargs.get("site", self.detector),
                        extension=kwargs.get("extension", "gwf"),
                        write=outputframecache,
                        append=appendframecache,
                    )
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
                    frt0, frdur = frame_time_duration(frfile)

                    if starttime < frt0 and endtime >= frt0:
                        startread = frt0

                        if endtime > frt0 + frdur:
                            endread = frt0 + frdur
                        else:
                            endread = endtime
                    elif starttime >= frt0 and endtime >= frt0:
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
                    data.append(thisdata)

                # extract channel from dictionary
                data = data[channel]
            except Exception as e:
                raise IOError("Could not read in frame data from cache: {}".format(e))
        else:
            # download data
            try:
                if host == GWOSC_DEFAULT_HOST:
                    # get GWOSC data
                    data = TimeSeries.fetch_open_data(
                        kwargs.get("site", self.detector), starttime, endtime, host=host
                    )
                else:  # pragma: no cover
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

    def heterodyne(self, type="coarse"):
        if type == "coarse":
            self._coarse_heterodyne()
        elif type == "fine":
            self._fine_heterodyne()
        else:
            raise ValueError("Heterodyne type must be 'coarse' or 'fine'")


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
    >>> cache = generate_cache(start, end, channels, frametype=frametype)

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
        See :method:`gwpy.io.datafind.find_best_frametype` for additional
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
    recursive=False,
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
        searched for files.
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
        files = glob.glob(
            os.path.join(path, "*.{}".format(extension)), recursive=recursive
        )
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
    for frfile in files:
        if not os.path.isfile(os.path.realpath(frfile)):
            continue

        basefile = os.path.basename(frfile)

        fileparts = basefile.rstrip(".{}".format(extension)).split("-")

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

            try:
                filetime = int(fileparts[2])
                filedur = int(fileparts[3])
            except Exception:
                if np.isfinite(starttime) or np.isfinite(endtime):
                    print("Warning: cannot check times for file '{}'".format(frfile))
                filetime = None

            if filetime is not None:
                if not (starttime < filetime + filedur and endtime > filetime):
                    # times do not match
                    continue
        else:
            # cannot check conditions
            if (
                np.isfinite(starttime)
                or np.isfinite(endtime)
                or site is not None
                or frametype is not None
            ):
                print("Warning: cannot check conditions for file '{}".format(frfile))

        cache.append("file://localhost{}".format(os.path.realpath(frfile)))

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


def frame_time_duration(framefile):
    """
    Get the start time and duration of a given frame file.

    Parameters
    ----------
    framefile: str
        The name of a frame file.

    Returns
    -------
    info: tuple
        A tuple containing the frame start time and duration
    """

    try:
        frp = lalframe.FrFileOpenURL(framefile)
        frt0 = lal.LIGOTimeGPS()
        frt0 = lalframe.FrFileQueryGTime(frt0, frp, 0)
        frdur = lalframe.lalframe.FrFileQueryDt(frp, 0)
    except RuntimeError:
        raise IOError("Could not check frame file information")

    return int(frt0), frdur


# def generate_segments():
