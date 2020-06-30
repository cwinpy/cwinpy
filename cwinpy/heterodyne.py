"""
Classes for heterodyning strain data.
"""

import os

import lal
import lalpulsar
from gwosc.api import DEFAULT_URL as GWOSC_DEFAULT_HOST
from gwpy.timeseries import TimeSeries

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
    frtype: str
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
    outputfrcache: str
        If a string is given it should give a file path to which a list of
        gravitational-wave data file paths, as found by the code, will be
        written. If not given then the file list will not be output.
    appendfrcache: bool
        If writing out the frame cache to a file, set this to True to append
        to the file rather than overwriting. Default is False.
    frcache: str, list
        If you have a pregenerated cache of gravitational-wave file paths
        (either in a file or as a list of files) you can use them.
    segments: str, list
        A list or data segment start and end times (stored a tuple pairs) in
        the list. Or, an ascii text file containing segment start and end times
        in two columns. Or, a list of data DQ flags to use to generate a
        segment list.
    """

    def __init__(
        self,
        starttime=None,
        endtime=None,
        stride=3600,
        detector=None,
        frtype=None,
        channel=None,
        host=None,
        outputfrcache=None,
        appendfrcache=False,
        frcache=None,
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
        if frcache is None:
            self.frtype = frtype
            self.host = host
        else:
            self.frcache = frcache

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
    def frtype(self):
        """
        The frame type of data to get.
        """

        if hasattr(self, "_frtype"):
            return self._frtype
        else:
            return None

    @frtype.setter
    def frtype(self, frtype):
        if frtype is None or isinstance(frtype, str):
            self._frtype = frtype
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
    def frcache(self):
        """
        A file name, or list of files, containing gravitational-wave strain
        data.
        """

        if hasattr(self, "_frcache"):
            return self._frcache
        else:
            return None

    @frcache.setter
    def frcache(self, frcache):
        if frcache is None:
            self._frcache = None
        elif isinstance(frcache, (str, list)):
            if isinstance(frcache, str):
                if not os.path.isfile(frcache):
                    raise ValueError(
                        "Frame cache file '{}' does not exist".format(frcache)
                    )
            if isinstance(frcache, list):
                # check files exist
                for fr in frcache:
                    if not isinstance(fr, str):
                        raise TypeError("Frame cache list must contain strings")
                    if not os.path.isfile(fr):
                        raise ValueError(
                            "Frame cache file '{}' does not exist".format(fr)
                        )
            self._frcache = frcache
        else:
            raise TypeError("Frame cache must be a string or list")

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

    def get_frame_data(self, starttime=None, endtime=None, **gwosc_data_kwargs):
        """
        Get gravitational-wave frame/hdf5 data between a given start and end
        time in GPS seconds.

        Parameters
        ----------
        starttime: int, float
            The start time of the data to extract in GPS seconds.
        endtime: int, float
            The end time of the data to extract in GPS seconds.
        **gwosc_data_kwargs:
            A set of keyword arguments to be passed to
            :class:`gwpy.timeseries.TimeSeries.fetch_open_data`.

        Returns
        -------
        data: TimeSeries
            A :class:`gwpy.timeseries.TimeSeries` containing the data.
        """

        starttime = starttime if starttime is not None else self.starttime
        endtime = endtime if endtime is not None else self.endtime

        if starttime is None:
            raise ValueError("A start time is not set")

        if endtime is None:
            raise ValueError("An end time is not set")

        if self.channel is None and self.host != GWOSC_DEFAULT_HOST:
            raise ValueError("No channel name has been set")

        if self.frcache is not None:
            # read data from cache
            try:
                data = TimeSeries.read(
                    self.frcache, self.channel, start=starttime, end=endtime
                )
            except Exception as e:
                raise IOError("Could not read in frame data from cache: {}".format(e))
        else:
            # download data
            try:
                if self.host == GWOSC_DEFAULT_HOST:
                    # get GWOSC data
                    data = TimeSeries.fetch_open_data(
                        self.detector,
                        starttime,
                        endtime,
                        host=self.host,
                        **gwosc_data_kwargs,
                    )
                else:
                    data = TimeSeries.get(
                        self.channel,
                        starttime,
                        endtime,
                        host=self.host,
                        frametype=self.frtype,
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
