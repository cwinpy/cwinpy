"""
Classes for heterodyning strain data.
"""

import lal
import lalpulsar

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
        ``datafind.ligo.org:443`` for open data available via CVMFS. See also
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
    ):
        # set analysis times
        self.starttime = starttime
        self.endtime = endtime

        # set detector
        self.detector = detector

        # set frame type and channel
        if frcache is None:
            self.frtype = frtype
        self.channel = channel

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
