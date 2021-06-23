"""
This module provides the :class:`~cwinpy.signal.HeterodynedCWSimulator` class
for simulating a signal from a continuous wave source after application of a
heterodyne as described in Equations 7 and 8 of [1]_.

Examples
========

An example usage to generate the complex heterodyned signal time series is:

.. code-block::

    from cwinpy.signal import HeterodynedCWSimulator
    from lalpulsar.PulsarParametersWrapper import PulsarParametersPy
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    import numpy as np

    # set the pulsar parameters
    par = PulsarParametersPy()
    pos = SkyCoord("01:23:34.5 -45:01:23.4", units=("hourangle", "deg"))
    par["RAJ"] = pos.ra.rad
    par["DECJ"] = pos.dec.rad
    par["F"] = [123.456789, -9.87654321e-12]  # frequency and first derivative
    par["PEPOCH"] = Time(58000, format="mjd", scale="tt").gps  # frequency epoch
    par["H0"] = 5.6e-26     # GW amplitude
    par["COSIOTA"] = -0.2   # cosine of inclination angle
    par["PSI"] = 0.4        # polarization angle (rads)
    par["PHI0"] = 2.3       # initial phase (rads)

    # set the GPS times of the data
    times = np.arange(1000000000.0, 1000086400.0, 3600, dtype=np.float128)

    # set the detector
    det = "H1"  # the LIGO Hanford Observatory

    # create the HeterodynedCWSimulator object
    het = HeterodynedCWSimulator(par, det, times=times)

    # get the model complex strain time series
    model = het.model(usephase=True)

An example of getting the time series for a signal that has phase parameters
that are not identical to the heterodyned parameters would be:

.. code-block::

   from cwinpy.signal import HeterodynedCWSimulator
   from lalpulsar.PulsarParametersWrapper import PulsarParametersPy
   from astropy.time import Time
   from astropy.coordinates import SkyCoord
   import numpy as np

   # set the "heterodyne" pulsar parameters
   par = PulsarParametersPy()
   pos = SkyCoord("01:23:34.5 -45:01:23.4", units=("hourangle", "deg"))
   par["RAJ"] = pos.ra.rad
   par["DECJ"] = pos.dec.rad
   par["F"] = [123.4567, -9.876e-12]  # frequency and first derivative
   par["PEPOCH"] = Time(58000, format="mjd", scale="tt").gps  # frequency epoch

   # set the times
   times = np.arange(1000000000.0, 1000000600.0, 60, dtype=np.float128)

   # set the detector
   det = "H1"  # the LIGO Hanford Observatory

   # create the HeterodynedCWSimulator object
   het = HeterodynedCWSimulator(par, det, times=times)

   # set the updated parameters
   parupdate = PulsarParametersPy()
   par["RAJ"] = pos.ra.rad
   par["DECJ"] = pos.dec.rad
   parupdate["F"] = [123.456789, -9.87654321e-12]  # different frequency and first derivative
   par["PEPOCH"] = Time(58000, format="mjd", scale="tt").gps  # frequency epoch
   parupdate["H0"] = 5.6e-26     # GW amplitude
   parupdate["COSIOTA"] = -0.2   # cosine of inclination angle
   parupdate["PSI"] = 0.4        # polarization angle (rads)
   parupdate["PHI0"] = 2.3       # initial phase (rads)

   # get the model complex strain time series
   model = het.model(parupdate, usephase=True, updateSSB=True)

Signal references
=================

.. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
    <https:arxiv.org/abs/1705.08978v1>`_, 2017.

"""

import lal
import lalpulsar
import numpy as np
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy

from .utils import initialise_ephemeris, is_par_file


class HeterodynedCWSimulator(object):
    def __init__(
        self,
        par,
        det,
        times=None,
        earth_ephem=None,
        sun_ephem=None,
        time_corr=None,
        ephem="DE405",
        units="TCB",
        t0=None,
        dt=None,
    ):
        """
        A class to simulate strain data for a continuous gravitational-wave
        signal after the data has been heterodyned, i.e., after multiplying
        the data by a complex phase vector. This uses the Equations 7 and 8
        from [1]_ accessed via the ``XLALHeterodynedPulsarGetModel()``
        function.

        Parameters
        ----------
        par: str, ``PulsarParametersPy``
            A TEMPO-style text file, or a PulsarParametersPy structure,
            containing the parameters for the source, in particular the phase
            parameters at which the data is "heterodyned".
        det: str
            The name of a gravitational-wave detector.
        times: array_like
            An array of GPS times at which to calculate the heterodyned strain.
        t0: float
            A time epoch in GPS seconds at which to calculate the detector
            response function. If not given and ``times`` is set, then the
            first value of ``times`` will be used.
        dt: int, float
            The time steps (in seconds) in the data over which to average the
            detector response. If not given and ``times`` is set, then the time
            difference between the first two values in ``times`` will be used.
        earth_ephem: str
            A file containing the LALSuite-style Earth ephemeris information.
            If not set then a default file will be used.
        sun_ephem: str
            A file containing the LALSuite-style Sun ephemeris information. If
            not set then a default file will be used.
        time_corr: str
            A file containing the LALSuite-style information on the time system
            corrections for, e.g., the TCB or TDB system. If not set then
            a default file will be used.
        ephem: str
            The solar system ephemeris system to use for the Earth and Sun
            ephemeris, i.e., ``"DE200"``, ``"DE405"``, ``"DE421"``, or
            ``"DE430"``. By default the ``EPHEM`` value from the supplied
            ``par`` will be used, but if not found, and if this value is not
            set, it will default to ``"DE405"``.
        units: str
            The time system used, i.e., ``"TDB"`` or ``"TCB"``. By default
            the ``UNITS`` value from the ``par`` will be used, but if not
            found, and if this value is not set, it will (like TEMPO2) default
            to ``"TCB"``.
        """

        self.__hetpar = self._read_par(par)
        self.detector = det
        self.times = times

        # mapping between time units and time correction file prefix
        self.__units_map = {"TCB": "te405", "TDB": "tdb"}

        self.ephem = ephem
        self.units = units

        # initialise the solar system ephemeris files
        self.__edat, self.__tdat = initialise_ephemeris(
            ephem=self.ephem,
            units=self.units,
            earthfile=earth_ephem,
            sunfile=sun_ephem,
            timefile=time_corr,
        )

        # set the "heterodyne" SSB time delay
        if self.times is not None:
            self.__hetSSBdelay = lalpulsar.HeterodynedPulsarGetSSBDelay(
                self.hetpar.PulsarParameters(),
                self.gpstimes,
                self.detector,
                self.__edat,
                self.__tdat,
                self.__units_type,
            )
        else:
            self.__hetSSBdelay = None

        # set the "heterodyne" BSB time delay
        if self.times is not None and self.hetpar["BINARY"] is not None:
            self.__hetBSBdelay = lalpulsar.HeterodynedPulsarGetBSBDelay(
                self.hetpar.PulsarParameters(),
                self.gpstimes,
                self.__hetSSBdelay,
                self.__edat,
            )
        else:
            self.__hetBSBdelay = None

        # set the "heterodyne" glitch phase
        if self.times is not None and self.hetpar["GLEP"] is not None:
            self.__hetglitchphase = lalpulsar.HeterodynedPulsarGetGlitchPhase(
                self.hetpar.PulsarParameters(),
                self.gpstimes,
                self.__hetSSBdelay,
                self.__hetBSBdelay,
            )
        else:
            self.__hetglitchphase = None

        # set the "heterodyne" FITWAVES phase
        if (
            self.times is not None
            and self.hetpar["WAVESIN"] is not None
            and self.hetpar["WAVECOS"] is not None
        ):
            self.__hetfitwavesphase = lalpulsar.HeterodynedPulsarGetFITWAVESPhase(
                self.hetpar.PulsarParameters(),
                self.gpstimes,
                self.__hetSSBdelay,
                self.hetpar["F0"],
            )
        else:
            self.__hetfitwavesphase = None

        # set the response function
        if self.times is None and t0 is None:
            raise ValueError(
                "Must supply either 'times' or 't0' to calculate "
                "the response function"
            )
        else:
            self.__t0 = float(t0) if t0 is not None else float(self.times[0])

        if dt is None and self.times is None:
            raise ValueError(
                "Must supply either 'times' or 'dt' to calculate "
                "the response function"
            )
        else:
            if self.times is not None and dt is None:
                if len(self.times) == 1:
                    raise ValueError("Must supply a 'dt' value")
                else:
                    self.__dt = float(self.times[1] - self.times[0])
            else:
                self.__dt = float(dt)

        ra = self.hetpar["RA"] if self.hetpar["RAJ"] is None else self.hetpar["RAJ"]
        dec = self.hetpar["DEC"] if self.hetpar["DECJ"] is None else self.hetpar["DECJ"]
        if ra is None or dec is None:
            raise ValueError("Right ascension and/or declination have not " "been set!")

        self.__resp = lalpulsar.DetResponseLookupTable(
            self.__t0, self.detector, ra, dec, 2880, self.__dt
        )

    @property
    def hetpar(self):
        return self.__hetpar

    @property
    def detector(self):
        return self.__detector

    @detector.setter
    def detector(self, det):
        if isinstance(det, lal.Detector):
            # value is already a lal.Detector
            self.__detector = det
        else:
            if not isinstance(det, str):
                raise TypeError("Detector name must be a string")
            else:
                try:
                    self.__detector = lalpulsar.GetSiteInfo(det)
                except RuntimeError:
                    raise ValueError(
                        "Detector '{}' was not a valid detector " "name.".format(det)
                    )

        self.__detector_name = self.__detector.frDetector.name

    @property
    def resp(self):
        """
        Return the response function look-up table.
        """

        return self.__resp

    @property
    def times(self):
        return self.__times

    @property
    def gpstimes(self):
        return self.__gpstimes

    @times.setter
    def times(self, times):
        """
        Set an array of times, and also a ``LIGOTimeGPSVector()`` containing
        the times.

        Parameters
        ----------
        times: array_like
            An array of GPS times. This can be a :class:`astropy.time.Time`
            object, for which inputs will be converted to GPS is not already
            held as GPS times.
        """

        from astropy.time import Time

        if times is None:
            self.__times = None
            self.__gpstimes = None
            return
        elif isinstance(times, lal.LIGOTimeGPS):
            self.__times = np.array(
                [times.gpsSeconds + 1e-9 * times.gpsNanoSeconds], dtype=np.float128
            )
            self.__gpstimes = lalpulsar.CreateTimestampVector(1)
            self.__gpstimes.data[0] = times
            return
        elif isinstance(times, lalpulsar.LIGOTimeGPSVector):
            self.__gpstimes = times
            self.__times = np.zeros(len(times.data), dtype=np.float128)
            for i, gpstime in enumerate(times.data):
                self.__times[i] = (
                    times.data[i].gpsSeconds + 1e-9 * times.data[i].gpsNanoSeconds
                )
            return
        elif isinstance(times, (int, float, np.float128, list, tuple, np.ndarray)):
            self.__times = np.atleast_1d(np.array(times, dtype=np.float128))
        elif isinstance(times, Time):
            self.__times = np.atleast_1d(times.gps).astype(np.float128)
        else:
            raise TypeError("Unknown data type for times")

        self.__gpstimes = lalpulsar.CreateTimestampVector(len(self.__times))
        for i, time in enumerate(self.__times):
            seconds = int(np.floor(time))
            nanoseconds = int((time - seconds) * 1e9)
            self.__gpstimes.data[i] = lal.LIGOTimeGPS(seconds, nanoseconds)

    @property
    def ephem(self):
        return self.__ephem

    @ephem.setter
    def ephem(self, ephem):
        """
        Set the heterodyne solar system ephemeris version. This will attempt to
        use the value set in the heterodyne source parameters, but otherwise
        defaults to ``DE405``.
        """

        if self.hetpar["EPHEM"] is not None:
            self.__ephem = self.hetpar["EPHEM"]
        else:
            self.__ephem = "DE405" if ephem is None else ephem

    @property
    def units(self):
        return self.__units

    @units.setter
    def units(self, units):
        """
        Set the time system units, i.e., either ``"TDB"`` or ``"TCB"``. This
        will attempt to use the value set in the heterodyne source parameters,
        but otherwise defaults to ``"TCB"``.
        """

        if self.hetpar["UNITS"] is not None:
            self.__units = self.hetpar["UNITS"]
        else:
            self.__units = "TCB" if units is None else units

        if self.__units not in ["TCB", "TDB"]:
            raise ValueError(
                "Unknown time system '{}' has been " "given.".format(self.__units)
            )

        if self.__units == "TCB":
            self.__units_type = lalpulsar.lalpulsar.TIMECORRECTION_TCB
        else:
            self.__units_type = lalpulsar.lalpulsar.TIMECORRECTION_TDB

    def model(
        self,
        newpar=None,
        updateSSB=False,
        updateBSB=False,
        updateglphase=False,
        updatefitwaves=False,
        freqfactor=2.0,
        usephase=False,
        roq=False,
    ):
        """
        Compute the heterodyned strain model using
        ``XLALHeterodynedPulsarGetModel()``.

        Parameters
        ----------
        newpar: str, ``PulsarParameterPy``
            A text parameter file, or ``PulsarParameterPy()`` object,
            containing a set of parameter at which to calculate the strain
            model. If this is ``None`` then the "heterodyne" parameters are used.
        updateSSB: bool
            Set to ``True`` to update the solar system barycentring time delays
            compared to those used in heterodyning, i.e., if the ``newpar``
            contains updated positional parameters.
        updateBSB: bool
            Set to ``True`` to update the binary system barycentring time
            delays compared to those used in heterodying, i.e., if the
            ``newpar`` contains updated binary system parameters
        updateglphase: bool
            Set to ``True`` to update the pulsar glitch evolution compared to
            that used in heterodyning, i.e., if the ``newpar`` contains updated
            glitch parameters.
        updatefitwaves: bool
            Set to ``True`` to update the pulsar FITWAVES phase evolution (used
            to model strong red timing noise) compared to that used in
            heterodyning.
        freqfactor: int, float
            The factor by which the frequency evolution is multiplied for the
            source model. This defaults to 2 for emission from the
            :math:`l=m=2` quadrupole mode.
        usephase: bool
            Set to ``True`` if the model is to include the phase evolution,
            i.e., if phase parameters are being updated, otherwise only two
            (six for non-GR sources) values giving the amplitides will be
            output.
        roq: bool
            A boolean value to set to ``True`` if requiring the output for
            a ROQ model (NOT YET IMPLEMENTED).

        Returns
        -------
        strain: array_like
            A complex array containing the strain data
        """

        if newpar is not None:
            parupdate = self._read_par(newpar)
        else:
            parupdate = self.hetpar

        origpar = self.hetpar

        self.__nonGR = self._check_nonGR(parupdate)
        compstrain = lalpulsar.HeterodynedPulsarGetModel(
            parupdate.PulsarParameters(),
            origpar.PulsarParameters(),
            freqfactor,
            int(usephase),  # phase is varying between par files
            int(roq),  # using ROQ?
            self.__nonGR,  # using non-tensorial modes?
            self.gpstimes,
            self.ssbdelay,
            int(updateSSB),  # the SSB delay should be updated compared to hetSSBdelay
            self.bsbdelay,
            int(updateBSB),  # the BSB delay should be updated compared to hetBSBdelay
            self.glitchphase,
            int(updateglphase),
            self.fitwavesphase,
            int(updatefitwaves),
            self.resp,
            self.__edat,
            self.__tdat,
            self.__units_type,
        )

        return compstrain.data.data

    def _read_par(self, par):
        """
        Read a TEMPO-style parameter file into a PulsarParameterPy object.
        """

        if isinstance(par, PulsarParametersPy):
            return par

        if isinstance(par, str):
            if not is_par_file(par):
                raise IOError("Could not read in parameter file: '{}'".format(par))
            else:
                raise PulsarParametersPy(par)
        else:
            raise TypeError("The parameter file must be a string")

    @property
    def ssbdelay(self):
        return self.__hetSSBdelay

    @property
    def bsbdelay(self):
        return self.__hetBSBdelay

    @property
    def glitchphase(self):
        return self.__hetglitchphase

    @property
    def fitwavesphase(self):
        return self.__hetfitwavesphase

    def _check_nonGR(self, par):
        """
        Check if the source parameters are for a non-GR model, i.e., are any of
        the amplitude/phase parameters for a non-GR model set
        """

        # non-GR amplitude parameters
        nonGRparams = [
            "HPLUS",
            "HCROSS",
            "HVECTORX",
            "HVECTORY",
            "HSCALARB",
            "HSCALARL",
            "HPLUS_F",
            "HCROSS_F",
            "HVECTORX_F",
            "HVECTORY_F",
            "HSCALARB_F",
            "HSCALARL_F",
        ]

        for param in nonGRparams:
            if param in par.keys():
                return 1

        return 0
