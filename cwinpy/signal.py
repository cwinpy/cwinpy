import math
from copy import deepcopy

import lal
import lalpulsar
import numpy as np
from packaging.version import Version

from .parfile import PulsarParameters
from .utils import (
    TEMPO2_GW_ALIASES,
    MuteStream,
    check_for_tempo2,
    get_psr_name,
    initialise_ephemeris,
    is_par_file,
)


class HeterodynedCWSimulator(object):
    def __init__(
        self,
        par=None,
        det=None,
        times=None,
        earth_ephem=None,
        sun_ephem=None,
        ephem="DE405",
        units="TCB",
        usetempo2=False,
        t0=None,
        dt=None,
        ref_freq=None,
        ref_epoch=None,
    ):
        """
        A class to simulate strain data for a continuous gravitational-wave
        signal after the data has been heterodyned, i.e., after multiplying
        the data by a complex phase vector. This uses the Equations 7 and 8
        from [1]_ accessed via the ``XLALHeterodynedPulsarGetModel()``
        function.

        Parameters
        ----------
        par: str, PulsarParameters
            A Tempo-style text file, or a
            :class:`~cwinpy.parfile.PulsarParameters` object, containing the
            parameters for the source, in particular the phase parameters at
            which the data is "heterodyned". This is required unless you are
            specifying a ``ref_freq`` for a static frame.
        det: str
            The name of a gravitational-wave detector at with to simulate the
            signal. This is required.
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
        usetempo2: bool
            Set to True to use TEMPO2, via libstempo, for calculating the phase
            of the signal. To use libstempo must be installed. If using TEMPO2
            the Earth, Sun and time ephemerides, and the ``ephem`` and
            ``units`` arguments are not required. Information on the correct
            ephemeris will all be calculated internally by TEMPO2 using the
            information from the pulsar parameter file.
        ref_freq: float
            Use this to set the source rotation frequency (Hz) if simulating an
            initial heterodyne at a fixed frequency without any barycentring.
            This will set the barycentring delays to zero, or, if using Tempo2,
            the "original" phase will be calculated at the SSB. This will be
            ignored if a ``par`` file is supplied.

            A reference epoch must either be supplied using the ``ref_epoch``
            argument, or it will be taken to be the first supplied time stamp.
        """

        self.usetempo2 = check_for_tempo2() if usetempo2 else False
        if usetempo2 and not self.usetempo2:
            raise ImportError(
                "TEMPO2 is not available, so the usetempo2 option cannot be used"
            )

        if det is None:
            raise ValueError("A detector must be supplied")

        self.detector = det
        self.times = times

        self.ref_freq = None
        if par is not None:
            self.__hetpar, self.__parfile = self._read_par(par)
        elif ref_freq is not None:
            # using reference frequency for original heterodyne
            self.ref_freq = ref_freq

            par = PulsarParameters()
            par["F"] = [ref_freq]
            par["RAJ"] = 0.0  # dummy RA
            par["DECJ"] = 0.0  # dummy dec

            if ref_epoch is not None:
                par["PEPOCH"] = ref_epoch
            elif self.times is not None:
                par["PEPOCH"] = self.times[0].astype(np.float64)
            else:
                raise ValueError(
                    "Times or a reference epoch must be specified to use reference frequency"
                )

            self.__hetpar, self.__parfile = self._read_par(par)
        else:
            raise ValueError("No par file or reference frequency specified")

        self.__hetSSBdelay = None
        self.__hetBSBdelay = None
        self.__hetglitchphase = None
        self.__hetfitwavesphase = None

        if not self.usetempo2:
            self.ephem = ephem
            self.units = units

            # initialise the solar system ephemeris files
            if isinstance(earth_ephem, dict):
                if self.ephem in earth_ephem:
                    earth_ephem_file = earth_ephem[self.ephem]
                else:
                    # fall back to looking up ephemeris
                    earth_ephem_file = None
            else:
                earth_ephem_file = earth_ephem

            if isinstance(sun_ephem, dict):
                if self.ephem in sun_ephem:
                    sun_ephem_file = sun_ephem[self.ephem]
                else:
                    # fall back to looking up ephemeris
                    sun_ephem_file = None
            else:
                sun_ephem_file = sun_ephem

            self.__edat, self.__tdat = initialise_ephemeris(
                ephem=self.ephem,
                units=self.units,
                earthfile=earth_ephem_file,
                sunfile=sun_ephem_file,
            )

            # set the "heterodyne" SSB time delay
            if self.times is not None:
                if self.ref_freq is None:
                    self.__hetSSBdelay = lalpulsar.HeterodynedPulsarGetSSBDelay(
                        self.hetpar.PulsarParameters(),
                        self.gpstimes,
                        self.detector,
                        self.__edat,
                        self.__tdat,
                        self.__units_type,
                    )
                else:
                    # set SSB delay to zero, i.e., a static frame
                    self.__hetSSBdelay = lal.CreateREAL8Vector(len(self.times))
                    self.__hetSSBdelay.data[:] = 0.0

            # set the "heterodyne" BSB time delay
            if self.times is not None and self.hetpar["BINARY"] is not None:
                if self.ref_freq is None:
                    self.__hetBSBdelay = lalpulsar.HeterodynedPulsarGetBSBDelay(
                        self.hetpar.PulsarParameters(),
                        self.gpstimes,
                        self.__hetSSBdelay,
                        self.__edat,
                    )
                else:
                    # set BSB delay to zero, i.e., a static frame
                    self.__hetBSBdelay = lal.CreateREAL8Vector(len(self.times))
                    self.__hetBSBdelay.data[:] = 0.0

            # set the "heterodyne" glitch phase
            if self.times is not None and self.hetpar["GLEP"] is not None:
                self.__hetglitchphase = lalpulsar.HeterodynedPulsarGetGlitchPhase(
                    self.hetpar.PulsarParameters(),
                    self.gpstimes,
                    self.__hetSSBdelay,
                    self.__hetBSBdelay,
                )

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
            raise ValueError("Right ascension and/or declination have not been set!")

        self.__resp = lalpulsar.DetResponseLookupTable(
            self.__t0, self.detector, ra, dec, 2880, self.__dt
        )

    @property
    def hetpar(self):
        return self.__hetpar

    @property
    def parfile(self):
        return self.__parfile

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
            for i in range(len(times.data)):
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
            self.__units_type = lalpulsar.TIMECORRECTION_TCB
        else:
            self.__units_type = lalpulsar.TIMECORRECTION_TDB

    def model(
        self,
        newpar=None,
        updateSSB=False,
        updateBSB=False,
        updateglphase=False,
        updatefitwaves=False,
        freqfactor=2.0,
        outputampcoeffs=False,
        roq=False,
        phase_only=False,
    ):
        """
        Compute the heterodyned strain model using
        ``XLALHeterodynedPulsarGetModel()``.

        Parameters
        ----------
        newpar: str, PulsarParameters
            A text parameter file, or :class:`~cwinpy.parfile.PulsarParameters`
            object, containing a set of parameter at which to calculate the
            strain model. If this is ``None`` then the "heterodyne" parameters
            are used.
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
        outputampcoeffs: bool
            If ``False`` (default) then the full complex time series of the
            signal (calculated at the time steps used when initialising the
            object) will be output. If ``True`` only two complex amplitudes
            (six for non-GR sources) will be output for use when a pre-summed
            signal model is being used - this should only be used if phase
            parameters are not being updated.
        roq: bool
            A boolean value to set to ``True`` if requiring the output for
            a ROQ model (NOT YET IMPLEMENTED).
        phase_only: bool
            Just return the phase evolution of the signal. If a ``newpar`` is
            supplied it will return the phase difference.

        Returns
        -------
        strain: array_like
            A complex array containing the strain data
        """

        if newpar is not None:
            parupdate, parfile = self._read_par(newpar)
        else:
            parupdate = deepcopy(self.hetpar)

        # get signal amplitude model
        self.__nonGR = self._check_nonGR(parupdate)

        if not phase_only:
            compstrain = lalpulsar.HeterodynedPulsarGetAmplitudeModel(
                parupdate.PulsarParameters(),
                freqfactor,
                int(not outputampcoeffs),
                int(roq),
                self.__nonGR,
                self.gpstimes,
                self.resp,
            )

        if (
            (not (outputampcoeffs or phase_only) and newpar is None)
            or roq
            or outputampcoeffs
        ):
            return compstrain.data.data
        else:
            from .heterodyne.fastheterodyne import fast_heterodyne

            origpar = None
            if not self.usetempo2:
                # use LAL function for phase calculation
                origpar = self.hetpar

                if newpar is not None:
                    if parupdate["PEPOCH"] != origpar["PEPOCH"] and Version(
                        lalpulsar.__version__
                    ) <= Version("7.1.0"):
                        # due to a bug in XLALHeterodynedPulsarPhaseDifference
                        # in LALpulsar versions <= 7.1.0, if the epochs
                        # mismatch the frequencies must be corrected to be at
                        # the same epoch here
                        deltat = origpar["PEPOCH"] - parupdate["PEPOCH"]
                        nfreqs = len(parupdate["F"])

                        # update frequencies
                        updated_freqs = np.copy(parupdate["F"])
                        for i in range(nfreqs - 1):
                            deltatpowu = deltat
                            for j in range(i + 1, nfreqs):
                                updated_freqs[i] += (
                                    updated_freqs[j] * deltatpowu
                                ) / math.factorial(j - i)
                                deltatpowu *= deltat

                        parupdate["F"] = updated_freqs
                        parupdate["PEPOCH"] = origpar["PEPOCH"]

                if updateSSB:
                    # if updating SSB other delays *must* also be updated if
                    # required
                    updateBSB = True if parupdate["BINARY"] is not None else updateBSB
                    updateglphase = (
                        True if parupdate["GLEP"] is not None else updateglphase
                    )
                    updatefitwaves = (
                        True if parupdate["WAVEEPOCH"] is not None else updatefitwaves
                    )
                elif updateBSB:
                    # if updating BSB the glitch phase must be updated if required
                    updateglphase = (
                        True if parupdate["GLEP"] is not None else updateglphase
                    )

                phase = lalpulsar.HeterodynedPulsarPhaseDifference(
                    parupdate.PulsarParameters()
                    if newpar is not None
                    else origpar.PulsarParameters(),
                    origpar.PulsarParameters() if newpar is not None else None,
                    self.gpstimes,
                    freqfactor,
                    self.ssbdelay,
                    int(
                        updateSSB
                    ),  # the SSB delay should be updated compared to hetSSBdelay
                    self.bsbdelay,
                    int(
                        updateBSB
                    ),  # the BSB delay should be updated compared to hetBSBdelay
                    self.glitchphase,
                    int(updateglphase),
                    self.fitwavesphase,
                    int(updatefitwaves),
                    self.resp.det,
                    self.__edat,
                    self.__tdat,
                    self.__units_type,
                )

                self._phasediff = -phase.data.astype(float)

            if self.usetempo2:
                # use TEMPO2 for phase calculation
                import sys

                from astropy.time import Time
                from libstempo import tempopulsar

                # maximum number of obsverations that can be used at once (see tempo2.h)
                MAX_OBS_VAL = 20000

                # convert times to MJD
                mjdtimes = Time(self.times, format="gps", scale="utc").mjd

                toaerr = 1e-15  # add tiny error value to stop errors
                psrorig = None

                phaseorig = np.zeros_like(self.times)
                phasenew = np.zeros_like(self.times)
                phasenum0 = None

                # calculate MAX_OBS_VAL chunks at a time, so as not to run out
                # of memory
                for i in range(1 + len(self.times) // MAX_OBS_VAL):
                    sidx = i * MAX_OBS_VAL
                    eidx = min(len(self.times), sidx + MAX_OBS_VAL)

                    if self.ref_freq is None:
                        with MuteStream(stream=sys.stdout):
                            psrorig = tempopulsar(
                                parfile=self.parfile,
                                toas=mjdtimes[sidx:eidx],
                                toaerrs=toaerr,
                                observatory=TEMPO2_GW_ALIASES[self.__detector_name],
                                dofit=False,
                                obsfreq=0.0,  # set to 0 so as not to calculate DM delay
                            )

                            # get phase residuals
                            # NOTE: referencing this to a site and epoch may not be
                            # necessary, but we'll do it as a precaution
                            phaseorig[sidx:eidx] = psrorig.phaseresiduals(
                                removemean="refphs",
                                site="@",
                                epoch=psrorig["PEPOCH"].val,
                            )

                            if phase_only and newpar is None:
                                # get the correction for the phase epoch
                                phasenum = psrorig.pulsenumbers(
                                    updatebats=False,
                                    formresiduals=False,
                                    removemean=False,
                                )

                                if phasenum0 is None:
                                    # get phase number from first loop
                                    phasenum0 = phasenum[0]

                                phaseorig[sidx:eidx] += phasenum - phasenum0
                    else:
                        # calculate the phase for the reference frequency
                        phaseorig[sidx:eidx] = self.ref_freq * (
                            self.times[sidx:eidx] - self.hetpar["PEPOCH"]
                        )

                    if newpar is not None:
                        with MuteStream(stream=sys.stdout):
                            psrnew = tempopulsar(
                                parfile=parfile,
                                toas=mjdtimes[sidx:eidx],
                                toaerrs=toaerr,
                                observatory=TEMPO2_GW_ALIASES[self.__detector_name],
                                dofit=False,
                                obsfreq=0.0,  # set to 0 so as not to calculate DM delay
                            )

                            # get phase residuals
                            phasenew[sidx:eidx] = psrnew.phaseresiduals(
                                removemean="refphs",
                                site="@",
                                epoch=psrorig["PEPOCH"].val
                                if psrorig is not None
                                else self.hetpar.convert_to_tempo_units(
                                    "PEPOCH", self.hetpar["PEPOCH"]
                                ).value,
                            )

                # get phase difference
                self._phasediff = freqfactor * (phaseorig - phasenew).astype(float)

            if phase_only:
                return self.phasediff
            else:
                # re-heterodyne with phase difference
                return fast_heterodyne(compstrain.data.data, self.phasediff)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _read_par(self, par):
        """
        Read a TEMPO-style parameter file into a
        :class:`~cwinpy.parfile.PulsarParameters` object.
        """

        if isinstance(par, PulsarParameters):
            if not self.usetempo2:
                return par, None
            else:
                # convert PulsarParameters to a file and return
                import tempfile

                parfile = tempfile.mkstemp(suffix=".par", prefix=get_psr_name(par))
                par.pp_to_par(parfile[1])
                return par, parfile[1]

        if isinstance(par, str):
            if not is_par_file(par):
                raise IOError("Could not read in parameter file: '{}'".format(par))
            else:
                return PulsarParameters(par), par
        else:
            raise TypeError("The parameter file must be a string")

    def phase_evolution(
        self,
        newpar=None,
        freqfactor=2.0,
        **kwargs,
    ):
        """
        Return the phase evolution of the signal. If ``newpar`` is supplied
        then the phase difference will be output, otherwise phase evolution of
        the original input par file will be output. See
        :meth:`~cwinpy.signal.HeterodynedCWSimulator.model` for input
        arguments. If you created the
        :class:`~cwinpy.signal.HeterodynedCWSimulator` using a reference
        frequency and you are supplying a ``newpar``,the "update" keyword
        arguments will all default to ``True``, but otherwise default to
        ``False``.
        """

        ref_freq_used = newpar is not None and self.ref_freq is not None

        return self(
            newpar=newpar,
            updateSSB=kwargs.get("updateSSB", True if ref_freq_used else False),
            updateBSB=kwargs.get("updateBSB", True if ref_freq_used else False),
            updateglphase=kwargs.get("updateglphase", True if ref_freq_used else False),
            updatefitwaves=kwargs.get(
                "updatefitwaves", True if ref_freq_used else False
            ),
            freqfactor=freqfactor,
            outputampcoeffs=False,
            roq=False,
            phase_only=True,
        )

    @property
    def phasediff(self):
        return getattr(self, "_phasediff", np.zeros(len(self.times), dtype=float))

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
        the amplitude/phase parameters for a non-GR model set. If there is a
        `H0` parameter, then return 0 for a GR model.
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

        if "H0" in par.keys():
            # it is a GR model if it contains H0
            return 0

        for param in nonGRparams:
            if param in par.keys():
                return 1

        return 0
