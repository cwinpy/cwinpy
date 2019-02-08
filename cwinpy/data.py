"""
Functions to deal with heterodyned data.
"""

from __future__ import division, print_function

import numpy as np
import warnings
from collections import OrderedDict
from scipy.special import gammaln

# import lal and lalpulsar
import lal
import lalpulsar


def logfactorial(n):
    """
    The natural logarithm of the factorial of an integer using the fact that

    .. math::

        \ln{(n!)} = \ln{\left(\Gamma (n+1)\\right)}

    Parameters
    ----------
    n: int
        An integer for which the natural logarithm of its factorial is
        required.

    Returns
    -------
    float
    """

    if isinstance(n, int):
        if n >= 0:
            return gammaln(n+1)
        else:
            raise ValueError("Can't find the factorial of a negative number")
    else:
        raise ValueError("Can't find the factorial of a non-integer value")


class MultiHeterodynedData(object):
    """
    A class to contain time series' of heterodyned data, using the
    :class:`~cwinpy.HeterodynedData` class, for multiple detectors/data
    streams.

    Parameters
    ----------
    data: (str, array_like, dict, HeterodynedData)
        The heterodyned data either as a string giving a file path, an array of
        data, or a dictionary of file paths/data arrays, that are keyed on
        valid detector names.
    times: (array_like, dict)
        If `data` is an array, or dictionary of arrays, then `times` must be
        set giving the time stamps for the data values. If `times` is a
        dictionary then it should be keyed on the same detector names as in
        `data`.
    detector: (str, lal.Detector)
        If `data` is a file name or data array then `detector` must be given as
        a string or :class:`lal.Detector`.

    Notes
    -----

    See the :class:`~cwinpy.HeterodynedData` documentation for information on
    additional keyword arguments.
    """

    def __init__(self, data=None, times=None, detector=None, window=30,
                 inject=False, par=None, injpar=None, freqfactor=2.0,
                 bbthreshold="default", **kwargs):

        # set keyword argument
        self.__heterodyned_data_kwargs = {}
        self.__heterodyned_data_kwargs['window'] = window
        self.__heterodyned_data_kwargs['par'] = par
        self.__heterodyned_data_kwargs['injpar'] = injpar
        self.__heterodyned_data_kwargs['inject'] = inject
        self.__heterodyned_data_kwargs['freqfactor'] = freqfactor
        self.__heterodyned_data_kwargs['bbthreshold'] = bbthreshold

        self.__data = OrderedDict()  # initialise empty dict
        self.__currentidx = 0  # index for iterator

        # add data
        if data is not None:
            self.add_data(data, times, detector=detector)

    def add_data(self, data, times, detector=None):
        """
        Add heterodyned data to the class.

        Parameters
        ----------
        data: (str, array_like, dict, :class:`~cwinpy.HeterodynedData`)
            The heterodyned data either as a string giving a file path, an
            array of data, or a dictionary of file paths/data arrays, that are
            keyed on valid detector names.
        times: (array_like, dict)
            If `data` is an array, or dictionary of arrays, then `times` must
            be set giving the time stamps for the data values. If `times` is
            a dictionary then it should be keyed on the same detector names as
            in `data`.
        detector: (str, lal.Detector)
            If `data` is a file name or data array then `detector` must be
            given as a string or :class:`lal.Detector`.
        """

        if isinstance(data, HeterodynedData):
            if data.detector is None and detector is None:
                raise ValueError("No detector is given!")

            if data.detector is None and detector is not None:
                data.detector = detector

            self._add_HeterodynedData(data)
        elif isinstance(data, dict):
            for detkey in data:
                if isinstance(data[detkey], HeterodynedData):
                    if data[detkey].detector is None:
                        data[detkey].detector = detkey
                    self._add_HeterodynedData(data[detkey])
                else:
                    if isinstance(times, dict):
                        if detkey not in times:
                            raise KeyError("`times` does not contain the "
                                           "detector: {}".format(detkey))
                        else:
                            dettimes = times[detkey]
                    else:
                        dettimes = times

                    self._add_data(data[detkey], detkey, dettimes)
        else:
            if isinstance(times, dict):
                raise TypeError('`times` should not be a dictionary')

            self._add_data(data, detector, times)

    def _add_HeterodynedData(self, data):
        detname = data.detector
        if detname not in self.__data:
            self.__data[detname] = [data]  # add as a list
        else:
            # if data from that detector already exists then append to the list
            self.__data[detname].append(data)

    def _add_data(self, data, detector, times=None):
        if detector is None or data is None:
            raise ValueError("data and detector must be set")

        het = HeterodynedData(data, times, detector=detector,
                              **self.__heterodyned_data_kwargs)

        self._add_HeterodynedData(het)

    def __getitem__(self, det):
        """
        Get the list of :class:`~cwinpy.HeterodynedData` objects keyed to a
        given detector.
        """

        if det in self.detectors:
            return self.__data[det]
        else:
            return None

    @property
    def to_list(self):
        datalist = []
        for key in self.__data:
            if isinstance(self.__data[key], list):
                datalist += self.__data[key]
            else:
                datalist.append(self.__data[key])

        return datalist

    @property
    def detectors(self):
        # return the list of detectors contained in the object
        return list(self.__data.keys())

    @property
    def laldetector(self):
        # return the list of :class:`lal.Detector` contained in the object
        return [det.laldetector for det in self.__data.values()]

    def __iter__(self):
        return self

    def __next__(self):
        if self.__currentidx >= len(self):
            self.__currentidx = 0  # reset iterator index
            raise StopIteration
        else:
            self.__currentidx += 1
            return self.to_list[self.__currentidx-1]

    def __len__(self):
        length = 0
        for key in self.__data:
            if isinstance(self.__data[key], list):
                length += len(self.__data[key])
            else:
                length += 1
        return length


class HeterodynedData(object):
    """
    A class to contain a time series of heterodyned data.

    Some examples of input `data` are:

    1. The path to a file containing (gzipped) ascii text with the
    following three columns::

        # GPS time stamps   real strain   imaginary strain
        1000000000.0         2.3852e-25    3.4652e-26
        1000000060.0        -1.2963e-26    9.7423e-25
        1000000120.0         5.4852e-25   -1.8964e-25
        ...

    or four columns::

        # GPS time stamps   real strain   imaginary strain   std. dev.
        1000000000.0         2.3852e-25    3.4652e-26        1.0e-25
        1000000060.0        -1.2963e-26    9.7423e-25        1.0e-25
        1000000120.0         5.4852e-25   -1.8964e-25        1.0e-25
        ...

    where any row that starts with a ``#`` or a ``%`` is considered a comment.

    2. A 1-dimensional array of complex data, and accompanying array of `time`
    values, e.g.,

    >>> import numpy as np
    >>> N = 100  # the data length
    >>> data = np.random.randn(N) + 1j*np.random.randn(N)
    >>> times = np.linspace(1000000000., 1000005940., N)

    or, a 2-dimensional array with the real and complex values held in separate
    columns, e.g.,

    >>> import numpy as np
    >>> N = 100  # the data length
    >>> data = np.random.randn(N, 2)
    >>> times = np.linspace(1000000000., 1000005940., N)

    or, a 2-dimensional array with the real and complex values held in separate
    columns, *and* a third column holding the standard deviation for each
    entry, e.g.,

    >>> import numpy as np
    >>> N = 100  # the data length
    >>> stds = np.ones(N)  # standard deviations
    >>> data = np.array([stds*np.random.randn(N),
    >>> ...              stds*np.random.randn(N), stds]).T
    >>> times = np.linspace(1000000000., 1000005940., N)

    Parameters
    ----------
    data: (str, array_like)
        A file (plain ascii text, or gzipped ascii text) containing a time
        series of heterodyned data, or an array containing the complex
        heterodyned data.
    times: array_like
        If the data was passed using the `data` argument, then the associated
        time stamps should be passed using this argument.
    par: (str, lalpulsar.PulsarParametersPy)
        A parameter file, or :class:`lalpulsar.PulsarParametersPy` object
        containing the parameters with which the data was heterodyned.
    detector: (str, lal.Detector)
        A string, or lal.Detector object, identifying the detector from which
        the data was generated.
    window: int, 30
        The length of a window used for calculating a running median over the
        data.
    inject: bool, False
        Set to ``True`` to add a simulated signal to the data based on the
        parameters supplied in `injpar`, or `par` if `injpar` is not given.
    injpar: (str, lalpulsar.PulsarParametersPy)
        A parameter file name or :class:`lalpulsar.PulsarParametersPy`
        object containing values for the injected signal. A `par` file must
        also have been provided, and the injected signal will assume that
        the data has already been heterodyned using the parameters from
        `par`, which could be different.
    injtimes: list, None
        A list containing pairs of times between which to add the simulated
        signal. By default the signal will be added into the whole data set.
    freqfactor: float, 2.0
        The frequency scale factor for the data signal, e.g., a value of two
        for emission from the l=m=2 mode at twice the rotation frequency of the
        source.
    fakeasd: (float, str)
        A amplitude spectral density value (in 1/sqrt(Hz)) at which to
        generate simulated Gaussian noise to add to the data. Alternatively, if
        a string is passed, and that string represents a known detector, then
        the amplitude spectral density for that detector at design sensitivity
        will be used (this requires a `par` value to be included, which
        contains the source rotation frequency).
    bbthreshold: (str, float), "default"
        The threshold method, or value for the
        :meth:`~cwinpy.HeterodynedData.bayesian_blocks` function.

        """

    def __init__(self, data=None, times=None, par=None, detector=None,
                 window=30, inject=False, injpar=None, injtimes=None,
                 freqfactor=2.0, fakeasd=None, bbthreshold="default"):
        self.window = window  # set the window size

        # set the data
        self.data = (data, times)

        # set the parameter file
        self.par = par

        # set the detector from which the data came
        self.detector = detector

        # set the frequency scale factor
        self.freq_factor = freqfactor

        # add noise, or create data containing noise
        if fakeasd is not None:
            self.add_noise(fakeasd)

        # set and add a simulated signal
        self.injection = bool(inject)
        if self.injection:
            # inject the signal
            if injpar is None:
                self.inject_signal(injtimes=injtimes)
            else:
                self.inject_signal(injpar=injpar, injtimes=injtimes)

    @property
    def window(self):
        """The running median window length."""

        return self.__window

    @window.setter
    def window(self, window):
        if isinstance(window, int):
            if window < 2:
                raise ValueError("Window length must be greater than 2")
            else:
                self.__window = window
        else:
            raise TypeError("Window must be an integer")

    @property
    def data(self):
        """
        A :class:`numpy.ndarray` containing the heterodyned data.
        """

        return self.__data

    @data.setter
    def data(self, data):
        if isinstance(data, tuple):
            try:
                dataval, times = data
            except ValueError:
                raise ValueError("Tuple of data must have two items")
        else:
            dataval = data
            times = None

        if isinstance(dataval, str):
            # read in data from a file
            try:
                dataarray = np.loadtxt(dataval, comments=['#', '%'])
            except Exception as e:
                raise IOError("Problem reading in data: {}".format(e))

            if len(dataarray.shape) != 2:
                raise ValueError("Data array is the wrong shape for "
                                 "heterodyned data.")

            if dataarray.shape[1] != 3 and dataarray.shape[1] != 4:
                raise ValueError("Data array is the wrong shape")

            self.times = dataarray[:, 0]  # set time stamps
        else:
            if times is None:
                raise ValueError("Time stamps must also be supplied")
            else:
                # use supplied time stamps
                self.times = times

            if dataval is None:
                # set data to zeros
                dataarray = np.zeros((len(times), 1), dtype=np.complex)
            else:
                dataarray = np.atleast_2d(np.asarray(dataval))
                if dataarray.shape[0] == 1:
                    dataarray = dataarray.T

        self.__stds = None  # initialise stds to None
        if dataarray.shape[1] == 1 and dataarray.dtype == np.complex:
            self.__data = dataarray.flatten()
        elif dataarray.shape[1] == 2:
            # real and imaginary components are separate
            self.__data = dataarray[:, 0] + 1j*dataarray[:, 1]
        elif dataarray.shape[1] == 3 or dataarray.shape[1] == 4:
            self.__data = dataarray[:, 1] + 1j*dataarray[:, 2]
            if dataarray.shape[1] == 4:
                # set pre-calculated data standard deviations
                self.__stds = dataarray[:, 3]
        else:
            raise ValueError("Data array is the wrong shape")

        if len(self.times) != len(self.data):
            raise ValueError("Data and time stamps are not the same length")

        # set the (minimum) time step and sampling frequency
        if len(self.times) > 1:
            self.__dt = np.min(np.diff(self.times))
            self.__fs = 1./self.dt
        else:
            warnings.warn("Your data is only one data point long!")
            self.__dt = None
            self.__fs = None

        # initialise the running median
        _ = self.compute_running_median(N=self.window)

        # initialise change points to None
        self.__change_point_indices_and_ratios = None
 
        # calculate change points (and variances)
        self.bayesian_blocks()

    @property
    def times(self):
        return self.__times

    @times.setter
    def times(self, times):
        """
        Set the data time stamps.
        """

        self.__times = np.asarray(times, dtype='float64')

    @property
    def dt(self):
        """
        The (minimum) time step between data points.
        """

        return self.__dt

    @property
    def fs(self):
        """
        The sampling frequency (assuming even sampling)
        """

        return self.__fs

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, par):
        self.__par = self._parse_par(par)

    @property
    def injpar(self):
        return self.__injpar

    @injpar.setter
    def injpar(self, par):
        self.__injpar = self._parse_par(par)

    def _parse_par(self, par):
        """
        Parse a pulsar parameter file or :class:`lalpulsar.PulsarParametersPy`
        object.

        Parameters
        ----------
        par: (str, lalpulsar.PulsarParametersPy)
            A file or object containing a set of pulsar parameters.

        Returns
        -------
        lalpulsar.PulsarParametersPy
        """

        if par is not None:
            from lalpulsar.PulsarParametersWrapper import PulsarParametersPy

            if isinstance(par, PulsarParametersPy):
                return par
            elif isinstance(par, str):
                try:
                    newpar = PulsarParametersPy(par)
                except Exception as e:
                    raise IOError("Could not read in pulsar parameter "
                                  "file: {}".format(e))
            else:
                raise TypeError("'par' is not a recognised type")
        else:
            newpar = None

        return newpar

    @property
    def detector(self):
        """The name of the detector from which the data came."""
        
        return self.__detector

    @property
    def laldetector(self):
        """
        The :class:`lal.Detector` containing the detector's response and
        location.
        """

        return self.__laldetector

    @detector.setter
    def detector(self, detector):
        if detector is None:
            self.__detector = None
            self.__laldetector = None
        else:
            if isinstance(detector, lal.Detector):
                self.__detector = detector.frDetector.prefix
                self.__laldetector = detector
            elif isinstance(detector, str):
                self.__detector = detector

                try:
                    self.__laldetector = lalpulsar.GetSiteInfo(detector)
                except RuntimeError:
                    raise ValueError("Could not set LAL detector!")

    @property
    def running_median(self):
        """A :class:`~numpy.ndarray` containing the running median of the data."""

        return self.__running_median

    def compute_running_median(self, N=30):
        """
        Calculate a running median from the data with the real and imaginary
        parts separately. The running median will be calculated using a window
        of samples of a given number. This does not account for any gaps in the
        data, so could contain discontinuities.

        Parameters
        ----------
        N: int, 30
            The window length of the running median. Defaults to 30 points.

        Returns
        -------
        array_like
            A :class:`numpy.ndarray` array containing the data with the
            running median subtracted.
        """

        if N < 2:
            raise ValueError("The running median window must be greater "
                             "than 1")

        self.__running_median = np.zeros(len(self), dtype=np.complex)
        for i in range(len(self)):
            if i < N//2:
                startidx = 0
                endidx = i + (N//2) + 1
            elif i > len(self) - N:
                startidx =  i - (N//2) + 1
                endix = len(self)
            else:
                startidx = i - (N//2) + 1
                endidx = i + (N//2) + 1
 
            self.__running_median.real[i] = np.median(self.data.real[startidx:endidx])
            self.__running_median.imag[i] = np.median(self.data.imag[startidx:endidx])

        return self.running_median

    def subtract_running_median(self):
        """
        Subtract the running median from the data.

        Returns
        -------
        array_like
            A :class:`~numpy.ndarray` array containing the data with with
            running median subtracted.
        """

        return self.data - self.running_median

    @property
    def vars(self):
        """
        The variances of the data points.
        """

        return self.__vars

    @property
    def stds(self):
        """
        The standard deviations of the data points.
        """

        return np.sqrt(self.__vars)

    def compute_variance(self, change_points=None, N=30):
        """
        Compute the (sample) variance of the data within a set of change
        points. The variance will be calculated after subtraction of a running
        median. As the data is complex, we calculate the variance of a vector
        in which the real and imaginary components are concatenated. This is
        equivalent to a two-sided power spectral density.

        Parameters
        ----------
        change_points: array_like, None
            An array of indices of statistical change points within the data
        N: int, 30
            The window size (in terms of data point number) of the running
            median.

        Returns
        -------
        array_like
            A :class:`numpy.ndarray` of variances for each data point.
        """

        if self.__stds is not None:
            self.__vars = self.__stds**2
            return self.vars
        else:
            self.__vars = np.zeros(len(self))

        # subtract running median from the data
        datasub = self.subtract_running_median()

        if (change_points is None and
            self.__change_point_indices_and_ratios is None):
            # return the (sample) variance (hence 'ddof=1')
            self.__vars = np.full(len(self),
                                  np.hstack((datasub.real,
                                             datasub.imag)).var(ddof=1))

        else:
            if change_points is not None:
                cps = np.concatenate(([0], np.asarray(change_points),
                                      [len(datasub)])).astype('int')
            else:
                cps = np.concatenate(([0], self.change_point_indices,
                                      [len(datasub)])).astype('int')

            if self.stds is None:
                self.stds = np.zeros(len(self))

            for i in range(len(cps)-1):
                if cps[i+1] < 1 or cps[i+1] > len(datasub):
                    raise ValueError("Change point index is out of bounds")

                if cps[i+1] <= cps[i]:
                    raise ValueError("Change point order is wrong")

                datachunk = datasub[cps[i]:cps[i+1]]

                # get (sample) variance of chunk
                self.__vars[cps[i]:cps[i+1]] = np.hstack((datachunk.real,
                                                          datachunk.imag)).var(ddof=1)

        return self.vars

    def inject_signal(self, injpar=None, injtimes=None, freqfactor=2.):
        """
        Inject a simulated signal into the data.

        Parameters
        ----------
        injpar: (str, lalpulsar.PulsarParametersPy)
            A parameter file or object containing the parameters for the
            simulated signal.
        injtimes: list
            A list of pairs of time values between which to inject the signal.
        freqfactor: float, 2.0
            A the frequency scaling for the signal model, i.e., "2.0" for
            emission from the l=m=2 mass quadrupole mode.
        """

        from lalpulsar.simulateHeterodynedCW import HeterodynedCWSimulator

        if self.par is None:
            raise ValueError("To perform an injection a parameter file "
                             "must be supplied")

        # set the times between which the injection will be added
        self.injtimes = injtimes

        # initialise the injection
        het = HeterodynedCWSimulator(self.par, self.detector, times=self.times)

        if freqfactor != self.freq_factor:
            self.freq_factor = freqfactor

        # initialise the injection to zero
        inj_data = np.ones_like(self.data)

        # get the injection
        if injpar is None:
            # use self.par for the injection parameters
            self.injpar = self.par
            inj = het.model(usephase=True, freqfactor=self.freq_factor)
        else:
            self.injpar = injpar
            inj = het.model(self.injpar, updateSSB=True, updateBSB=True,
                            usephase=True, freqfactor=self.freq_factor)

        for timerange in self.injtimes:
            timeidxs = ((self.times >= timerange[0]) &
                        (self.times <= timerange[1]))
            inj_data[timeidxs] = inj[timeidxs]

        # add injection to data
        self.__data = self.data + inj_data

        # save injection data
        self.__inj_data = inj_data

    @property
    def injtimes(self):
        """
        A list of times at which an injection was added to the data. 
        """

        return self.__injtimes

    @injtimes.setter
    def injtimes(self, injtimes):
        if injtimes is None:
            # include all time
            timelist = np.array([[self.times[0], self.times[-1]]])

        try:
            timelist = np.atleast_2d(injtimes)
        except Exception as e:
            raise ValueError("Could not parse list of injection "
                             "times: {}".format(e))

        for timerange in timelist:
            if timerange[0] >= timerange[1]:
                raise ValueError("Injection time ranges are incorrect")

        self.__injtimes = timelist

    @property
    def injection_data(self):
        """
        The pure simulated signal that was added to the data.
        """

        return self.__inj_data

    @property
    def injection_optimal_snr(self):
        """
        Return the optimal signal-to-noise ratio using the pure injected signal
        and true noise calculated using:

        .. math::

           \\rho = \sqrt{\sum_i \left(\left[\\frac{\Re{(s_i)}}{\Re{(d_i)}}\\right]^2 + \left[\\frac{\Im{(s_i)}}{\Im{(d_i)}}\\right]^2\\right)}

        where :math:`d` is the signal-free data, and :math:`s` is the pure
        signal.

        """

        if not self.injection:
            return None

        noinj = self.data - self.injection_data  # data with injection removed

        return np.sqrt(((self.injection_data.real/noinj.real)**2).sum() +
                       ((self.injection_data.imag/noinj.imag)**2).sum())

    @property
    def freq_factor(self):
        """
        The scale factor of the source rotation frequency with which the data
        was heterodyned.
        """

        return self.__freq_factor

    @freq_factor.setter
    def freq_factor(self, freqfactor):
        if not isinstance(freqfactor, (float, int)):
            raise TypeError("Frequency scale factor must be a number")

        if freqfactor <= 0.:
            raise ValueError("Frequency scale factor must be a positive "
                             "number")

        self.__freq_factor = float(freqfactor)

    def add_noise(self, asd, issigma=False):
        """
        Add white Gaussian noise to the data based on a supplied one-sided
        noise amplitude spectral density (in 1/sqrt(Hz)).

        Parameters
        ----------
        asd: (float, str)
            The noise amplitude spectral density (1/sqrt(Hz)) at which to
            generate the Gaussian noise, or a string containing a valid
            detector name for which the design sensitivity ASD can be used.
        issigma: bool, False
            If `issigma` is ``True`` then the value passed to `asd` is assumed
            to be a dimensionless time domain standard deviation for the noise
            level rather than an amplitude spectral density.
        """

        if isinstance(asd, str):
            import lalsimulation as lalsim

            aliases = {'AV': ['Virgo', 'V1', 'AdV', 'AdvancedVirgo', 'AV'],
                       'AL': ['H1', 'L1', 'LHO', 'LLO', 'aLIGO', 'AdvancedLIGO', 'AL'],
                       'IL': ['iH1', 'iL1', 'InitialLIGO', 'IL'],
                       'IV': ['iV1', 'InitialVirgo', 'IV'],
                       'G1': ['G1', 'GEO', 'GEOHF'],
                       'IG': ['IG', 'GEO600', 'InitialGEO'],
                       'T1': ['T1', 'TAMA', 'TAMA300'],
                       'K1': ['K1', 'KAGRA', 'LCGT']}

            # set mapping of detector names to lalsimulation PSD functions
            simmap = {'AV': lalsim.SimNoisePSDAdvVirgo,  # advanced Virgo
                      'AL': lalsim.SimNoisePSDaLIGOZeroDetHighPower,  # aLIGO
                      'IL': lalsim.SimNoisePSDiLIGOSRD,               # iLIGO
                      'IV': lalsim.SimNoisePSDVirgo,                  # iVirgo
                      'IG': lalsim.SimNoisePSDGEO,                    # GEO600
                      'G1': lalsim.SimNoisePSDGEOHf,                  # GEOHF
                      'T1': lalsim.SimNoisePSDTAMA,                   # TAMA
                      'K1': lalsim.SimNoisePSDKAGRA}                  # KAGRA

            # check if string is valid
            detalias = None
            for dkey in aliases:
                if asd.upper() in aliases[dkey]:
                    detalias = dkey

            if detalias is None:
                raise ValueError("Detector '{}' is not as known detector "
                                 "alias".format(asd))

            freqs = self.par['F']

            if freqs is None:
                raise ValueError("Heterodyne parameter file contains no "
                                 "frequency value")

            # set amplitude spectral density value
            asdval = np.sqrt(simmap[detalias](self.freq_factor * freqs[0]))

            # convert to time domain standard deviation
            if self.dt is None:
                raise ValueError("No time step present. Does your data only "
                                 "consist of one value?")

            sigmaval = 0.5*asdval/np.sqrt(self.dt)
        elif isinstance(asd, float):
            if issigma:
                sigmaval = asd
            else:
                if self.dt is None:
                    raise ValueError("No time step present. Does your data "
                                     "only consist of one value?")

                sigmaval = 0.5*asd/np.sqrt(self.dt)
        else:
            raise TypeError("ASD must be a float or a string with a detector "
                            "name.")

        # get noise for real and imaginary components
        noise = np.random.normal(loc=0., scale=sigmaval,
                                 size=(len(self.data), 2))

        # add the noise to the data
        self.__data.real += noise[:, 0]
        self.__data.imag += noise[:, 1]

    def bayesian_blocks(self, threshold='default', minlength=5,
                        maxlength=np.inf):
        """
        Apply a Bayesian-Block-style algorithm to cut the data (after
        subtraction of a running median) up into chunks with different
        statistical properties using the formalism described in Section 2.4 of
        [1]_. Within each chunk the data should be well described by a single
        Gaussian distribution with zero mean.

        Splitting of the data relies on a threshold on the natural logarithm of
        the odds comparing the hypothesis that the data is best described by
        two different contiguous zero mean Gaussian distributions with
        different unknown variances to the hypothesis that the data is
        described by a single zero mean Gaussian with unknown variance. The
        former hypothesis is a compound hypothesis consisting of the sum of
        evidences for the split in the data at any point.

        The ``'default'`` threshold for splitting is empirically derived in
        [1]_ for the cases that the prior odds between the two hypotheses is
        equal, and has a 1% false alarm probability for splitting data that is
        actually drawn from a single zero mean Gaussian. The ``'trials'``
        threshold comes from assigning equal priors to the single Gaussian
        hypothesis and the full compound hypotheses that there is a split
        (in the ``'default'`` threshold it implicitly assume the single
        Gaussian hypothesis and *each* numerator sub-hypothesis have equal
        prior probability). This is essentially like a trials factor.
        Alternatively, the `threshold` value can be any real number.

        Parameters
        ----------
        threshold: (str, float)
            A string giving the method for determining the threshold for
            splitting the data (described above), or a value of the threshold.
        minlength: int
            The minimum length that a chunk can be split into. Defaults to 5.
        maxlength: int
            The maximum length that a chunk can be split into. Defaults to inf.

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1 <https:arxiv.org/abs/1705.08978v1>`_,
           2017.
        """

        if not isinstance(minlength, int):
            raise ValueError("Minimum chunk length must be an integer")

        if minlength < 1:
            raise ValueError("Minimum chunk length must be a positive integer")

        if maxlength <= minlength:
            raise ValueError("Maximum chunk length must be greater than the "
                             "minimum chunk length.")

        # chop up the data
        self.__change_point_indices_and_ratios = []
        self._chop_data(self.subtract_running_median(), threshold=threshold,
                        minlength=minlength)
        
        # sort the indices
        self.__change_point_indices_and_ratios = sorted(self.__change_point_indices_and_ratios)

        # if any chunks are longer than maxlength, then split them
        if maxlength < len(self):
            insertcps = []
            cppos = 0
            for i, clength in enumerate(self.chunk_lengths):
                if clength > maxlength:
                    insertcps.append((cppos + maxlength, 0))
                cppos += clength

            self.__change_point_indices_and_ratios.append(insertcps)
            self.__change_point_indices_and_ratios = sorted(self.__change_point_indices_and_ratios)

        # (re)calculate the variances for each chunk
        _ = self.compute_variance(N=self.window)

    @property
    def change_point_indices(self):
        """
        Return a list of indices of statistical change points in the data.
        """

        return [cps[0] for cps in self.__change_point_indices_and_ratios]

    @property
    def change_point_ratios(self):
        """
        Return a list of the log marginal likelihood ratios for the statistical
        change points in the data.
        """

        return [cps[1] for cps in self.__change_point_indices_and_ratios]

    @property
    def chunk_lengths(self):
        """
        A list with the lengths of the chunks into which the data has been
        split.
        """

        return np.diff(np.concatenate(([0], self.change_point_indices,
                                       [len(self)])))

    def _chop_data(self, data, threshold='default', minlength=5):
        # find change point
        lratio, cpidx, ntrials = self._find_change_point(data, minlength)
        
        # set the threshold
        if threshold == 'default':
            # default threshold for data splitting
            thresh = 4.07 + 1.33 * np.log10(len(data))
        elif threshold == 'trials':
            # assign equal prior probability for each hypothesis
            thresh = np.log(ntrials)
        elif isinstance(threshold, float):
            thresh = threshold
        else:
            raise ValueError("threshold is not recognised")

        if lratio > thresh:
            # split the data at the change point
            self.__change_point_indices_and_ratios.append((cpidx, lratio))

            # split the data and check for another change point
            chunk1 = data[0:cpidx]
            chunk2 = data[cpidx:]

            self._chop_data(chunk1, threshold, minlength)
            self._chop_data(chunk2, threshold, minlength)

    def _find_change_point(self, subdata, minlength):
        """
        Find the change point in the data, i.e., the "most likely" point at
        which the data could be split to be described by two independent
        zero mean Gaussian distributions. This also finds the evidence ratio
        for the data being described by any two independent zero mean Gaussian
        distributions compared to being described by only a single zero mean
        Gaussian.

        Parameters
        ----------
        subdata: array_like
            A complex array containing a chunk of data.
        minlength: int
            The minimum length of a chunk.
        
        Returns
        -------
        tuple:
            A tuple containing the maximum log Bayes factor, the index of the
            change point (i.e. the "best" point at which to split the data into
            two independent Gaussian distributions), and the number of
            denominator sub-hypotheses. 
        """

        if len(subdata) < 2*minlength:
            return (-np.inf, 0, 1)

        dlen = len(subdata)
        datasum = (np.abs(subdata)**2).sum()

        # calculate the evidence that the data is drawn from a zero mean
        # Gaussian with a single unknown standard deviation
        logsingle = (-lal.LN2 - dlen * lal.LNPI + logfactorial(dlen - 1) -
                     dlen * np.log(datasum))

        lsum = dlen - 2 * minlength + 1
        logtot = -np.inf

        logdouble = np.zeros(lsum)

        # go through each possible splitting of the data in two
        for i in range(lsum):
            sumforwards = (np.abs(subdata[:minlength+i])**2).sum()
            sumbackwards = (np.abs(subdata[minlength+i:])**2).sum()

            dlenf = minlength + i
            dlenb = dlen - (minlength + 1)

            logf = (-lal.LN2 - dlenf * lal.LNPI + logfactorial(dlenf - 1) -
                    dlenf * np.log(sumforwards))
            logb = (-lal.LN2 - dlenb * lal.LNPI + logfactorial(dlenb - 1) -
                    dlenb * np.log(sumbackwards))
            
            # evidence for that split
            logdouble[i] = logf + logb

            # evidence for *any* split
            logtot = np.logaddexp(logtot, logdouble[i])
        
        # change point (maximum of the split evidences)
        cp = np.argmax(logdouble) + minlength

        # ratio of any change point compared to no splits
        logratio = logtot - logsingle

        return (logratio, cp, lsum)

    def __len__(self):
        return len(self.data)
