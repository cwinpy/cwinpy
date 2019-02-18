"""
Functions to deal with heterodyned data.
"""

import numpy as np
import warnings
from collections import OrderedDict

# import lal and lalpulsar
import lal
import lalpulsar

# import utility functions
from .utils import logfactorial, gcd_array


class MultiHeterodynedData(object):
    """
    A class to contain time series' of heterodyned data, using the
    :class:`~cwinpy.data.HeterodynedData` class, for multiple detectors/data
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

    See the :class:`~cwinpy.data.HeterodynedData` documentation for information
    on additional keyword arguments.
    """

    def __init__(self, data=None, times=None, detector=None, window=30,
                 inject=False, par=None, injpar=None, freqfactor=2.0,
                 bbthreshold="default", remove_outliers=False, thresh=3.5,
                 **kwargs):

        # set keyword argument
        self.__heterodyned_data_kwargs = {}
        self.__heterodyned_data_kwargs['window'] = window
        self.__heterodyned_data_kwargs['par'] = par
        self.__heterodyned_data_kwargs['injpar'] = injpar
        self.__heterodyned_data_kwargs['inject'] = inject
        self.__heterodyned_data_kwargs['freqfactor'] = freqfactor
        self.__heterodyned_data_kwargs['bbthreshold'] = bbthreshold
        self.__heterodyned_data_kwargs['remove_outliers'] = remove_outliers
        self.__heterodyned_data_kwargs['thresh'] = thresh

        self.__data = OrderedDict()  # initialise empty dict
        self.__currentidx = 0  # index for iterator

        # add data
        if data is not None:
            self.add_data(data, times, detector=detector)

    def add_data(self, data, times=None, detector=None):
        """
        Add heterodyned data to the class.

        Parameters
        ----------
        data: (str, array_like, dict, HeterodynedData)
            The heterodyned data either as a string giving a file path, an
            array of data, a dictionary of file paths/data arrays that are
            keyed on valid detector names, or a
            :class:`~cwinpy.data.HeterodynedData` object.
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
        Get the list of :class:`~cwinpy.data.HeterodynedData` objects keyed to
        a given detector.
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
        :meth:`~cwinpy.data.HeterodynedData.bayesian_blocks` function.
    remove_outliers: bool, False
        If ``True`` outliers will be found (using
        :meth:`~cwinpy.data.HeterodynedData.find_outliers`) and removed from the
        data. They will not be stored anywhere in the class.
    thresh: float, 3.5
        The modified z-score threshold for outlier removal (see
        :meth:`~cwinpy.data.HeterodynedData.find_outliers`)
    """

    # set default Matplotlib setup parameters for plots
    defaultmplparams = {'backend': 'Agg',
                        'axes.linewidth': 0.5,  # set axes linewidths to 0.5
                        'axes.grid': True,      # add a grid
                        'grid.linewidth': 0.5,
                        'font.family': 'sans-serif',
                        'font.size': 15}

    # set some default detector color maps for plotting
    coldic = {'H1': 'r', 'L1': 'g', 'V1': 'b', 'G1': 'm'}
    colmapdic = {'H1': 'Reds', 'L1': 'Greens', 'V1': 'Blues', 'G1': 'PuRd'}

    def __init__(self, data=None, times=None, par=None, detector=None,
                 window=30, inject=False, injpar=None, injtimes=None,
                 freqfactor=2.0, fakeasd=None, bbthreshold="default",
                 remove_outliers=False, thresh=3.5):
        self.window = window  # set the window size
        self.__bbthreshold = bbthreshold
        self.__remove_outliers = remove_outliers
        self.__outlier_thresh = thresh

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

        # remove outliers if requested
        if self.__remove_outliers:
            outliers = self.find_outliers(thresh=self.__outlier_thresh)
            self.__data = self.data[~outliers]
            self.times = self.times[~outliers]

            if self.__stds is not None:
                self.__stds = self.__stds[~outliers]

        # set the (minimum) time step and sampling frequency
        if len(self.times) > 1:
            self.__dt = np.min(np.diff(self.times))
            self.__fs = 1./self.dt
        else:
            warnings.warn("Your data is only one data point long!")
            self.__dt = None
            self.__fs = None

        # for the moment there is the assumption that time stamps are integers,
        # so check this
        if not self.tottime.is_integer() or not self.dt.is_integer():
            raise ValueError("Time stamps must be integers")

        # initialise the running median
        _ = self.compute_running_median(N=self.window)

        # initialise change points to None
        self.__change_point_indices_and_ratios = None

        # calculate change points (and variances)
        self.bayesian_blocks(threshold=self.__bbthreshold)

    @property
    def times(self):
        return self.__times

    @property
    def tottime(self):
        """
        The total time (in seconds) of the data.
        """

        return self.__tottime

    @times.setter
    def times(self, times):
        """
        Set the data time stamps.
        """

        self.__times = np.asarray(times, dtype='float64')
        self.__tottime = self.times[-1] - self.times[0]

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
                startidx = i - (N//2) + 1
                endidx = len(self)
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

        if self.par is None:
            raise ValueError("To perform an injection a parameter file "
                             "must be supplied")

        if self.detector is None:
            raise ValueError("To perform an injection a detector "
                             "must be supplied")

        from lalpulsar.simulateHeterodynedCW import HeterodynedCWSimulator

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
            timeidxs = ((self.__times >= timerange[0]) &
                        (self.__times <= timerange[1]))
            inj_data[timeidxs] = inj[timeidxs]

        # add injection to data
        self.__data = self.data + inj_data

        # save injection data
        self.__inj_data = inj_data

        # (re)compute the running median
        _ = self.compute_running_median(N=self.window)

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
        else:
            timelist = injtimes

        try:
            timelist = np.atleast_2d(timelist)
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

           \\rho = \\sqrt{\\sum_i \\left(\\left[\\frac{\\Re{(s_i)}}{\\Re{(d_i)}}\\right]^2 + \\left[\\frac{\\Im{(s_i)}}{\\Im{(d_i)}}\\right]^2\\right)}

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

        If generating noise from a given detector's design curve, a frequency
        is required, which itself requires a 

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
                      'G1': lalsim.SimNoisePSDGEOHF,                  # GEOHF
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

            if self.par is None:
                raise AttributeError("A source parameter file containing a "
                                     "frequency is required")

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
                                 size=(len(self), 2))

        # add the noise to the data
        self.__data.real += noise[:, 0]
        self.__data.imag += noise[:, 1]

        # (re)compute the running median
        _ = self.compute_running_median(N=self.window)

        # (re)compute change points (and variances)
        self.bayesian_blocks(threshold=self.__bbthreshold)

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

        References
        ----------

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """

        if not isinstance(minlength, int):
            raise TypeError("Minimum chunk length must be an integer")

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
            for clength in self.chunk_lengths:
                if clength > maxlength:
                    insertcps.append((cppos + maxlength, 0))
                cppos += clength

            self.__change_point_indices_and_ratios += insertcps
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

        # don't try and split if all data is zero
        if np.all(self.subtract_running_median() == (0.+0*1j)):
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
            if (np.all(subdata[:minlength+i] == (0.+0*1j)) or
                    np.all(subdata[minlength+i:] == (0.+0*1j))):
                # do this to avoid warnings about np.log(0.0)
                logdouble[i] = -np.inf
            else:
                sumforwards = (np.abs(subdata[:minlength+i])**2).sum()
                sumbackwards = (np.abs(subdata[minlength+i:])**2).sum()

                dlenf = minlength + i
                dlenb = dlen - (minlength + i)

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

    def find_outliers(self, thresh=3.5):
        """
        Find, and return the indices of, and "outliers" in the data. This is a
        modified version of the median-absolute-deviation (MAD) function from
        [1]_, using the algorithm of [2]_.

        Parameters
        ----------
        thresh: float, 3.5
            The modified z-score to use as a threshold. Real or imaginary data
            with a modified z-score (based on the median absolute deviation)
            greater than this value will be classified as outliers.

        Returns
        -------
        array_like:
            A boolean :class:`numpy.ndarray` that is ``True`` for values that
            are outliers.

        References
        ----------

        .. [1] https://github.com/joferkington/oost_paper_code/blob/master/utilities.py and
           https://stackoverflow.com/a/22357811/1862861

        .. [2] Boris Iglewicz and David Hoaglin (1993), `"Volume 16: How to Detect and
           Handle Outliers"
           <https://hwbdocuments.env.nm.gov/Los%20Alamos%20National%20Labs/TA%2054/11587.pdf>`_,
           The ASQC Basic References in Quality Control:
           Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        """

        if not isinstance(thresh, float):
            raise TypeError("Threshold must be a float")
        else:
            if thresh <= 0.:
                raise ValueError("Threshold must be a positive number")

        modzscore = []

        for points in [self.data.real, self.data.imag]:
            median = np.median(points)
            diff = np.abs(points - median)  # only 1d data, so different from https://stackoverflow.com/a/22357811/1862861
            mad = np.median(diff)
            modzscore.append(0.6745 * diff / mad)

        # return boolean array of real or imaginary indices above the threshold
        return (modzscore[0] > thresh) | (modzscore[1] > thresh)

    def plot(self, rcparams={}, which='abs', figsize=(12, 4), ax=None,
             remove_outliers=False, thresh=3.5, zero_time=True,
             **plotkwargs):
        """
        Plot the data time series.

        Parameters
        ----------
        which: str, 'abs'
            Say whehther to plot the absolute value of the data, ``'abs'``, the
            ``'real'`` component of the data, the ``'imag'`` component of
            the data, or ``'both'`` the real and imaginary components.
        rcparams: dict, {}
            A dictionary of Matplotlib configuration parameter. Default values
            will be used if not set.
        figsize: tuple, (12, 4)
            A tuple with the size of the figure. Values set in `rcparams` will
            override this value.
        ax: Axes
            A :class:`matplotlib.axes.Axes` onto which to add the figure.
        remove_outliers: bool, False
            Set whether to remove outlier for the plot.
        thresh: float, 3.5
            The threshold for outlier removal (see
            :meth:`~cwinpy.data.HeterodynedData.find_outliers`).
        zero_time: bool, True
            Start the time axis at zero.
        plotkwargs:
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.
        
        Returns
        -------
        figure:
            The :class:`matplotlib.figure.Figure` containing the plot.

        Examples
        --------

        To plot both the real and imginary data one would do:

        >>> import numpy as np
        >>> from cwinpy import HeterodynedData
        >>> # create some fake data (as an example)
        >>> times = np.linspace(1000000000., 1000086340., 1440)
        >>> het = HeterdynedData(times=times, fakeasd=1e-48)
        >>> # plot real data
        >>> fig = het.plot(which='both')
        
        """

        if remove_outliers and not self.__remove_outliers:
            idx = self.find_outliers(thresh=thresh)
        else:
            idx = np.zeros(len(self), dtype=np.bool)

        # set the data to use
        if which.lower() in ['abs', 'absolute']:
            pldata = np.abs(self.data[~idx])
        elif which.lower() in ['real', 're']:
            pldata = self.data.real[~idx]
        elif which.lower() in ['im', 'imag', 'imaginary']:
            pldata = self.data.imag[~idx]
        elif which.lower() == 'both':
            pldata = (self.data.real[~idx], self.data.imag[~idx])
        else:
            raise ValueError("'which' must be 'abs', 'real', 'imag' or 'both")

        pltimes = self.times[~idx]
        t0 = pltimes[0]
        if zero_time:
            pltimes -= pltimes[0]

        try:
            from matplotlib import pyplot as pl
            from matplotlib.axes import Axes
            import matplotlib as mpl

            # make sure usetex is True (it doesn't work properly using it
            # inside the rc_content manager)
            mpl.rcParams['text.usetex'] = True

            if len(rcparams) == 0:
                rcparams = self.defaultmplparams.copy()

            # set default figure sizes
            if 'figure.figsize' not in rcparams:
                rcparams['figure.figsize'] = figsize

            if 'figure.autolayout' not in rcparams:
                rcparams['figure.autolayout'] = True

            # set 'label' and 'color' defaults
            if self.detector is not None:
                if 'label' not in plotkwargs:
                    plotkwargs['label'] = self.detector

                if self.detector in self.coldic and 'color' not in plotkwargs:
                    plotkwargs['color'] = self.coldic[self.detector]

            # set some default plotting styles
            if 'ls' not in plotkwargs:
                # set the line style to "None"
                plotkwargs['ls'] = 'None'

            if 'marker' not in plotkwargs:
                # set marker to a circle
                plotkwargs['marker'] = 'o'

            with mpl.rc_context(rc=rcparams):
                if isinstance(ax, Axes):
                    fig = ax.get_figure()
                    thisax = ax
                else:
                    fig, thisax = pl.subplots()

                if which.lower() != 'both':
                    thisax.plot(pltimes, pldata, **plotkwargs)
                else:
                    # plot real and imaginary components
                    plotkwargs['markerfacecolor'] = 'None'
                    for i in range(2):
                        copykwargs = plotkwargs.copy()
                        if i == 0:
                            copykwargs['marker'] = 'o'
                            if 'label' in plotkwargs:
                                copykwargs['label'] = 'Real {}'.format(plotkwargs['label'])
                            else:
                                copykwargs['label'] = 'Real'
                        else:
                            copykwargs['marker'] = '+'
                            if 'label' in plotkwargs:
                                copykwargs['label'] = 'Imag {}'.format(plotkwargs['label'])
                            else:
                                copykwargs['label'] = 'Imag'

                        thisax.plot(pltimes, pldata[i], **copykwargs)
                    plotkwargs['label'] = True  # add to produce legend below

                if zero_time:
                    thisax.set_xlabel('GPS - {}'.format(int(t0)))
                    thisax.set_xlim([0., pltimes[-1]])
                else:
                    thisax.set_xlabel('GPS time')
                    thisax.set_xlim([pltimes[0], pltimes[-1]])

                if which.lower() in ['abs', 'absolute']:
                    thisax.set_ylabel('$|B_k|$')
                elif which.lower() in ['real', 're']:
                    thisax.set_ylabel('$\\Re{(B_k)}$')
                elif which.lower() in ['imag', 'im']:
                    thisax.set_ylabel('$\\Im{(B_k)}$')
                else:
                    thisax.set_ylabel('$B_k$')

                if 'label' in plotkwargs:
                    thisax.legend(loc='best')
        except Exception as e:
            raise RuntimeError("Problem with plotting: {}".format(e))

        return fig

    def spectrogram(self, dt=86400, window=None, overlap=0.5, plot=True,
                    ax=None, rcparams={}, remove_outliers=False, thresh=3.5,
                    fraction_labels=True, fraction_label_num=4,
                    **plotkwargs):
        """
        Compute and plot a spectrogram from the data using the
        :func:`matplotlib.mlab.specgram` function.

        Parameters
        ----------
        dt: (float, int)
            The length of time (in seconds) for each spectrogram time bin.
            The default is 86400 seconds (i.e., one day).
        window: (callable, np.ndarray)
            The window to apply to each FFT block. Default is to use
            :func:`scipy.signal.tukey` with the `alpha` parameter set to 0.1.
        overlap: (float, int)
            If a floating point number between [0, 1) this gives the fractional
            overlap between adjacent FFT blocks (which defaults to 0.5, i.e., a
            50% overlap). If an integer of 1 or more this is the number of
            points to overlap between adjacent FFT blocks (this is how the
            argument is used in :func:`~matplotlib.mlab.specgram`).
        plot: bool, True
            By default a plot of the spectrogram will be produced (this can be
            plotted on a supplied :class:`~matplotlib.axes.Axes` or
            :class:`~matplotlib.figure.Figure`), but the plotting can be turned
            off if this is set to ``False``.
        ax: (axes, figure)
            If `ax` is a :class:`matplotlib.axes.Axes` then the spectrogram
            will be plotted on the supplied axis.
        rcparams: dict, None
            A dictionary of Matplotlib configuration parameters
            (:class:`matplotlib.RcParams`) for plotting. If ``None``, and an
            :class:`~matplotlib.axes.Axes` or
            :class:`~matplotlib.figure.Figure` is not supplied, the some
            default styles will be used.
        remove_outliers: bool, False
            Set to ``True`` to remove outliers points before generating the
            spectrogram. This is not required if the class was created with
            the `remove_outliers` keyword already set to ``True``.
        thresh: float, 3.5
            The modified z-score threshold for outlier removal (see
            :meth:`~cwinpy.data.HeterodynedData.find_outliers`).
        fraction_labels: bool, True
            Set to ``True`` to output the frequency labels on the plot as
            fractions.
        fraction_label_num: int, 4
            The fraction labels will be spaced at `Fs`/`fraction_label_num`
            intervals, between the upper and lower Nyquist values. The default
            if 4, i.e., spacing will be at a quarter of the Nyquist frequency. 
        plotkwargs:
            Keyword arguments for :func:`matplotlib.pyplot.imshow`.

        Returns
        -------
        array_like:
            A :class:`numpy.ndarray` of frequencies for the spectrogram
        array_like:
            A 2d :class:`numpy.ndarray` of the spectrogram power at each
            frequency and time
        array_like:
            A :class:`numpy.ndarray` of the central times of each FFT in the
            spectrogram.
        figure:
            The :class:`~matplotlib.figure.Figure` containing the spectrogram
            plot. This is not returned if `plot` is set to ``False``.
        """

        speckwargs = {}
        speckwargs['dt'] = dt
        speckwargs['window'] = window
        speckwargs['overlap'] = overlap
        speckwargs['rcparams'] = rcparams
        speckwargs['remove_outliers'] = remove_outliers
        speckwargs['thresh'] = thresh
        speckwargs['fraction_labels'] = fraction_labels
        speckwargs['fraction_label_num'] = fraction_label_num

        return self._plot_power('spectrogram', speckwargs, **plotkwargs)

    def periodogram(self, plot=True, ax=None, rcparams={},
                    remove_outliers=False, thresh=3.5, fraction_labels=True,
                    fraction_label_num=4, **plotkwargs):
        """
        Compute and plot a two-sided periodogram of the data using
        :func:`scipy.signal.periodogram`. Note that this uses zero-padded
        uniformly sampled data, rather than using the Lomb-Scargle method (such
        as :class:`astropy.stats.LombScargle`) that can deal with data with
        gaps, but doesn't work for complex data.

        See :meth:`~cwinpy.data.HeterodynedData.spectrogram` for input
        parameters, excluding `dt`, `window` and `overlap`. 

        Parameters
        ----------
        plotkwargs:
            Keyword parameters for :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        array_like:
            The frequency series
        array_like:
            The periodogram power
        figure:
            The :class:`~matplotlib.figure.Figure` is a plot is requested.
        """

        speckwargs = {}
        speckwargs['plot'] = plot
        speckwargs['ax'] = ax
        speckwargs['rcparams'] = rcparams
        speckwargs['remove_outliers'] = remove_outliers
        speckwargs['thresh'] = thresh
        speckwargs['fraction_labels'] = fraction_labels
        speckwargs['fraction_label_num'] = fraction_label_num

        return self._plot_power('periodogram', speckwargs, **plotkwargs)

    def power_spectrum(self, plot=True, ax=None, rcparams={},
                       remove_outliers=False, thresh=3.5, fraction_labels=True,
                       fraction_label_num=4, average='median', dt=86400,
                       **plotkwargs):
        """
        Compute and plot the power spectrum of the data. This compute the
        spectrogram, and averages the power over time.

        See :meth:`~cwinpy.data.HeterodynedData.spectrogram` for input parameters,
        excluding `window` and `overlap`. 

        Parameters
        ----------
        average: str, 'median'
            The method by which to "average" the spectrum in time. This can be
            'median' (the default) or 'mean'.
        plotkwargs:
            Keyword parameters for :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        array_like:
            The frequency series
        array_like:
            The power spectrum
        figure:
            The :class:`~matplotlib.figure.Figure` is a plot is requested.
        """

        speckwargs = {}
        speckwargs['plot'] = plot
        speckwargs['ax'] = ax
        speckwargs['rcparams'] = rcparams
        speckwargs['remove_outliers'] = remove_outliers
        speckwargs['thresh'] = thresh
        speckwargs['fraction_labels'] = fraction_labels
        speckwargs['fraction_label_num'] = fraction_label_num
        speckwargs['dt'] = dt
        speckwargs['average'] = average

        return self._plot_power('power', speckwargs, **plotkwargs)

    def _plot_power(self, ptype, speckwargs={}, **plotkwargs):
        """
        General function for plotting the
        :meth:`~cwinpy.data.HeterodynedData.spectrogram`,
        :meth:`~cwinpy.data.HeterodynedData.power_spectrum` or
        :meth:`~cwinpy.data.HeterodynedData.periodogram`.

        Parameters
        ----------
        ptype: str
            A string with 'spectrogram' for
            :meth:`~cwinpy.data.HeterodynedData.spectrogram`, 'periodogram' for
            :meth:`~cwinpy.data.HeterodynedData.periodogram`, or 'power' for
            :meth:`~cwinpy.data.HeterodynedData.power_spectrum`.
        speckwargs: dict
            A dictionary of spectrum generation keyword arguments.
        plotkwargs:
            Additional plotting keyword arguments.
        """

        if not isinstance(ptype, str):
            raise TypeError("Power spectrum type must be a string")

        if ptype not in ['spectrogram', 'periodogram', 'power']:
            raise ValueError("Type must be 'spectrogram', 'periodogram', or "
                             "'power'")

        # get some options
        remove_outliers = speckwargs.get('remove_outliers', False)
        thresh = speckwargs.get('thresh', 3.5)
        plot = speckwargs.get('plot', True)
        ax = speckwargs.get('ax', None)
        rcparams = speckwargs.get('rcparams', {})
        if len(rcparams) == 0:
            rcparams = self.defaultmplparams.copy()

        # get the zero padded data
        padded = self._zero_pad(remove_outliers=remove_outliers, thresh=thresh)

        if not self.__remove_outliers and remove_outliers:
            idx = self.find_outliers(thresh=thresh)
            times = self.times[~idx]
            tottime = times[-1] - times[0]
        else:
            times = self.times
            tottime = self.tottime

        Fs = 1./gcd_array(np.diff(times))  # sampling frequency

        if ptype in ['spectrogram', 'power']:
            dt = speckwargs.get('dt', 86400)
            overlap = speckwargs.get('overlap', 0.5)
            window = speckwargs.get('window', None)
            
            if not isinstance(dt, (float, int)):
                raise ValueError("Time bin must be an integer or float")

            if dt < 1./Fs or dt > (tottime + (1./Fs)):
                raise ValueError("The time bin selected is invalid")

            # set the number of samples for each FFT block
            nfft = int(dt*Fs)

            if isinstance(overlap, float):
                if overlap >= 0. and overlap < 1.:
                    noverlap = int(overlap*nfft)
                else:
                    raise ValueError("Overlap must be a float between 0 and 1")
            elif isinstance(overlap, int):
                if overlap >= 0 and overlap <= len(self)-1:
                    noverlap = overlap
                else:
                    raise ValueError("Overlap is out of allowed range")
            else:
                raise TypeError("Overlap must be an integer or float")

            if window is None:
                from scipy.signal import tukey

                window = tukey(nfft, alpha=0.1)

            # generate spectrogram
            try:
                from matplotlib.mlab import specgram

                power, frequencies, stimes = specgram(padded, Fs=Fs,
                                                      window=window,
                                                      NFFT=nfft,
                                                      noverlap=noverlap)
            except Exception as e:
                raise RuntimeError("Problem creating spectrogram: "
                                   "{}".format(e))

            if ptype == 'power':
                # average the spectrogram for a power spectrum
                average = speckwargs.get('average', 'median')

                if average not in ['median', 'mean']:
                    raise ValueError("Average method must be 'median' or 'mean'")

                if average == 'median':
                    power = np.median(power, axis=-1)
                else:
                    power = np.mean(power, axis=-1)
        else:
            # perform periodogram
            try:
                from scipy.signal import periodogram

                frequencies, power = periodogram(padded, fs=Fs,
                                                 return_onesided=False,
                                                 detrend=lambda x: x)

                # sort results in frequency
                frequencies, power = np.array(sorted(zip(frequencies,
                                                         power))).T
            except Exception as e:
                raise RuntimeError("Problem creating periodogram: "
                                   "{}".format(e))

        if ax is None and not plot:
            if ptype == 'spectrogram':
                return frequencies, power, stimes
            else:
                return frequencies, power

        # perform plotting
        try:
            from matplotlib import pyplot as pl
            from matplotlib.axes import Axes
            import matplotlib as mpl

            # make sure usetex is True (it doesn't work properly using it
            # inside the rc_content manager)
            mpl.rcParams['text.usetex'] = True

            # set default figure sizes
            if ptype == 'spectrogram':
                if 'figure.figsize' not in rcparams:
                    rcparams['figure.figsize'] = (12, 4)
            else:
                if 'figure.figsize' not in rcparams:
                    rcparams['figure.figsize'] = (6, 5)

            fraction_labels = speckwargs.get('fraction_labels', True)
            fraction_label_num = speckwargs.get('fraction_label_num', 4)

            # set whether to output frequency labels as fractions
            if fraction_labels:
                # add to LaTeX preamble
                if 'xfrac' not in mpl.rcParams['text.latex.preamble']:
                    mpl.rcParams['text.latex.preamble'].append('\\usepackage{xfrac}')

                # set at quarters of the sample frequency
                if not isinstance(fraction_label_num, int):
                    raise TypeError("'fraction_label_num' must be an integer")

                if fraction_label_num < 1:
                    raise ValueError("'fraction_label_num' must be positive")

                df = Fs / fraction_label_num
                ticks = np.linspace(-2 / Fs, 2 / Fs, int(Fs / df) + 1)
                labels = []
                for tick in ticks:
                    if tick == 0.:
                        labels.append('$0$')
                    else:
                        sign = '-' if tick < 0. else ''
                        label = "${0}\\sfrac{{{1}}}{{{2}}}$".format(sign,
                                                                    1,
                                                                    int(np.abs(tick)))
                        labels.append(label)

                if ptype != 'spectrogram':
                    ticks = np.linspace(-Fs / 2, Fs / 2, int(Fs / df) + 1)

            if ptype == 'spectrogram':
                from matplotlib import colors

                # set plotting keyword arguments
                if 'cmap' not in plotkwargs:
                    if self.detector is not None:
                        if self.detector in self.colmapdic:
                            plotkwargs['cmap'] = self.colmapdic[self.detector]

                # extents of the plot
                if 'extent' not in plotkwargs:
                    plotkwargs['extent'] = [0, tottime, -2 / Fs, 2 / Fs]

                if 'aspect' not in plotkwargs:
                    plotkwargs['aspect'] = 'auto'

                if 'norm' not in plotkwargs:
                    plotkwargs['norm'] = colors.Normalize()

                with mpl.rc_context(rc=rcparams):
                    if isinstance(ax, Axes):
                        fig = ax.get_figure()
                        thisax = ax
                    else:
                        fig, thisax = pl.subplots()

                    thisax.imshow(np.sqrt(np.flipud(power)), **plotkwargs)

                    if self.detector is not None:
                        from matplotlib.offsetbox import AnchoredText
                        legend = AnchoredText(self.detector, loc=1)
                        thisax.add_artist(legend)

                    thisax.set_xlabel('GPS - {}'.format(int(times[0])))
                    thisax.set_ylabel('Frequency (Hz)')

                    if fraction_labels:
                        thisax.set_yticks(ticks)
                        thisax.set_yticklabels(labels)
            else:
                # set plot color
                color = None
                label = None
                if self.detector is not None:
                    if self.detector in self.coldic:
                        color = self.coldic[self.detector]
                    label = self.detector

                if self.detector is not None:
                    if 'color' not in plotkwargs:
                        if self.detector in self.coldic:
                            plotkwargs['color'] = self.coldic[self.detector]

                    if 'label' not in plotkwargs:
                        plotkwargs['label'] = self.detector

                with mpl.rc_context(rc=rcparams):
                    if isinstance(ax, Axes):
                        fig = ax.get_figure()
                        thisax = ax
                    else:
                        fig, thisax = pl.subplots()

                    thisax.plot(frequencies, power, **plotkwargs)

                    if self.detector is not None:
                        thisax.legend()

                    thisax.set_ylabel('Power')
                    thisax.set_xlabel(r'Frequency (Hz)')
                    thisax.set_xlim([-Fs / 2, Fs / 2])

                    if fraction_labels:
                        thisax.set_xticks(ticks)
                        thisax.set_xticklabels(labels)
        except Exception as e:
            raise RuntimeError("Problem creating spectrogram: {}".format(e))

        if ptype == 'spectrogram':
            return frequencies, power, stimes, fig
        else:
            return frequencies, power, fig

    def _zero_pad(self, remove_outliers=False, thresh=3.5):
        """
        If required zero pad the data to return an evenly sampled dataset for
        use in generating a power spectrum.

        Parameters
        ----------
        remove_outliers: bool, False
            If ``True`` remove outliers before zero padding (nothing is done
            if outliers have already been removed).
        thresh: float, 3.5
            The modified z-score threshold for outlier removal.

        Returns
        -------
        :class:`numpy.ndarray`:
            An array of the data padded with zeros.
        """

        if not self.__remove_outliers and remove_outliers:
            idx = self.find_outliers(thresh=thresh)
            times = self.times[~idx]
            data = self.data[~idx]
        else:
            times = self.times
            data = self.data

        # check diff of times
        if len(times) < 2:
            raise ValueError("There must be at least two samples!")

        dts = np.diff(times).astype(np.float32)  # convert to float32 due to precision errors

        if np.all(dts == self.dt):
            # no zero padding required as data is evenly sampled
            return data

        # get the greatest common divisor of the deltaTs
        gcd = gcd_array(dts)

        # get the "new" padded time stamps
        tottime = times[-1] - times[0]
        newtimes = np.linspace(times[0], times[-1],
                               1 + int(tottime) // gcd)

        # get indices of original times im new times
        tidxs = np.where(np.in1d(newtimes, times))[0]

        # get zero array and add data
        padded = np.zeros(len(newtimes), dtype=np.complex)
        padded[tidxs] = data

        return padded

    def __len__(self):
        return len(self.data)
