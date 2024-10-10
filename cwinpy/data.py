"""
Classes for dealing with data products.
"""

import ast
import os
from pathlib import Path
from warnings import warn

import lal
import lalpulsar
import numpy as np
from astropy.io import registry as io_registry
from gwpy.detector import Channel
from gwpy.io.mp import read_multi
from gwpy.plot.colors import GW_OBSERVATORY_COLORS
from gwpy.segments import SegmentList
from gwpy.timeseries import TimeSeries, TimeSeriesBase
from gwpy.types import Series
from numba import jit
from scipy.ndimage import median_filter
from scipy.stats import hmean

import cwinpy

from .parfile import PulsarParameters
from .utils import allzero, gcd_array, get_psr_name, is_par_file, logfactorial


class MultiHeterodynedData:
    """
    A class to contain time series' of heterodyned data, using the
    :class:`~cwinpy.data.HeterodynedData` class, for multiple detectors/data
    streams.

    Parameters
    ----------
    data: str, array_like, dict, HeterodynedData
        The heterodyned data either as a string giving a file path, an array of
        data, or a dictionary of file paths/data arrays, that are keyed on
        valid detector names.
    times: array_like, dict
        If `data` is an array, or dictionary of arrays, then `times` must be
        set giving the time stamps for the data values. If `times` is a
        dictionary then it should be keyed on the same detector names as in
        `data`.
    detector: str, ``lal.Detector``
        If `data` is a file name or data array then `detector` must be given as
        a string or :class:`lal.Detector`.

    Notes
    -----

    See the :class:`~cwinpy.data.HeterodynedData` documentation for information
    on additional keyword arguments.
    """

    def __init__(
        self,
        data=None,
        times=None,
        detector=None,
        **kwargs,
    ):

        # set keyword argument
        self._heterodyned_data_kwargs = {}
        self._heterodyned_data_kwargs["window"] = kwargs.pop("window", 30)
        self._heterodyned_data_kwargs["par"] = kwargs.pop("par", None)
        self._heterodyned_data_kwargs["injpar"] = kwargs.pop("injpar", None)
        self._heterodyned_data_kwargs["inject"] = kwargs.pop("inject", False)
        self._heterodyned_data_kwargs["freqfactor"] = kwargs.pop("freqfactor", 2.0)
        self._heterodyned_data_kwargs["bbthreshold"] = kwargs.pop(
            "bbthreshold", "default"
        )
        self._heterodyned_data_kwargs["remove_outliers"] = kwargs.pop(
            "remove_outliers", False
        )
        self._heterodyned_data_kwargs["thresh"] = kwargs.pop("thresh", 3.5)
        self._heterodyned_data_kwargs.update(kwargs)

        self._data = dict()  # initialise empty dict
        self._currentidx = 0  # index for iterator

        # add data
        if data is not None:
            self.add_data(data, times, detector=detector)

    def add_data(self, data, times=None, detector=None):
        """
        Add heterodyned data to the class.

        Parameters
        ----------
        data: str, array_like, dict, HeterodynedData
            The heterodyned data either as a string giving a file path, an
            array of data, a dictionary of file paths/data arrays that are
            keyed on valid detector names, or a
            :class:`~cwinpy.data.HeterodynedData` object.
        times: array_like, dict
            If `data` is an array, or dictionary of arrays, then `times` must
            be set giving the time stamps for the data values. If `times` is
            a dictionary then it should be keyed on the same detector names as
            in `data`.
        detector: str, ``lal.Detector``
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
                            raise KeyError(
                                "'times' does not contain the "
                                "detector: {}".format(detkey)
                            )
                        else:
                            dettimes = times[detkey]
                    else:
                        dettimes = times

                    self._add_data(data[detkey], detkey, dettimes)
        else:
            if isinstance(times, dict):
                raise TypeError("'times' should not be a dictionary")

            self._add_data(data, detector, times)

    def _add_HeterodynedData(self, data):
        detname = data.detector
        if detname not in self._data:
            self._data[detname] = [data]  # add as a list
        else:
            # if data from that detector already exists then append to the list
            self._data[detname].append(data)

    def _add_data(self, data, detector, times=None):
        if not isinstance(data, (str, Path)):
            if detector is None or data is None:
                raise ValueError("data and detector must be set")

        het = HeterodynedData(
            data, times, detector=detector, **self._heterodyned_data_kwargs
        )

        self._add_HeterodynedData(het)

    def __getitem__(self, det):
        """
        Get the list of :class:`~cwinpy.data.HeterodynedData` objects keyed to
        a given detector.
        """

        if det in self.detectors:
            return self._data[det]
        else:
            return None

    def pop(self, det):
        return self._data.pop(det)

    @property
    def to_list(self):
        datalist = []
        for key in self._data:
            if isinstance(self._data[key], list):
                datalist += self._data[key]
            else:
                datalist.append(self._data[key])

        return datalist

    @property
    def detectors(self):
        """
        Return the list of detectors contained in the object.
        """

        return list(self._data.keys())

    @property
    def pars(self):
        """
        Return the list of heterodyne source parameter files for each data set
        contained in the object.
        """

        return [het.par for het in self]

    @property
    def freq_factors(self):
        """
        Return the this of heterodyne frequency scaling factors for each data
        set contained in the object.
        """

        return [het.freq_factor for het in self]

    @property
    def injection_snr(self):
        """
        Get the coherent optimal signal-to-noise ratio of an injected signal in
        all heterodyned data sets. See
        :meth:`cwinpy.data.HeterodynedData.injection_snr`.
        """

        snr2 = 0.0
        for het in self:
            if het.injpar is not None:
                snr2 += het.injection_snr**2

        return np.sqrt(snr2)

    def signal_snr(self, signalpar):
        """
        Get the coherent signal-to-noise ratio of a given signal. See
        :meth:`cwinpy.data.HeterodynedData.signal_snr`.
        """

        snr2 = 0.0
        for het in self:
            snr2 += het.signal_snr(signalpar) ** 2

        return np.sqrt(snr2)

    def __iter__(self):
        self._currentidx = 0  # reset iterator index
        return self

    def __next__(self):
        if self._currentidx >= len(self):
            raise StopIteration
        else:
            self._currentidx += 1
            return self.to_list[self._currentidx - 1]

    def plot(
        self,
        det=None,
        together=False,
        which="abs",
        figsize=(12, 4),
        remove_outliers=False,
        thresh=3.5,
        zero_time=False,
        labelsize=None,
        fontsize=None,
        legendsize=None,
        fontname=None,
        labelname=None,
        **plotkwargs,
    ):
        """
        Plot all, or some of, the time series' contained in the class. The
        general arguments can be seen in
        :meth:`cwinpy.data.HeterodynedData.plot` and additional arguments are
        given below.

        Parameters
        ----------
        together: bool, False
            Set to ``True`` to put all the plots onto one figure, otherwise
            they will be created on individual
            :class:`~matplotlib.figure.Figure` objects.
        det: str
            If a detector name is supplied, then only the time series' for that
            detector will be plotted.

        Returns
        -------
        list:
            A :class:`~matplotlib.figure.Figure` object, or list of
            :class:`~matplotlib.figure.Figure` objects.
        """

        from gwpy.plot import Plot

        if len(self) == 0:
            # nothing in the class!
            return None

        # set which plots to output
        ndet = 1
        if det is not None:
            if det not in self.detectors:
                raise ValueError("Detector {} is not in the class".format(det))

            # get the number of time series' for the requested detector
            ndet = len(self[det])

        nplots = 1
        if together:
            if ndet > 1:
                nplots = ndet
                hets = self[det]
            else:
                nplots = len(self)
                hets = self  # datasets to plot

            # create the figure
            if figsize[0] == 12 and figsize[1] == 4:
                # check default size and increase
                figsize = (figsize[0], figsize[1] * nplots)

            # set up plot using dummy data
            figs = Plot(
                *[
                    TimeSeries(
                        d.real, times=d.times - (0 if not zero_time else d.times[0])
                    )
                    for d in hets
                ],
                figsize=figsize,
                separate=True,
                sharex=True,
            )

            # replot data on axes
            for het, ax in zip(hets, figs.get_axes()):
                # remove dummy data
                ax.clear()
                ax.set_prop_cycle(None)  # reset color cycle

                # set color based on detector
                if "color" not in plotkwargs:
                    color = GW_OBSERVATORY_COLORS[het.detector]
                else:
                    color = None

                # re-plot
                _ = het.plot(
                    which=which,
                    ax=ax,
                    remove_outliers=remove_outliers,
                    thresh=thresh,
                    zero_time=zero_time,
                    labelsize=labelsize,
                    fontsize=fontsize,
                    legendsize=legendsize,
                    fontname=fontname,
                    labelname=labelname,
                    color=color,
                    **plotkwargs,
                )
        else:
            # a list of figures
            figs = []

            if det is not None:
                hets = self[det]
            else:
                hets = self

            # loop over data and produce plots
            for het in hets:
                figs.append(
                    het.plot(
                        which=which,
                        figsize=figsize,
                        remove_outliers=remove_outliers,
                        thresh=thresh,
                        zero_time=zero_time,
                        labelsize=labelsize,
                        fontsize=fontsize,
                        legendsize=legendsize,
                        fontname=fontname,
                        labelname=labelname,
                        **plotkwargs,
                    )
                )

        return figs

    def power_spectrum(
        self,
        det=None,
        together=False,
        figsize=None,
        remove_outliers=None,
        thresh=None,
        labelsize=None,
        fontsize=None,
        legendsize=None,
        fontname=None,
        labelname=None,
        dt=None,
        fraction_labels=None,
        fraction_label_num=None,
        average=None,
        window=None,
        overlap=None,
        **plotkwargs,
    ):
        """
        Plot all, or some of, the power spectral densities of the time series'
        contained in the class. The general arguments can be seen in
        :meth:`cwinpy.data.HeterodynedData.power_spectrum` and additional
        arguments are given below.

        Parameters
        ----------
        together: bool, False
            Set to ``True`` to put all the plots onto one figure, otherwise
            they will be created on individual
            :class:`~matplotlib.figure.Figure` objects.
        det: str
            If a detector name is supplied, then only the time series' for that
            detector will be plotted.

        Returns
        -------
        list:
            A :class:`~matplotlib.figure.Figure` object, or list of
            :class:`~matplotlib.figure.Figure` objects.
        """

        return self._plot_power(
            "power",
            det=det,
            together=together,
            figsize=figsize,
            remove_outliers=remove_outliers,
            thresh=thresh,
            labelsize=labelsize,
            fontsize=fontsize,
            legendsize=legendsize,
            labelname=labelname,
            fontname=fontname,
            dt=dt,
            fraction_labels=fraction_labels,
            fraction_label_num=fraction_label_num,
            average=average,
            window=window,
            overlap=overlap,
            **plotkwargs,
        )

    def periodogram(
        self,
        det=None,
        together=False,
        figsize=None,
        remove_outliers=None,
        thresh=None,
        labelsize=None,
        fontsize=None,
        legendsize=None,
        fontname=None,
        labelname=None,
        fraction_labels=None,
        fraction_label_num=None,
        **plotkwargs,
    ):
        """
        Plot all, or some of, the periodograms of the time series' contained
        in the class. The general arguments can be seen in
        :meth:`cwinpy.data.HeterodynedData.periodogram` and additional
        arguments are given below.

        Parameters
        ----------
        together: bool, False
            Set to ``True`` to put all the plots onto one figure, otherwise
            they will be created on individual
            :class:`~matplotlib.figure.Figure` objects.
        det: str
            If a detector name is supplied, then only the time series' for that
            detector will be plotted.

        Returns
        -------
        list:
            A :class:`~matplotlib.figure.Figure` object, or list of
            :class:`~matplotlib.figure.Figure` objects.
        """

        return self._plot_power(
            "periodogram",
            det=det,
            together=together,
            figsize=figsize,
            remove_outliers=remove_outliers,
            thresh=thresh,
            labelsize=labelsize,
            fontsize=fontsize,
            legendsize=legendsize,
            labelname=labelname,
            fontname=fontname,
            fraction_labels=fraction_labels,
            fraction_label_num=fraction_label_num,
            **plotkwargs,
        )

    def spectrogram(
        self,
        det=None,
        together=False,
        figsize=None,
        remove_outliers=None,
        thresh=None,
        labelsize=None,
        fontsize=None,
        legendsize=None,
        fontname=None,
        labelname=None,
        fraction_labels=None,
        fraction_label_num=None,
        dt=None,
        overlap=None,
        window=None,
        **plotkwargs,
    ):
        """
        Plot all, or some of, the spectograms of the time series' contained
        in the class. The general arguments can be seen in
        :meth:`~cwinpy.data.HeterodynedData.spectrogram` and additional
        arguments are given below.

        Parameters
        ----------
        together: bool, False
            Set to ``True`` to put all the plots onto one figure, otherwise
            they will be created on individual
            :class:`~matplotlib.figure.Figure` objects.
        det: str
            If a detector name is supplied, then only the time series' for that
            detector will be plotted.

        Returns
        -------
        list:
            A :class:`~matplotlib.figure.Figure` object, or list of
            :class:`~matplotlib.figure.Figure` objects.
        """

        return self._plot_power(
            "spectrogram",
            det=det,
            together=together,
            figsize=figsize,
            window=window,
            remove_outliers=remove_outliers,
            thresh=thresh,
            labelsize=labelsize,
            fontsize=fontsize,
            legendsize=legendsize,
            labelname=labelname,
            fontname=fontname,
            dt=dt,
            fraction_labels=fraction_labels,
            fraction_label_num=fraction_label_num,
            overlap=overlap,
            **plotkwargs,
        )

    def _plot_power(
        self,
        plottype,
        det=None,
        together=False,
        figsize=None,
        remove_outliers=None,
        thresh=None,
        labelsize=None,
        fontsize=None,
        legendsize=None,
        fontname=None,
        labelname=None,
        dt=None,
        average=None,
        overlap=None,
        window=None,
        fraction_labels=None,
        fraction_label_num=None,
        **plotkwargs,
    ):
        """
        General purpose function for plotting the various spectrum figures.

        Parameters
        ----------
        plottype: str
            The "spectrum" plots that are required: ``"power_spectrum"``,
            ``"periodogram"``, or ``"spectrogram"``
        """

        from matplotlib import pyplot as plt

        if plottype.lower() not in ["spectrogram", "periodogram", "power"]:
            raise ValueError("Spectrum plot type is not known")

        if len(self) == 0:
            # nothing in the class!
            return None

        # set which plots to output
        ndet = 1
        if det is not None:
            if det not in self.detectors:
                raise ValueError("Detector {} is not in the class".format(det))

            # get the number of time series' for the requested detector
            ndet = len(self[det])

        # set keyword arguments
        speckwargs = {}
        for key, value in zip(
            [
                "thresh",
                "remove_outliers",
                "labelsize",
                "labelname",
                "fontsize",
                "fontname",
                "legendsize",
                "fraction_labels",
                "fraction_label_num",
                "figsize",
            ],
            [
                thresh,
                remove_outliers,
                labelsize,
                labelname,
                fontsize,
                fontname,
                legendsize,
                fraction_labels,
                fraction_label_num,
                figsize,
            ],
        ):
            if value is not None:
                speckwargs[key] = value

        if plottype.lower() == "power" and average is not None:
            speckwargs["average"] = average
        if plottype.lower() in ["spectrogram", "power"]:
            if overlap is not None:
                speckwargs["overlap"] = overlap
            if window is not None:
                speckwargs["window"] = window
            if dt is not None:
                speckwargs["dt"] = dt

        nplots = 1
        if together:
            if ndet > 1:
                nplots = ndet
                hets = self[det]
            else:
                nplots = len(self)
                hets = self  # datasets to plot

            # create the figure
            if figsize is None:
                # create default size
                if plottype.lower() == "spectrogram":
                    figsize = (12, 4 * nplots)
                else:
                    figsize = (6, 5 * nplots)

            figs, axs = plt.subplots(nplots, 1, figsize=figsize)

            for ax, het in zip(axs, hets):
                if plottype.lower() == "periodogram":
                    plfunc = het.periodogram
                elif plottype.lower() == "power":
                    plfunc = het.power_spectrum
                else:
                    plfunc = het.spectrogram

                _ = plfunc(**speckwargs, ax=ax, **plotkwargs)

            figs.tight_layout()
        else:
            # a list of figures
            figs = []

            if det is not None:
                hets = self[det]
            else:
                hets = self

            # loop over data and produce plots
            for het in hets:
                if plottype.lower() == "periodogram":
                    plfunc = het.periodogram
                    figidx = 2
                elif plottype.lower() == "power":
                    plfunc = het.power_spectrum
                    figidx = 2
                else:
                    plfunc = het.spectrogram
                    figidx = 3

                figs.append(plfunc(**speckwargs, **plotkwargs)[figidx])

        return figs

    def __len__(self):
        length = 0
        for key in self._data:
            if isinstance(self._data[key], list):
                length += len(self._data[key])
            else:
                length += 1
        return length


class HeterodynedData(TimeSeriesBase):
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
    data: str, list, array_like
        A file (plain ascii text, gzipped ascii text, ascii CSV, or HDF5 file)
        containing a time series of heterodyned data, or list of files names
        (from the same detector/frequency scale factor/pulsar) to be
        concatenated, or an array containing the complex heterodyned data.
    times: array_like
        If the data was passed using the `data` argument, then the associated
        time stamps should be passed using this argument.
    par: str, PulsarParameters
        A parameter file, or :class:`~cwinpy.parfile.PulsarParameters` object
        containing the parameters with which the data was heterodyned.
    detector: str, ``lal.Detector``
        A string, or :class:`lal.Detector` object, identifying the detector
        from which the data was generated.
    window: int, 30
        The length of a window used for calculating a running median over the
        data. If set to zero the running median will just be initialised with
        zero values.
    inject: bool, False
        Set to ``True`` to add a simulated signal to the data based on the
        parameters supplied in `injpar`, or `par` if `injpar` is not given.
    injpar: str, PulsarParameters
        A parameter file name or :class:`~cwinpy.parfile.PulsarParameters`
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
    fakeasd: float, str
        A amplitude spectral density value (in 1/sqrt(Hz)) at which to
        generate simulated Gaussian noise to add to the data. Alternatively, if
        a string is passed, and that string represents a known detector, then
        the amplitude spectral density for that detector at design sensitivity
        will be used (this requires a `par` value to be included, which
        contains the source rotation frequency).
    fakeseed: (int, class:`numpy.random.Generator`), None
        A seed for the random number generator used to create the fake data
        (see :meth:`numpy.random.seed` and :class:`numpy.random.Generator`
        for more information).
    issigma: bool
        Set to ``True`` if the ``fakeasd`` value passed is actually a noise
        standard deviation value rather than an amplitude spectral density.
    bbthreshold: (str, float), "default"
        The threshold method, or value for the
        :meth:`~cwinpy.data.HeterodynedData.bayesian_blocks` function.
    bbminlength: int, 5
        The minimum length (in numbers of data points) of a chunk that the data
        can be split into by the
        :meth:`~cwinpy.data.HeterodynedData.bayesian_blocks` function. To
        perform no splitting of the data set this value to be larger than the
        total data length, e.g., ``inf``.
    bbmaxlength: int, inf
        The maximum length (in numbers of data points) of a chunk that the data
        can be split into by the
        :meth:`~cwinpy.data.HeterodynedData.bayesian_blocks` function. By
        default this is ``inf``, i.e., chunks can be as long as possible.
    remove_outliers: bool, False
        If ``True`` outliers will be found (using
        :meth:`~cwinpy.data.HeterodynedData.find_outliers`) and removed from the
        data. They will not be stored anywhere in the class.
    thresh: float, 3.5
        The modified z-score threshold for outlier removal (see
        :meth:`~cwinpy.data.HeterodynedData.find_outliers`)
    comments: str
        A string containing any comments about the data.
    earthephemeris: str, None
        The path to the Earth ephemeris used for the signal phase model.
    sunephemeris: str, None
        The path to the Sun ephemeris used for the signal phase model.
    timeephemeris: str, None
        The path to the time correction ephemeris used for the signal phase
        model.
    """

    # set some default detector color maps for plotting
    colmapdic = {"H1": "Reds", "L1": "Blues", "V1": "PuRd", "G1": "Greys"}

    # set some default plotting values
    PLOTTING_DEFAULTS = {
        "labelsize": 14,  # font size for axes tick labels
        "fontsize": 16,  # font size for axes labels
        "fontname": "Gentium",  # font name for axes labels
        "labelname": "Carlito",  # font names for axes tick labels
    }

    _metadata_slots = Series._metadata_slots + (
        "dt",
        "comments",
        "par",
        "injpar",
        "window",
        "laldetector",
        "vars",
        "bbthreshold",
        "bbminlength",
        "bbmaxlength",
        "outlier_thresh",
        "injtimes",
        "freqfactor",
        "filter_history",
        "running_median",
        "input_stds",
        "outlier_mask",
        "include_ssb",
        "include_bsb",
        "include_glitch",
        "include_fitwaves",
        "cwinpy_version",
        "heterodyne_arguments",
        "cwinpy_heterodyne_pipeline_config",
    )

    def __new__(
        cls,
        data=None,
        times=None,
        par=None,
        detector=None,
        window=30,
        inject=False,
        injpar=None,
        injtimes=None,
        freqfactor=2.0,
        fakeasd=None,
        fakeseed=None,
        issigma=False,
        bbthreshold="default",
        bbminlength=5,
        bbmaxlength=np.inf,
        remove_outliers=False,
        thresh=3.5,
        comments="",
        earthephemeris=None,
        sunephemeris=None,
        timeephemeris=None,
        **kwargs,
    ):
        stds = None  # initialise standard deviations

        # read/parse data
        if (
            isinstance(data, (str, list)) and np.array(data).dtype.type == np.str_
        ) or isinstance(data, Path):
            try:
                new = cls.read(data)
            except Exception as e:
                raise IOError("Error reading file '{}':\n{}".format(data, e))

            if new.detector is None:
                new.detector = detector
        else:
            if type(data) is not HeterodynedData:
                # use data
                hettimes = times
                if hettimes is None and data is None:
                    raise ValueError("Time stamps and/or data must be supplied")
                elif data is not None:
                    dataarray = np.atleast_2d(np.asarray(data))

                    if dataarray.shape[0] == 1:
                        dataarray = dataarray.T
                else:
                    # set data to zeros
                    dataarray = np.zeros((len(hettimes), 1), dtype=complex)

                if (
                    dataarray.shape[1] == 1
                    and dataarray.dtype == complex
                    and hettimes is not None
                ):
                    dataarray = dataarray.flatten()
                elif dataarray.shape[1] == 2 and hettimes is not None:
                    # real and imaginary components are separate
                    dataarray = dataarray[:, 0] + 1j * dataarray[:, 1]
                elif dataarray.shape[1] == 3:
                    if hettimes is None:
                        # first column of array should be times
                        hettimes = dataarray[:, 0]
                        dataarray = dataarray[:, 1] + 1j * dataarray[:, 2]
                    else:
                        # third column can be standard deviations
                        stds = dataarray[:, 2]
                        dataarray = dataarray[:, 0] + 1j * dataarray[:, 1]
                elif dataarray.shape[1] == 4:
                    if hettimes is None:
                        # first column of array should be times
                        hettimes = dataarray[:, 0]
                        stds = dataarray[:, 3]
                        dataarray = dataarray[:, 1] + 1j * dataarray[:, 2]
                    else:
                        raise ValueError("Supplied data array is the wrong shape")
                else:
                    raise ValueError("Supplied data array is the wrong shape")

                if len(hettimes) != dataarray.shape[0]:
                    raise ValueError(
                        "Supplied times is not that same length as the data"
                    )

                if hettimes is not None and times is not None:
                    if not np.array_equal(hettimes, times):
                        raise ValueError(
                            "Supplied times and times in data file are not the same"
                        )

                # check for and remove any duplicate time stamps
                utimes, uidx = np.unique(hettimes, return_index=True)

                if len(utimes) < len(hettimes):
                    warn(
                        "Data contained duplicate time stamps. Any duplicates will be removed."
                    )

                # generate TimeSeriesBase
                new = super(HeterodynedData, cls).__new__(
                    cls, dataarray[uidx], times=utimes
                )

                new.stds = None
                if stds is not None:
                    # set pre-calculated data standard deviations
                    new.stds = stds[uidx]
                    new.input_stds = True
                else:
                    new.input_stds = False

                new.detector = detector
            else:
                new = data  # already a HeterodyneData class

        # set the window size
        if new.window is None or window != new.window:
            new.window = window

        # set Bayesian Block parameters
        if new.bbminlength is None or bbminlength != new.bbminlength:
            new.bbminlength = bbminlength
        if new.bbmaxlength is None or bbmaxlength != new.bbmaxlength:
            new.bbmaxlength = bbmaxlength
        if new.bbthreshold is None or bbthreshold != new.bbthreshold:
            new.bbthreshold = bbthreshold

        # remove outliers
        new.outlier_mask = None
        if remove_outliers and fakeasd is None:
            new.remove_outliers(thresh=thresh)

        # set the (minimum) time step and sampling frequency
        try:
            _ = new.dt
        except AttributeError:
            # times do not get set in a TimeSeries if steps are irregular, so
            # manually set the time step to the minimum time difference
            if len(new) > 1:
                new.dt = np.min(np.diff(new.times))
            else:
                warn("Your data is only one data point long!")
                new.dt = None

        # don't recompute values on data that has had outliers removed (the
        # remove method already does this).
        if not remove_outliers:
            # initialise the running median
            _ = new.compute_running_median(N=new.window)

            # calculate change points (and variances)
            new.bayesian_blocks()

        # set the parameter file
        if par is not None:
            # overwrite existing par file
            new.par = par
        else:
            if not hasattr(new, "par"):
                new.par = None

        # set the frequency scale factor
        if new.freq_factor is None or freqfactor != new.freq_factor:
            # note: if reading from an ascii text file, the new object will
            # default to containing a freq_factor of 2, so this needs to be
            # overwritten if a different value is passed. Hence, the
            # "freqfactor != new.freq_factor"
            new.freq_factor = freqfactor

        # add noise, or create data containing noise
        if fakeasd is not None:
            new.add_noise(fakeasd, issigma=issigma, seed=fakeseed)

        # set solar system ephemeris files if provided
        if (
            not hasattr(new, "ephemearth")
            and not hasattr(new, "ephemsun")
            and not hasattr(new, "ephemtime")
        ):
            try:
                earthephemeris = ast.literal_eval(earthephemeris)
            except ValueError:
                pass
            try:
                sunephemeris = ast.literal_eval(sunephemeris)
            except ValueError:
                pass
            try:
                timeephemeris = ast.literal_eval(timeephemeris)
            except ValueError:
                pass

            new.set_ephemeris(earthephemeris, sunephemeris, timeephemeris)

        # set and add a simulated signal
        if bool(inject):
            # inject the signal
            if injpar is None:
                new.inject_signal(injtimes=injtimes)
            else:
                new.inject_signal(injpar=injpar, injtimes=injtimes)

        # add/update comments if given
        if comments is not None:
            if len(comments) > 0:
                new.comments = comments

        # add CWInPy version used for creation of data if not present
        if not hasattr(new, "_cwinpy_version"):
            new.cwinpy_version = cwinpy.__version__

        return new

    @property
    def cwinpy_version(self):
        """
        Return the version of CWInPy used to produce the dataset.
        """

        return self._cwinpy_version

    @cwinpy_version.setter
    def cwinpy_version(self, version):
        self._cwinpy_version = version

    @classmethod
    def read(cls, source, *args, **kwargs):
        """
        Read in a time series of data from a given file or list of files.
        Currently this supports ascii text files as described for the
        :class:`~cwinpy.data.HeterodynedData` class or HDF5 files.

        See :meth:`gwpy.timeseries.TimeSeries.read` for more information.
        """

        if isinstance(source, (str, Path)):
            datafiles = [source]
        else:
            datafiles = list(source)

        # remove any files that have zero size
        datafiles = [df for df in datafiles if Path(df).stat().st_size]
        if not datafiles:
            raise IOError("No non-empty heterodyned data files were given")

        hetdata = read_multi(lambda x: x[0], cls, datafiles[0], *args, **kwargs)

        for dfile in datafiles[1:]:
            moredata = read_multi(lambda x: x[0], cls, dfile, *args, **kwargs)
            hetdata.merge(moredata)

        return cls(data=hetdata, **kwargs)

    def write(self, target, *args, **kwargs):
        """
        Write this :class:`~cwinpy.data.HeterodynedData` object to a file.
        """

        return io_registry.write(self, target, *args, **kwargs)

    def merge(self, other, sort=True):
        """
        Merge another class:`~cwinpy.data.HeterodynedData` with the current one
        in-place. The times series will be sorted to be in ascending order if
        required.

        Parameters
        ----------
        other: class:`~cwinpy.data.HeterodynedData`
            Another class:`~cwinpy.data.HeterodynedData` object.
        sort: bool
            Sort the merged data in ascending time order.
        """

        # check compatibility
        self.is_compatible(other)

        # check data does not overlap
        start0 = self.times[0]
        end0 = self.times[-1]

        startother = other.times[0]
        endother = other.times[0]

        if start0 <= startother <= end0 or start0 <= endother <= end0:
            raise ValueError("Cannot merge overlapping data")

        M = len(self) + len(other)
        N = len(other)

        # get indices of sorted array
        if sort:
            sortidx = np.argsort(np.concatenate((self.times.value, other.times.value)))

        # resize times series
        self.resize(M, refcheck=False)
        self.value[-N:] = other.value
        if sort:
            self.value[:] = self.value[sortidx]

        # resize of time values
        try:
            self.xindex.resize(M, refcheck=False)
        except ValueError as exc:
            if "cannot resize" in str(exc):
                self._xindex = self.xindex.copy()
                self._xindex.resize((M,))
            else:
                raise
        self.xindex[-N:] = other._xindex
        if sort:
            self.xindex[:] = self.xindex[sortidx]

        # resize the variances if given
        if self.vars is not None and other.vars is not None:
            self._vars.resize(M, refcheck=False)
            self._vars[-N:] = other.vars
            if sort:
                self._vars[:] = self._vars[sortidx]

        # resize the injection times if given
        if self.injtimes is not None:
            # sort injection times
            injtimes = np.vstack((self._injtimes, other._injtimes))
            injtimes = np.array(sorted(zip(injtimes[:, 0], injtimes[:, 1])))

            self._injtimes.resize(injtimes.shape, refcheck=False)
            self._injtimes[:, :] = injtimes

        if other.heterodyne_arguments is not None:
            # combine Heterodyne arguments if given
            self.heterodyne_arguments = other.heterodyne_arguments

    def is_compatible(self, other):
        """
        Check if another class:`~cwinpy.data.HeterodynedData` object is
        "compatible", i.e., contains data for the same detector, pulsar and
        frequency scale, as the current object.

        Parameters
        ----------
        other: class:`~cwinpy.data.HeterodynedData`
            Another class:`~cwinpy.data.HeterodynedData` object.

        Returns
        -------
        bool:
            Returns True if compatible otherwise it will raise an exception.
        """

        # check compatible detectors
        if self.detector is not None and other.detector is not None:
            if self.detector != other.detector:
                raise ValueError("Incompatible detectors")

        # check for compatible pulsars
        if self.par is not None and other.par is not None:
            if get_psr_name(self.par) != get_psr_name(other.par):
                raise ValueError("Incompatible pulsars")

        # check for compatible frequency scale factors
        if self.freq_factor is not None and other.freq_factor is not None:
            if self.freq_factor != other.freq_factor:
                raise ValueError("Incompatible frequency factors")

        # check variances either are all set or not set
        if self.input_stds != other.input_stds:
            raise ValueError("Incompatible setting of variances")

        # check injection times are all set or not set
        if (self.injtimes is not None and other.injtimes is None) or (
            self.injtimes is None and other.injtimes is not None
        ):
            raise ValueError("Incompatible injection times")

        return True

    @property
    def dt(self):
        try:
            return self.dx
        except AttributeError:
            return self._dt

    @dt.setter
    def dt(self, dt):
        """
        Overload the default setting of the time step in a TimeSeries, so that
        it does not delete non-uniform time values.
        """

        self._dt = dt

    @property
    def window(self):
        """The running median window length."""

        try:
            return self._window
        except AttributeError:
            return None

    @window.setter
    def window(self, window):
        if isinstance(window, (int, np.int64)):
            if window < 2 and window != 0:
                raise ValueError("Window length must be greater than 2")
            else:
                self._window = int(window)
        else:
            raise TypeError("Window must be an integer")

    @property
    def comments(self):
        """Any comments on the data"""

        try:
            return self._comments
        except AttributeError:
            return None

    @comments.setter
    def comments(self, comment):
        if comment is None:
            self._comments = None
        elif isinstance(comment, str):
            self._comments = comment
        else:
            raise TypeError("Data comment should be a string")

    @property
    def data(self):
        """
        A :class:`numpy.ndarray` containing the heterodyned data.
        """

        if self.outlier_mask is not None:
            return self.value[self.outlier_mask]
        else:
            return self.value

    @property
    def times(self):
        if self.outlier_mask is not None:
            return super(HeterodynedData, self).times[self.outlier_mask]
        else:
            return super(HeterodynedData, self).times

    @property
    def tottime(self):
        """
        The total time (in seconds) of the data.
        """

        return self.times[-1] - self.times[0]

    @property
    def par(self):
        try:
            return self._par
        except AttributeError:
            return None

    @par.setter
    def par(self, par):
        self._par = self._parse_par(par)

    @property
    def injpar(self):
        try:
            return self._injpar
        except AttributeError:
            return None

    @injpar.setter
    def injpar(self, par):
        self._injpar = self._parse_par(par)

    def _parse_par(self, par):
        """
        Parse a pulsar parameter file or
        :class:`~cwinpy.parfile.PulsarParameters` object.

        Parameters
        ----------
        par: (str, PulsarParameters)
            A file or object containing a set of pulsar parameters.

        Returns
        -------
        PulsarParameters
        """

        if par is not None:
            if isinstance(par, PulsarParameters):
                return par
            elif isinstance(par, (str, Path)):
                if is_par_file(par):
                    newpar = PulsarParameters(par)
                else:
                    raise IOError("Could not read in pulsar parameter file")
            else:
                raise TypeError("'par' is not a recognised type")
        else:
            newpar = None

        return newpar

    @property
    def detector(self):
        """The name of the detector from which the data came."""

        try:
            return self.channel.ifo
        except AttributeError:
            return None

    @property
    def laldetector(self):
        """
        The :class:`lal.Detector` containing the detector's response and
        location.
        """

        try:
            return self._laldetector
        except AttributeError:
            return None

    @laldetector.setter
    def laldetector(self, detector):
        if isinstance(detector, lal.Detector):
            self._laldetector = detector

    @detector.setter
    def detector(self, detector):
        if isinstance(detector, lal.Detector):
            self.channel = Channel("{}:".format(detector.frDetector.prefix))
            self.laldetector = detector
        elif isinstance(detector, str):
            self.channel = Channel("{}:".format(detector))

            try:
                self.laldetector = lalpulsar.GetSiteInfo(detector)
            except RuntimeError:
                raise ValueError("Could not set LAL detector!")

    @property
    def running_median(self):
        """A :class:`~numpy.ndarray` containing the running median of the data."""

        try:
            return self._running_median
        except AttributeError:
            return None

    @running_median.setter
    def running_median(self, rm):
        self._running_median = rm

    def compute_running_median(self, N=30):
        """
        Calculate a running median from the data with the real and imaginary
        parts separately. The running median will be calculated using a window
        of samples of a given number. This does not account for any gaps in the
        data, so could contain discontinuities.

        Parameters
        ----------
        N: int, 30
            The window length of the running median. Defaults to 30 points. If
            set to 0 the running median will be initialised as an array of
            zeros.

        Returns
        -------
        array_like
            A :class:`numpy.ndarray` array containing the data with the
            running median subtracted.
        """

        if N < 2 and N != 0:
            raise ValueError("The running median window must be greater than 1")

        self.running_median = TimeSeriesBase(
            np.zeros(len(self), dtype=complex), times=self.times
        )

        if N > 0:
            median_filter(
                self.data.real, size=N, output=self._running_median.real, mode="mirror"
            )
            median_filter(
                self.data.imag, size=N, output=self._running_median.imag, mode="mirror"
            )

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

        return self.data - self.running_median.value

    @property
    def vars(self):
        """
        The variances of the data points.
        """

        try:
            if self.outlier_mask is None:
                return self._vars
            else:
                return self._vars[self.outlier_mask]
        except (AttributeError, TypeError):
            return None

    @vars.setter
    def vars(self, vars):
        if vars is not None:
            if isinstance(vars, float):
                if vars <= 0.0:
                    raise ValueError("Variance cannot be negative")
                tmpmsk = None
                if self.outlier_mask is not None:
                    tmpmsk = np.copy(self.outlier_mask)

                self._vars = vars * np.ones(len(self))

                if tmpmsk is not None:
                    self.outlier_mask = tmpmsk  # reset mask
            else:
                if len(vars) != len(self):
                    raise ValueError("Supplied variances are wrong length")

                if self.outlier_mask is None:
                    self._vars = np.asarray(vars)
                else:
                    tmpmsk = np.copy(self.outlier_mask)
                    self.outlier_mask = None
                    self._vars = np.zeros(len(self))
                    self._vars[tmpmsk] = vars
                    self.outlier_mask = tmpmsk  # reset mask
        else:
            self._vars = None

    @property
    def stds(self):
        """
        The standard deviations of the data points.
        """

        try:
            if self._vars is None:
                return None
            else:
                return np.sqrt(self.vars)
        except AttributeError:
            return None

    @stds.setter
    def stds(self, stds):
        if stds is not None:
            self.vars = stds**2
        else:
            self.vars = None

    @property
    def input_stds(self):
        """
        A boolean stating whether the standard deviations where provides as an
        input.
        """

        try:
            return self._input_stds
        except AttributeError:
            return False

    @input_stds.setter
    def input_stds(self, inputstds):
        self._input_stds = bool(inputstds)

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

        if self.vars is not None:
            if len(self.vars) == len(self):
                return self.vars

        # subtract running median from the data
        datasub = self.subtract_running_median()

        if change_points is None and len(self._change_point_indices_and_ratios) == 0:
            # return the (sample) variance (hence 'ddof=1')
            self.vars = np.full(
                len(self), np.hstack((datasub.real, datasub.imag)).var(ddof=1)
            )
        else:
            tmpvars = np.zeros(len(self))

            if change_points is not None:
                cps = np.concatenate(
                    ([0], np.asarray(change_points), [len(datasub)])
                ).astype("int")
            else:
                if len(self.change_point_indices) == 1:
                    cps = np.array([0, len(datasub)], dtype=int)
                else:
                    cps = np.concatenate(
                        (self.change_point_indices, [len(datasub)])
                    ).astype("int")

            if self.stds is None:
                self.stds = np.zeros(len(self))

            for i in range(len(cps) - 1):
                if cps[i + 1] < 1 or cps[i + 1] > len(datasub):
                    raise ValueError("Change point index is out of bounds")

                if cps[i + 1] <= cps[i]:
                    raise ValueError("Change point order is wrong")

                datachunk = datasub[cps[i] : cps[i + 1]]

                # get (sample) variance of chunk
                tmpvars[cps[i] : cps[i + 1]] = np.hstack(
                    (datachunk.real, datachunk.imag)
                ).var(ddof=1)

            self.vars = tmpvars

        return self.vars

    def inject_signal(self, injpar=None, injtimes=None, inject=True, **kwargs):
        """
        Inject a simulated signal into the data.

        Parameters
        ----------
        injpar: (str, PulsarParameters)
            A parameter file or object containing the parameters for the
            simulated signal.
        injtimes: list
            A list of pairs of time values between which to inject the signal.
        inject: bool
            If True (default) the simulated signal will be generated and added
            to the data. If False the signal will be created and returned, but
            not added into the data.
        """

        # save any kwargs
        if not hasattr(self, "_inj_kwargs") or len(kwargs) > 1:
            self._inj_kwargs = kwargs.copy()

        # create the signal to inject
        if injpar is None:
            self.injpar = self.par
            signal = self.make_signal(**self._inj_kwargs)
        else:
            self.injpar = injpar
            signal = self.make_signal(signalpar=self.injpar, **self._inj_kwargs)

        # set the times between which the injection will be added
        self.injtimes = injtimes

        # initialise the injection to zero
        inj_data = TimeSeriesBase(
            np.zeros_like(self.data), times=self.times, channel=self.channel
        )

        for timerange in self.injtimes:
            timeidxs = np.arange(len(self))[
                (self.times.value >= timerange[0]) & (self.times.value <= timerange[1])
            ]
            inj_data[timeidxs] = signal[timeidxs]

        # add injection to data
        if inject:
            self += inj_data

            # (re)compute the running median
            _ = self.compute_running_median(N=self.window)
        else:
            return inj_data

    @property
    def injtimes(self):
        """
        A list of times at which an injection was added to the data.
        """

        if not hasattr(self, "_injtimes"):
            self._injtimes = None
        return self._injtimes

    @injtimes.setter
    def injtimes(self, injtimes):
        if injtimes is None:
            # include all time
            timelist = np.array([[self.times[0].value, self.times[-1].value]])
        else:
            timelist = injtimes

        try:
            timelist = np.atleast_2d(timelist)
        except Exception as e:
            raise ValueError("Could not parse list of injection times: {}".format(e))

        for timerange in timelist:
            if timerange[0] >= timerange[1]:
                raise ValueError("Injection time ranges are incorrect")

        self._injtimes = timelist

    @property
    def injection(self):
        """
        Return a boolean to describe whether the data contains a user-generated
        injection or not.
        """

        if self.injpar is not None and self.injtimes is not None:
            return True
        else:
            return False

    def set_ephemeris(self, earth=None, sun=None, time=None):
        """
        Set the solar system ephemeris and time correction files.

        Parameters
        ----------
        earth: str, dict, None
            The Earth ephemeris file used for the phase model. Defaults to
            None, in which case the ephemeris files will be determined from the
            pulsar parameter file information. If a dictionary is passed, it
            should be keyed to ephemeris types (e.g., DE405) and point to the
            equivalent file. The type in the par file will be used to determine
            which key to use.
        sun: str, dict, None
            The Sun ephemeris file used for the phase model. Defaults to
            None, in which case the ephemeris files will be determined from the
            pulsar parameter file information. If a dictionary is passed, it
            should be keyed to ephemeris types (e.g., DE405) and point to the
            equivalent file. The type in the par file will be used to determine
            which key to use.
        time: str, dict, None
            The time correction ephemeris file used for the phase model.
            Defaults to None, in which case the ephemeris files will be
            determined from the pulsar parameter file information. If a
            dictionary is passed, it should be keyed to unit type (e.g., TCB)
            and point to the equivalent file. The type in the par file will be
            used to determine which key to use.
        """

        efiles = [earth, sun, time]

        efiles = []

        # check for dictionaries
        for ef in [earth, sun]:
            if isinstance(ef, dict):
                efiles.append(ef.get(self.par["EPHEM"], None))
            else:
                efiles.append(ef)

        if isinstance(time, dict):
            efiles.append(time.get(self.par["UNIT"], None))
        else:
            efiles.append(time)

        # check ephemeris files
        for ef in efiles:
            if ef is None:
                continue
            if isinstance(ef, str):
                if not os.path.isfile(ef):
                    raise IOError("Ephemeris file '{}' does not exist".format(ef))
            else:
                raise TypeError("Ephemeris file is not a string")

        self.ephemearth = efiles[0]
        self.ephemsun = efiles[1]
        self.ephemtime = efiles[2]

    @property
    def injection_data(self):
        """
        The pure simulated signal that was added to the data.
        """

        if self.injection:
            return self.inject_signal(
                injpar=self.injpar, injtimes=self.injtimes, inject=False
            ).value
        else:
            return None

    @property
    def injection_snr(self):
        """
        Return the optimal signal-to-noise ratio using the pure injected signal
        and true noise calculated using:

        .. math::

           \\rho = \\sqrt{\\sum_i \\left(\\left[\\frac{\\Re{(s_i)}}{\\sigma_i}\\right]^2 +
           \\left[\\frac{\\Im{(s_i)}}{\\sigma_i}\\right]^2\\right)}

        where and :math:`s` is the pure signal and :math:`\\sigma` is the
        estimated noise standard deviation.
        """

        if not self.injection:
            return None

        return np.sqrt(
            ((self.injection_data.real / self.stds) ** 2).sum()
            + ((self.injection_data.imag / self.stds) ** 2).sum()
        )

    def make_signal(self, signalpar=None, **kwargs):
        """
        Make a signal at the data time stamps given a parameter file.

        Note that the antenna response applied to the signal will be that after
        averaging over the data time step (e.g., if a time step of 30 minutes
        is used then the antenna response will be the average of ±15 minutes
        around the timestamp). However, it assumes other variations are slower,
        so it does not average out any differences in the phase evolution
        between the heterodyne parameters and any injected parameters (if
        specified as different) and just produced a point estimate at the data
        timestamp.

        Note
        ----
        If generating a simulated signal in IPython or a Jupyter notebook, this
        function can be very slow (taking minutes compared to a fraction of a
        second), due to redirections of ``stdout``/``stderr`` in the
        SWIG-wrapped ``LIGOTimeGPS`` class. To avoid this the call to
        :meth:`~cwinpy.data.HeterodynedData.make_signal` should be either done
        within a context manager, e.g.,:

        .. code-block:: python

            import lal

            hetdata = HeterodynedData(...)

            with lal.no_swig_redirect_standard_output_error():
                hetdata.make_signal(...)

        or by globally disabling redirection with:

        .. code-block:: python

            import lal

            lal.swig_redirect_standard_output_error(True)

            hetdata = HeterodynedData(...)
            hetdata.make_signal(...)

        Parameters
        ----------
        signalpar: str, PulsarParameters
            A parameter file or object containing the parameters for the
            simulated signal.

        Returns
        -------
        array_like
            A complex :class:`numpy.ndarray` containing the signal.
        """

        if self.par is None:
            raise ValueError(
                "To perform an injection a parameter file must be supplied"
            )

        if self.detector is None:
            raise ValueError("To perform an injection a detector must be supplied")

        from .signal import HeterodynedCWSimulator

        # initialise the injection
        het = HeterodynedCWSimulator(
            self.par,
            self.detector,
            times=self.times,
            earth_ephem=self.ephemearth,
            sun_ephem=self.ephemsun,
            usetempo2=kwargs.get("usetempo2", False),
        )

        # get the injection
        if signalpar is None:
            # use self.par for the injection parameters
            signal = het.model(freqfactor=self.freq_factor)
        else:
            signal = het.model(
                signalpar,
                updateSSB=kwargs.get("updateSSB", True),
                updateBSB=kwargs.get("updateBSB", True),
                updateglphase=kwargs.get("updateglphase", True),
                updatefitwaves=kwargs.get("updatefitwaves", True),
                freqfactor=self.freq_factor,
            )

        return TimeSeriesBase(signal, times=self.times, channel=self.channel)

    def signal_snr(self, signalpar):
        """
        Get the signal-to-noise ratio of a signal based on the supplied
        parameter file.

        Parameters
        ----------
        signalpar: str, PulsarParameters
            A parameter file or object containing the parameters for the
            simulated signal.

        Returns
        -------
        float:
            The signal-to-noise ratio.
        """

        # generate the signal
        signal = self.make_signal(signalpar=signalpar)

        # get signal-to-noise ratio based on estimated data standard deviation
        return np.sqrt(
            ((signal.real / self.stds) ** 2).sum()
            + ((signal.imag / self.stds) ** 2).sum()
        ).value

    @property
    def freq_factor(self):
        """
        The scale factor of the source rotation frequency with which the data
        was heterodyned.
        """

        try:
            return self._freqfactor
        except AttributeError:
            return None

    @freq_factor.setter
    def freq_factor(self, freqfactor):
        if not isinstance(freqfactor, (float, int)):
            raise TypeError("Frequency scale factor must be a number")

        if freqfactor <= 0.0:
            raise ValueError("Frequency scale factor must be a positive number")

        self._freqfactor = float(freqfactor)

    @property
    def freqfactor(self):
        """
        Alias for ``freq_factor`` to be consistent with input keyword
        arguments.
        """

        return self.freq_factor

    @freqfactor.setter
    def freqfactor(self, freqfactor):
        self.freq_factor = freqfactor

    def add_noise(self, asd, issigma=False, seed=None):
        """
        Add white Gaussian noise to the data based on a supplied one-sided
        noise amplitude spectral density (in 1/sqrt(Hz)).

        If generating noise from a given detector's design curve, a frequency
        is required, which itself requires a pulsar parameter file to have been
        supplied.

        Parameters
        ----------
        asd: float, str
            The noise amplitude spectral density (1/sqrt(Hz)) at which to
            generate the Gaussian noise, or a string containing a valid
            detector name for which the design sensitivity ASD can be used, or
            a file containing an amplitude spectral density frequency series.
        issigma: bool, False
            If ``issigma`` is ``True`` then the value passed to `asd` is assumed
            to be a dimensionless time domain standard deviation for the noise
            level rather than an amplitude spectral density.
        seed: int, :class:`numpy.random.Generator`, None
            A seed for the random number generator used to create the fake data
            (see :meth:`numpy.random.seed` and :class:`numpy.random.Generator`
            for more information).
        """

        if isinstance(asd, str):
            import lalsimulation as lalsim

            if self.par is None:
                raise AttributeError(
                    "A source parameter file containing a frequency is required"
                )

            # check a frequency is available
            freqs = self.par["F"]
            if freqs is None:
                raise ValueError(
                    "Heterodyne parameter file contains no " "frequency value"
                )

            # check if the str is a file or not
            if os.path.isfile(asd):
                # frequency series to contain the PSD
                psdfs = lal.CreateREAL8FrequencySeries(
                    "",
                    lal.LIGOTimeGPS(1000000000),  # dummy epoch
                    self.freq_factor * freqs[0],  # frequency to find
                    0.1,  # dummy delta f
                    lal.HertzUnit,
                    2,  # need two points as last element is set to zero
                )

                # read PSD from ASD file
                try:
                    _ = lalsim.SimNoisePSDFromFile(psdfs, psdfs.f0, asd)
                except Exception as e:
                    raise RuntimeError("Problem getting ASD from file: {}".format(e))

                # convert to ASD
                asdval = np.sqrt(psdfs.data.data[0])
            else:
                # check is str is a detector alias
                aliases = {
                    "AV": ["VIRGO", "V1", "ADV", "ADVANCEDVIRGO", "AV"],
                    "AL": [
                        "H1",
                        "L1",
                        "LHO",
                        "LLO",
                        "ALIGO",
                        "ADVANCEDLIGO",
                        "AL",
                        "AH1",
                        "AL1",
                    ],
                    "IL": ["IH1", "IL1", "INITIALLIGO", "IL"],
                    "IV": ["iV1", "INITIALVIRGO", "IV"],
                    "G1": ["G1", "GEO", "GEOHF"],
                    "IG": ["IG", "GEO600", "INITIALGEO"],
                    "T1": ["T1", "TAMA", "TAMA300"],
                    "K1": ["K1", "KAGRA", "LCGT"],
                    "ET": [
                        "E0",
                        "E1",
                        "E2",
                        "E3",
                        "ET0",
                        "ET1",
                        "ET2",
                        "ET3",
                        "ET",
                        "EINSTEINTELESCOPE",
                    ],
                }

                # set mapping of detector names to lalsimulation PSD functions
                simmap = {
                    "AV": lalsim.SimNoisePSDAdvVirgo,  # advanced Virgo
                    "AL": PSDwrapper(
                        lalsim.SimNoisePSDaLIGOaLIGODesignSensitivityT1800044
                    ),  # aLIGO
                    "IL": lalsim.SimNoisePSDiLIGOSRD,  # iLIGO
                    "IV": lalsim.SimNoisePSDVirgo,  # iVirgo
                    "IG": lalsim.SimNoisePSDGEO,  # GEO600
                    "G1": lalsim.SimNoisePSDGEOHF,  # GEOHF
                    "T1": lalsim.SimNoisePSDTAMA,  # TAMA
                    "K1": lalsim.SimNoisePSDKAGRA,  # KAGRA
                    "ET": PSDwrapper(lalsim.SimNoisePSDEinsteinTelescopeP1600143),  # ET
                }

                # set detector if not already set
                if self.channel is None:
                    namemap = {
                        "H1": ["H1", "LHO", "IH1", "AH1"],
                        "L1": ["L1", "LLO", "IL1", "AL1"],
                        "V1": [
                            "V1",
                            "VIRGO",
                            "ADV",
                            "ADVANCEDVIRGO",
                            "AV",
                            "IV1",
                            "INITIALVIRGO",
                            "IV",
                        ],
                        "G1": ["G1", "GEO", "GEOHF", "IG", "GEO600", "INITIALGEO"],
                        "T1": ["T1", "TAMA", "TAMA300"],
                        "K1": ["K1", "KAGRA", "LCGT"],
                        "E0": ["E0", "ET0", "EINSTEINTELESCOPE0"],
                        "E1": ["E1", "ET1", "EINSTEINTELESCOPE1", "ET"],
                        "E2": ["E2", "ET2", "EINSTEINTELESCOPE2"],
                        "E3": ["E3", "ET3", "EINSTEINTELESCOPE3"],
                    }

                    nameval = None
                    for dkey in namemap:
                        if asd.upper() in namemap[dkey]:
                            nameval = dkey
                            self.channel = Channel("{}:".format(dkey))
                            break

                    if nameval is None:
                        raise ValueError(
                            "Detector '{}' is not a known detector alias".format(asd)
                        )

                # check if string is valid
                detalias = None
                for dkey in aliases:
                    if asd.upper() in aliases[dkey]:
                        detalias = dkey
                        break

                if detalias is None:
                    raise ValueError(
                        "Detector '{}' is not as known detector alias".format(asd)
                    )

                freqs = self.par["F"]
                if freqs is None:
                    raise ValueError(
                        "Heterodyne parameter file contains no frequency value"
                    )

                # set amplitude spectral density value
                asdval = np.sqrt(simmap[detalias](self.freq_factor * freqs[0]))

            # convert to time domain standard deviation
            if self.dt is None:
                raise ValueError(
                    "No time step present. Does your data only consist of one value?"
                )

            sigmaval = 0.5 * asdval / np.sqrt(self.dt.value)
        elif isinstance(asd, float):
            if issigma:
                sigmaval = asd
            else:
                if self.dt is None:
                    raise ValueError(
                        "No time step present. Does your data "
                        "only consist of one value?"
                    )

                sigmaval = 0.5 * asd / np.sqrt(self.dt.value)
        else:
            raise TypeError("ASD must be a float or a string with a detector name.")

        # set noise seed
        if isinstance(seed, np.random.Generator):
            rstate = seed
        else:
            rstate = np.random.default_rng(seed)

        # get noise for real and imaginary components
        noise = TimeSeriesBase(
            (
                rstate.normal(loc=0.0, scale=sigmaval, size=len(self))
                + 1j * rstate.normal(loc=0.0, scale=sigmaval, size=len(self))
            ),
            times=self.times,
        )

        self += noise

        # (re)compute the running median
        _ = self.compute_running_median(N=self.window)

        # (re)compute change points (and variances)
        self.bayesian_blocks()

        # set noise based on provided value
        self.stds = np.full(len(self), sigmaval)

        # standard devaitions have been provided rather than calculated
        self.input_stds = True

    def bayesian_blocks(self, **kwargs):
        """
        Apply a Bayesian-Block-style algorithm to cut the data (after
        subtraction of a running median) up into chunks with different
        statistical properties using the formalism described in Section 2.4 of
        [5]_. Within each chunk the data should be well described by a single
        Gaussian distribution with zero mean.

        Splitting of the data relies on a threshold on the natural logarithm of
        the odds comparing the hypothesis that the data is best described by
        two different contiguous zero mean Gaussian distributions with
        different unknown variances to the hypothesis that the data is
        described by a single zero mean Gaussian with unknown variance. The
        former hypothesis is a compound hypothesis consisting of the sum of
        evidences for the split in the data at any point.

        The ``'default'`` threshold for splitting is empirically derived in
        [5]_ for the cases that the prior odds between the two hypotheses is
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
        threshold: str, float
            A string giving the method for determining the threshold for
            splitting the data (described above), or a value of the threshold.
        minlength: int
            The minimum length that a chunk can be split into. Defaults to 5.
        maxlength: int
            The maximum length that a chunk can be split into. Defaults to inf.
        """

        # chop up the data (except if minlength is greater than the data length)
        self._change_point_indices_and_ratios = []

        if kwargs.get("threshold", None) is not None:
            self.bbthreshold = kwargs.get("threshold")
        elif self.bbthreshold is None:
            self.bbthreshold = "default"

        if kwargs.get("minlength", None) is not None:
            self.bbminlength = kwargs.get("minlength")
        elif self.bbminlength is None:
            self.bbminlength = 5

        if kwargs.get("maxlength", None) is not None:
            self.bbmaxlength = kwargs.get("maxlength")
        elif self.bbmaxlength is None:
            self.bbmaxlength = np.inf

        if self.bbminlength < len(self):
            self._chop_data(self.subtract_running_median())

            # sort the indices
            self._change_point_indices_and_ratios = sorted(
                self._change_point_indices_and_ratios
            )

            # if any chunks are longer than maxlength, then split them while
            # making sure to not leave chunks smaller than minlength
            if self.bbmaxlength < len(self):
                insertcps = []
                cppos = 0
                for clength in self.chunk_lengths:
                    cl = clength
                    chunkcount = 0
                    while cl > self.bbmaxlength:
                        chunkcount += self.bbmaxlength
                        if cl < (self.bbminlength + self.bbmaxlength):
                            # split final chunk to leave at least minlength
                            insertcps.append((cppos + clength - self.bbminlength, 0))
                        else:
                            insertcps.append((cppos + chunkcount, 0))

                        cl -= self.bbmaxlength

                    cppos += clength

                self._change_point_indices_and_ratios += insertcps
                self._change_point_indices_and_ratios = sorted(
                    self._change_point_indices_and_ratios
                )

        # (re)calculate the variances for each chunk
        if not self.input_stds:
            _ = self.compute_variance(N=self.window)

    @property
    def bbthreshold(self):
        """
        The threshold method/value for cutting the data in the Bayesian Blocks
        algorithm.
        """

        try:
            return self._bbthreshold
        except AttributeError:
            return None

    @bbthreshold.setter
    def bbthreshold(self, thresh):
        if isinstance(thresh, str):
            if thresh.lower() not in ["default", "trials"]:
                raise ValueError("Threshold '{}' is not a valid type".format(thresh))
        elif not isinstance(thresh, float) and thresh is not None:
            raise ValueError("Threshold '{}' is not a valid type".format(thresh))

        self._bbthreshold = thresh

    @property
    def bbminlength(self):
        """
        The minimum length of a chunk that the data can be split into by
        the Bayesian Blocks algorithm.
        """

        try:
            return self._bbminlength
        except AttributeError:
            return None

    @bbminlength.setter
    def bbminlength(self, minlength):
        if minlength is None:
            self._bbminlength = None
            return

        if not isinstance(minlength, (int, np.int64)) and not np.isinf(minlength):
            raise TypeError("Minimum chunk length must be an integer")

        if not np.isinf(minlength):
            if minlength < 1:
                raise ValueError("Minimum chunk length must be a positive integer")

        self._bbminlength = (
            minlength if not isinstance(minlength, np.int64) else int(minlength)
        )

    @property
    def bbmaxlength(self):
        """
        The maximum length of a data chunk.
        """

        try:
            return self._bbmaxlength
        except AttributeError:
            return None

    @bbmaxlength.setter
    def bbmaxlength(self, maxlength):
        if maxlength is None:
            self._bbmaxlength = None
            return

        if maxlength < self.bbminlength:
            raise ValueError(
                "Maximum chunk length must be greater than the minimum chunk length."
            )

        self._bbmaxlength = maxlength

    @property
    def change_point_indices(self):
        """
        Return a list of indices of statistical change points in the data.
        """

        if len(self._change_point_indices_and_ratios) == 0:
            return [0]
        else:
            return [0] + [cps[0] for cps in self._change_point_indices_and_ratios]

    @property
    def change_point_ratios(self):
        """
        Return a list of the log marginal likelihood ratios for the statistical
        change points in the data.
        """

        if len(self._change_point_indices_and_ratios) == 0:
            return [-np.inf]
        else:
            return [-np.inf] + [cps[1] for cps in self._change_point_indices_and_ratios]

    @property
    def chunk_lengths(self):
        """
        A list with the lengths of the chunks into which the data has been
        split.
        """

        if len(self._change_point_indices_and_ratios) == 0:
            return [len(self)]
        else:
            return np.diff(np.concatenate((self.change_point_indices, [len(self)])))

    @property
    def num_chunks(self):
        """
        The number of chunks into which the data has been split.
        """

        if len(self.change_point_indices) == 0:
            return 1
        else:
            return len(self.change_point_indices)

    def _chop_data(self, data, startidx=0):
        # find change point (don't split if data is zero)
        if allzero(self.subtract_running_median()):
            lratio, cpidx, ntrials = (-np.inf, 0, 1)
        else:
            lratio, cpidx, ntrials = self._find_change_point(data, self.bbminlength)

        # set the threshold
        if isinstance(self.bbthreshold, float):
            thresh = self.bbthreshold
        elif self.bbthreshold.lower() == "default":
            # default threshold for data splitting
            thresh = 4.07 + 1.33 * np.log10(len(data))
        elif self.bbthreshold.lower() == "trials":
            # assign equal prior probability for each hypothesis
            thresh = np.log(ntrials)
        else:
            raise ValueError("threshold is not recognised")

        if lratio > thresh:
            # split the data at the change point
            self._change_point_indices_and_ratios.append((cpidx + startidx, lratio))

            # split the data and check for another change point
            chunk1 = data[0:cpidx]
            chunk2 = data[cpidx:]

            self._chop_data(chunk1, startidx=startidx)
            self._chop_data(chunk2, startidx=(cpidx + startidx))

    @staticmethod
    @jit(nopython=True)
    def _find_change_point(subdata, minlength):
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

        if len(subdata) < 2 * minlength:
            return (-np.inf, 0, 1)

        dlen = len(subdata)
        datasum = (np.abs(subdata) ** 2).sum()

        # calculate the evidence that the data is drawn from a zero mean
        # Gaussian with a single unknown standard deviation
        logsingle = (
            -lal.LN2 - dlen * lal.LNPI + logfactorial(dlen - 1) - dlen * np.log(datasum)
        )

        lsum = dlen - 2 * minlength + 1
        logtot = -np.inf

        logdouble = np.zeros(lsum)

        sumforwards = (np.abs(subdata[:minlength]) ** 2).sum()
        sumbackwards = (np.abs(subdata[minlength:]) ** 2).sum()

        # go through each possible splitting of the data in two
        for i in range(lsum):
            if allzero(subdata[: minlength + i]) or allzero(subdata[minlength + i :]):
                # do this to avoid warnings about np.log(0.0)
                logdouble[i] = -np.inf
            else:
                dlenf = minlength + i
                dlenb = dlen - (minlength + i)

                logf = (
                    -lal.LN2
                    - dlenf * lal.LNPI
                    + logfactorial(dlenf - 1)
                    - dlenf * np.log(sumforwards)
                )
                logb = (
                    -lal.LN2
                    - dlenb * lal.LNPI
                    + logfactorial(dlenb - 1)
                    - dlenb * np.log(sumbackwards)
                )

                # evidence for that split
                logdouble[i] = logf + logb

            adval = np.abs(subdata[minlength + i]) ** 2
            sumforwards += adval
            sumbackwards -= adval

            # evidence for *any* split
            logtot = np.logaddexp(logtot, logdouble[i])

        # change point (maximum of the split evidences)
        cp = logdouble.argmax() + minlength

        # ratio of any change point compared to no splits
        logratio = logtot - logsingle

        return (logratio, cp, lsum)

    def find_outliers(self, thresh=3.5):
        """
        Find, and return the indices of, and "outliers" in the data. This is a
        modified version of the median-absolute-deviation (MAD) function from
        [6]_, using the algorithm of [7]_.

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
        """

        if not isinstance(thresh, float):
            raise TypeError("Threshold must be a float")
        else:
            if thresh <= 0.0:
                raise ValueError("Threshold must be a positive number")

        modzscore = []

        # reset mask to show all points
        self._outlier_mask = None

        for points in [self.data.real, self.data.imag]:
            median = np.median(points)
            diff = np.abs(
                points - median
            )  # only 1d data, so different from https://stackoverflow.com/a/22357811/1862861
            mad = np.median(diff)
            modzscore.append(0.6745 * diff / mad)

        # return boolean array of real or imaginary indices above the threshold
        return (modzscore[0] > thresh) | (modzscore[1] > thresh)

    def _not_outliers(self, thresh):
        """
        Get an array of indexes of points that are not outliers as identified
        by :meth:`cwinpy.data.HeterodynedData.find_outliers`.
        """

        oidx = ~self.find_outliers(thresh=thresh)
        return np.arange(len(self))[oidx]

    def remove_outliers(self, thresh=3.5):
        """
        Remove any outliers from the object using the method described in
        :meth:`cwinpy.data.HeterodynedData.find_outliers`.

        Parameters
        ----------
        thresh: float
        """

        if self.outlier_thresh is None:
            self.outlier_thresh = thresh

        idx = ~self.find_outliers(thresh=self.outlier_thresh)

        if not np.all(idx):
            self.outliers_removed = True
            self.remove(idx)

    def remove(self, idx):
        """
        Create a mask to effectively remove values at given indices from
        the object. This will recalculate the Bayesian Blocks data splitting
        and variances if required.

        Parameters
        ----------
        idx: int, array_like
            A list of indices to remove.
        """

        try:
            self.outlier_mask = idx

            if self.outlier_mask is None:
                return

            # recalculate running median, Bayesian Blocks and variances
            _ = self.compute_running_median(N=self.window)

            self.bayesian_blocks()

        except Exception as e:
            raise RuntimeError("Problem removing elements from data:\n{}".format(e))

    @property
    def outlier_mask(self):
        """
        Masking array to remove outliers.
        """

        try:
            return self._outlier_mask
        except AttributeError:
            return None

    @outlier_mask.setter
    def outlier_mask(self, mask):
        self._outlier_mask = None  # reset mask

        if mask is None:
            return

        idx = np.asarray(mask)
        if idx.dtype == int:
            zidx = np.ones(len(self), dtype=bool)
            zidx[idx] = False
        elif idx.dtype == bool:
            if len(idx) != len(self):
                raise ValueError("Outlier mask is the wrong size")
            else:
                zidx = idx
        else:
            raise TypeError("Outlier mask is the wrong type")

        if np.all(zidx):
            self._outlier_mask = None
        else:
            self._outlier_mask = zidx

    @property
    def outlier_thresh(self):
        """
        The modified z-score threshold for removing outliers (see
        :meth:`~cwinpy.data.HeterodynedData.find_outliers`).
        """

        try:
            thresh = self._outlier_thresh
        except AttributeError:
            thresh = None

        return thresh

    @outlier_thresh.setter
    def outlier_thresh(self, thresh):
        if not isinstance(thresh, (float, int)) and thresh is not None:
            raise TypeError("Outlier threshold must be a number")

        self._outlier_thresh = thresh

    @property
    def outliers_removed(self):
        """
        Return a boolean stating whether outliers have been removed from the
        data set or not.
        """

        try:
            rem = self._outliers_removed
        except AttributeError:
            rem = False

        return rem

    @outliers_removed.setter
    def outliers_removed(self, rem):
        try:
            self._outliers_removed = bool(rem)
        except Exception as e:
            raise TypeError("Value must be boolean: {}".format(e))

    def segment_list(self):
        """
        Using the heterodyned data timestamps, generate a list of segments
        (i.e., set of times between which there are uniformly spaced time
        stamps). This is returns as a list of 2-lists, with each item giving
        the start and end time (in GPS seconds) of that segment.
        """

        dt = self.dt.value
        times = self.times.value

        # initial segment
        segs = [[times[0], times[1]]]

        i = 2
        while i < len(self):
            if (times[i] - segs[-1][1]) == dt:
                segs[-1][1] = times[i]
            else:
                segs.append([times[i], times[i + 1]])
                i += 1
            i += 1

        return segs

    def plot(
        self,
        which="abs",
        figsize=(12, 4),
        ax=None,
        remove_outliers=False,
        thresh=3.5,
        zero_time=False,
        labelsize=None,
        fontsize=None,
        legendsize=None,
        fontname=None,
        labelname=None,
        **plotkwargs,
    ):
        """
        Plot the data time series.

        Parameters
        ----------
        which: str, 'abs'
            Say whehther to plot the absolute value of the data, ``'abs'``, the
            ``'real'`` component of the data, the ``'imag'`` component of
            the data, or ``'both'`` the real and imaginary components.
        figsize: tuple, (12, 4)
            A tuple with the size of the figure. Values set in `rcparams` will
            override this value.
        ax: Axes
            A :class:`matplotlib.axes.Axes` onto which to add the figure.
        remove_outliers: bool, False
            Set whether to remove outlier for the plot. If outliers removal
            was already specified when creating the
            :class:`~cwinpy.data.HeterodynedData` object they will
            automatically not be included in the plot.
        thresh: float, 3.5
            The threshold for outlier removal (see
            :meth:`~cwinpy.data.HeterodynedData.find_outliers`).
        zero_time: bool, False
            Start the time axis at zero.
        labelsize: int
            Set the fontsize for the axes tick labels.
        fontsize: int
            Set the fontsize for the axes labels.
        legendsize: int
            Set the fontsize for the legend (defaults to be the same as the
            value or `fontsize`).
        fontname: str
            Set the font name for the axes labels and legend.
        labelname: str
            Set the font name for the axes tick labels. If not set, this will
            default to the value given in `fontname`.
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

        if remove_outliers and self.outlier_mask is None:
            idx = self._not_outliers(thresh=thresh)
            times = self.times[idx] - (0 if not zero_time else self.times[0])
        else:
            idx = (
                np.arange(len(self))
                if self.outlier_mask is None
                else np.arange(len(self.outlier_mask))[self.outlier_mask]
            )
            times = self.times - (0 if not zero_time else self.times[0])

        # set some default plotting styles
        if "ls" not in plotkwargs:
            # set the line style to "None"
            plotkwargs["ls"] = "None"

        if "marker" not in plotkwargs:
            # set marker to a circle
            plotkwargs["marker"] = "o"

        if "xscale" not in plotkwargs and zero_time:
            # switch from auto-gps to linear scale if zeroing x-axis
            plotkwargs["xscale"] = "linear"
        elif "xscale" not in plotkwargs:
            plotkwargs["xscale"] = "auto-gps"

        # set the data to use
        if which.lower() in ["abs", "absolute"]:
            if "ylabel" not in plotkwargs:
                plotkwargs["ylabel"] = "$|B_k|$"
            if ax is None:
                plot = TimeSeries(self.take(idx).abs(), times=times).plot(
                    figsize=figsize, **plotkwargs
                )
            else:
                ax.set_ylabel(plotkwargs.pop("ylabel"))
                ax.set_xscale(plotkwargs.pop("xscale"))
                ax.plot(times, self.take(idx).abs(), **plotkwargs)
        elif which.lower() in ["real", "re"]:
            if "ylabel" not in plotkwargs:
                plotkwargs["ylabel"] = "$\\Re{(B_k)}$"
            if ax is None:
                plot = TimeSeries(self.take(idx).real, times=times).plot(
                    figsize=figsize, **plotkwargs
                )
            else:
                ax.set_ylabel(plotkwargs.pop("ylabel"))
                ax.set_xscale(plotkwargs.pop("xscale"))
                ax.plot(times, self.take(idx).real, **plotkwargs)
        elif which.lower() in ["im", "imag", "imaginary"]:
            if "ylabel" not in plotkwargs:
                plotkwargs["ylabel"] = "$\\Im{(B_k)}$"
            if ax is None:
                plot = TimeSeries(self.take(idx).imag, times=times).plot(
                    figsize=figsize, **plotkwargs
                )
            else:
                ax.set_ylabel(plotkwargs.pop("ylabel"))
                ax.set_xscale(plotkwargs.pop("xscale"))
                ax.plot(times, self.take(idx).imag, **plotkwargs)
        elif which.lower() == "both":
            if "ylabel" not in plotkwargs:
                plotkwargs["ylabel"] = "$B_k$"

            if ax is None:
                from gwpy.timeseries import TimeSeriesDict

                pldata = TimeSeriesDict()
                pldata["Real"] = TimeSeries(self.take(idx).real, times=times)
                pldata["Imag"] = TimeSeries(self.take(idx).imag, times=times)

                plot = pldata.plot(figsize=figsize, **plotkwargs)
                plot.gca().legend(loc="upper right", numpoints=1)
            else:
                ax.set_ylabel(plotkwargs.pop("ylabel"))
                ax.set_xscale(plotkwargs.pop("xscale"))
                ax.plot(times, self.take(idx).real, **plotkwargs)
                ax.plot(times, self.take(idx).imag, **plotkwargs)
                ax.legend(loc="upper right", numpoints=1)
        else:
            raise ValueError("'which' must be 'abs', 'real', 'imag' or 'both")

        if ax is None:
            return plot
        else:
            return ax

    def spectrogram(
        self,
        dt=86400,
        window=None,
        overlap=0.5,
        plot=True,
        ax=None,
        remove_outliers=False,
        thresh=3.5,
        fraction_labels=True,
        fraction_label_num=4,
        figsize=(12, 4),
        labelsize=None,
        fontsize=None,
        fontname=None,
        labelname=None,
        legendsize=None,
        **plotkwargs,
    ):
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
        figsize: tuple, (12, 4)
            A tuple containing the size (in inches) to set for the figure.
        labelsize: int
            Set the fontsize for the axes tick labels.
        fontsize: int
            Set the fontsize for the axes labels.
        legendsize: int
            Set the fontsize for the legend (defaults to be the same as the
            value or `fontsize`).
        fontname: str
            Set the font name for the axes labels and legend.
        labelname: str
            Set the font name for the axes tick labels. If not set, this will
            default to the value given in `fontname`.
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
        speckwargs["dt"] = dt
        speckwargs["plot"] = plot
        speckwargs["ax"] = ax
        speckwargs["window"] = window
        speckwargs["overlap"] = overlap
        speckwargs["remove_outliers"] = remove_outliers
        speckwargs["thresh"] = thresh
        speckwargs["fraction_labels"] = fraction_labels
        speckwargs["fraction_label_num"] = fraction_label_num

        return self._plot_power(
            "spectrogram",
            speckwargs,
            figsize=figsize,
            labelsize=labelsize,
            fontsize=fontsize,
            labelname=labelname,
            fontname=fontname,
            legendsize=legendsize,
            **plotkwargs,
        )

    def periodogram(
        self,
        plot=True,
        ax=None,
        remove_outliers=False,
        thresh=3.5,
        fraction_labels=True,
        fraction_label_num=4,
        figsize=(6, 5),
        labelsize=None,
        labelname=None,
        fontsize=None,
        fontname=None,
        legendsize=None,
        **plotkwargs,
    ):
        """
        Compute and plot a two-sided periodogram of the data using
        :func:`scipy.signal.periodogram`. Note that this uses zero-padded
        uniformly sampled data, rather than using the Lomb-Scargle method (such
        as :class:`astropy.stats.LombScargle`) that can deal with data with
        gaps, but doesn't work for complex data.

        See :meth:`~cwinpy.data.HeterodynedData.spectrogram` for input
        parameters, excluding `dt`, `window` and `overlap`. The default figure
        size is (6, 5).

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
        speckwargs["plot"] = plot
        speckwargs["ax"] = ax
        speckwargs["remove_outliers"] = remove_outliers
        speckwargs["thresh"] = thresh
        speckwargs["fraction_labels"] = fraction_labels
        speckwargs["fraction_label_num"] = fraction_label_num

        return self._plot_power(
            "periodogram",
            speckwargs,
            figsize=figsize,
            labelsize=labelsize,
            fontsize=fontsize,
            labelname=labelname,
            fontname=fontname,
            legendsize=legendsize,
            **plotkwargs,
        )

    def power_spectrum(
        self,
        plot=True,
        ax=None,
        remove_outliers=False,
        thresh=3.5,
        fraction_labels=True,
        fraction_label_num=4,
        average="median",
        dt=86400,
        figsize=(6, 5),
        labelsize=None,
        labelname=None,
        fontsize=None,
        fontname=None,
        legendsize=None,
        window=None,
        overlap=0.5,
        asd=False,
        **plotkwargs,
    ):
        """
        Compute and plot the power spectral density of the data. This computes
        the spectrogram, and averages the power over time.

        See :meth:`~cwinpy.data.HeterodynedData.spectrogram` for input
        parameters. The default figure size is (6, 5).

        Parameters
        ----------
        average: str, 'median'
            The method by which to "average" the spectrum in time. This can be
            'median' (the default), 'mean', 'harmonic_mean', 'max' (return the
            maximum) or 'min' (return the minimum).
        asd: bool
            If True, the amplitude spectral density will be returned rather
            than the power spectrum.
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
        speckwargs["plot"] = plot
        speckwargs["ax"] = ax
        speckwargs["remove_outliers"] = remove_outliers
        speckwargs["thresh"] = thresh
        speckwargs["fraction_labels"] = fraction_labels
        speckwargs["fraction_label_num"] = fraction_label_num
        speckwargs["dt"] = dt
        speckwargs["average"] = average
        speckwargs["window"] = window
        speckwargs["overlap"] = overlap
        speckwargs["asd"] = asd

        return self._plot_power(
            "power",
            speckwargs,
            figsize=figsize,
            labelsize=labelsize,
            fontsize=fontsize,
            labelname=labelname,
            fontname=fontname,
            legendsize=legendsize,
            **plotkwargs,
        )

    def _plot_power(
        self,
        ptype,
        speckwargs={},
        figsize=None,
        labelsize=None,
        labelname=None,
        fontsize=None,
        fontname=None,
        legendsize=None,
        **plotkwargs,
    ):
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
        figsize: tuple
            The size (in inches) of the created figure.
        plotkwargs:
            Additional plotting keyword arguments.
        """

        if not isinstance(ptype, str):
            raise TypeError("Power spectrum type must be a string")

        if ptype not in ["spectrogram", "periodogram", "power"]:
            raise ValueError("Type must be 'spectrogram', 'periodogram', or 'power'")

        # set plotting defaults
        if labelsize is None:
            labelsize = self.PLOTTING_DEFAULTS["labelsize"]
        if labelname is None:
            labelname = self.PLOTTING_DEFAULTS["labelname"]
        if fontsize is None:
            fontsize = self.PLOTTING_DEFAULTS["fontsize"]
        if fontname is None:
            fontname = self.PLOTTING_DEFAULTS["fontname"]
        if legendsize is None:
            legendsize = fontsize

        # get some options
        remove_outliers = speckwargs.get("remove_outliers", False)
        thresh = speckwargs.get("thresh", 3.5)
        plot = speckwargs.get("plot", True)
        ax = speckwargs.get("ax", None)

        # get the zero padded data
        padded = self._zero_pad(remove_outliers=remove_outliers, thresh=thresh)

        if self.outlier_mask is None and remove_outliers:
            idx = self._not_outliers(thresh=thresh)
            times = self.times[idx].value
            tottime = times[-1] - times[0]
        else:
            times = self.times.value
            tottime = self.tottime.value

        Fs = 1.0 / gcd_array(np.diff(times))  # sampling frequency of padded data
        Fn = 1.0 / np.diff(times).min()  # sampling frequency of non-padded data

        if ptype in ["spectrogram", "power"]:
            dt = speckwargs.get("dt", 86400)
            overlap = speckwargs.get("overlap", 0.5)
            window = speckwargs.get("window", None)

            if not isinstance(dt, (float, int)):
                raise ValueError("Time bin must be an integer or float")

            if dt < 1.0 / Fs or dt > (tottime + (1.0 / Fs)):
                raise ValueError("The time bin selected is invalid")

            # set the number of samples for each FFT block
            nfft = int(dt * Fs)

            if isinstance(overlap, float):
                if overlap >= 0.0 and overlap < 1.0:
                    noverlap = int(overlap * nfft)
                else:
                    raise ValueError("Overlap must be a float between 0 and 1")
            elif isinstance(overlap, int):
                if overlap >= 0 and overlap <= len(self) - 1:
                    noverlap = overlap
                else:
                    raise ValueError("Overlap is out of allowed range")
            else:
                raise TypeError("Overlap must be an integer or float")

            if window is None:
                try:
                    from scipy.signal import tukey
                except ImportError:
                    from scipy.signal.windows import tukey

                window = tukey(nfft, alpha=0.1)

            # generate spectrogram
            try:
                from matplotlib.mlab import specgram

                power, frequencies, stimes = specgram(
                    padded, Fs=Fs, window=window, NFFT=nfft, noverlap=noverlap
                )
            except Exception as e:
                raise RuntimeError("Problem creating spectrogram: {}".format(e))

            # get frequencies within +/-Fn / 2
            fidx = (frequencies >= -Fn / 2) & (frequencies <= Fn / 2)
            power = power[fidx, :]
            frequencies = frequencies[fidx]

            # rescale power due to zero padding spreading out power and different sample rate
            power *= (Fs / Fn) ** 2

            if ptype == "power":
                # average the spectrogram for a power spectrum
                average = speckwargs.get("average", "median").lower()

                if average not in [
                    "median",
                    "mean",
                    "harmonic_mean",
                    "hmean",
                    "max",
                    "min",
                ]:
                    raise ValueError(
                        "Average method must be 'median', 'mean', 'harmonic_mean', 'max' or 'min'."
                    )

                # ignore any power time bins that are zero
                nonzero = np.r_[[i for i in range(power.shape[1]) if power[0, i] != 0]]

                if average == "median":
                    power = np.median(power[:, nonzero], axis=-1)
                elif average == "mean":
                    power = np.mean(power[:, nonzero], axis=-1)
                elif average in ["harmonic_mean", "hmean"]:
                    power = hmean(power[:, nonzero], axis=-1)
                elif average == "max":
                    power = np.max(power[:, nonzero], axis=-1)
                else:
                    power = np.min(power[:, nonzero], axis=-1)
        else:
            # perform periodogram
            try:
                from scipy.signal import periodogram

                frequencies, power = periodogram(
                    padded, fs=Fs, return_onesided=False, detrend=lambda x: x
                )

                # sort results in frequency
                frequencies, power = np.array(sorted(zip(frequencies, power))).T
            except Exception as e:
                raise RuntimeError("Problem creating periodogram: {}".format(e))

        if ax is None and not plot:
            if ptype == "spectrogram":
                return frequencies, power, stimes
            else:
                if speckwargs.get("asd", False):
                    # convert PSD to ASD
                    power = np.sqrt(power)

                return frequencies, power

        # perform plotting
        try:
            from matplotlib import pyplot as plt
            from matplotlib.axes import Axes

            fraction_labels = speckwargs.get("fraction_labels", True)
            fraction_label_num = speckwargs.get("fraction_label_num", 4)

            # set whether to output frequency labels as fractions
            if fraction_labels:
                # set at quarters of the sample frequency
                if not isinstance(fraction_label_num, int):
                    raise TypeError("'fraction_label_num' must be an integer")

                if fraction_label_num < 1:
                    raise ValueError("'fraction_label_num' must be positive")

                df = Fn / fraction_label_num
                ticks = np.linspace(-Fn / 2, Fn / 2, int(Fn / df) + 1)
                labels = []
                for tick in ticks:
                    if tick == 0.0:
                        labels.append("$0$")
                    else:
                        # set the fraction label
                        sign = "-" if tick < 0.0 else ""
                        if np.abs(tick) == 1.0:
                            label = f"{sign}1"
                        else:
                            label = "${0}^{{{1}}}\u2044_{{{2}}}$".format(
                                sign, 1, round(1 / np.abs(tick))
                            )
                            labels.append(label)

            if ptype == "spectrogram":
                from matplotlib import colors

                # set plotting keyword arguments
                if "cmap" not in plotkwargs:
                    if self.detector is not None:
                        if self.detector in self.colmapdic:
                            plotkwargs["cmap"] = self.colmapdic[self.detector]

                # extents of the plot
                if "extent" not in plotkwargs:
                    plotkwargs["extent"] = [0, tottime, -Fn / 2, Fn / 2]

                if "aspect" not in plotkwargs:
                    plotkwargs["aspect"] = "auto"

                if "norm" not in plotkwargs:
                    plotkwargs["norm"] = colors.Normalize()

                if isinstance(ax, Axes):
                    fig = ax.get_figure()
                    thisax = ax
                else:
                    fig, thisax = plt.subplots(figsize=figsize)

                thisax.imshow(np.sqrt(np.flipud(power)), **plotkwargs)

                if self.detector is not None:
                    from matplotlib.offsetbox import AnchoredText

                    legend = AnchoredText(self.detector, loc=1)
                    thisax.add_artist(legend)

                thisax.set_xlabel(
                    "GPS - {}".format(int(times[0])),
                    fontname=fontname,
                    fontsize=fontsize,
                )
                thisax.set_ylabel(
                    "Frequency (Hz)", fontname=fontname, fontsize=fontsize
                )

                if fraction_labels:
                    thisax.set_yticks(ticks)
                    thisax.set_yticklabels(labels)

                # set axes to use scientific notation
                thisax.ticklabel_format(
                    axis="x", style="sci", scilimits=(0, 5), useMathText=True
                )
            else:
                # set plot color
                if self.detector is not None:
                    if "color" not in plotkwargs:
                        if self.detector in GW_OBSERVATORY_COLORS:
                            plotkwargs["color"] = GW_OBSERVATORY_COLORS[self.detector]

                    if "label" not in plotkwargs:
                        plotkwargs["label"] = self.detector

                if isinstance(ax, Axes):
                    fig = ax.get_figure()
                    thisax = ax
                else:
                    fig, thisax = plt.subplots(figsize=figsize)

                if speckwargs.get("asd", False):
                    # convert PSD to ASD
                    power = np.sqrt(power)

                thisax.plot(frequencies, power, **plotkwargs)

                if self.detector is not None:
                    from matplotlib.font_manager import FontProperties

                    legfont = FontProperties(family=fontname, size=legendsize)
                    thisax.legend(prop=legfont)

                if speckwargs.get("asd", False):
                    thisax.set_ylabel(
                        r"Amplitude / $\sqrt{\rm{Hz}}$",
                        fontname=fontname,
                        fontsize=fontsize,
                    )
                else:
                    thisax.set_ylabel(
                        "Power / Hz", fontname=fontname, fontsize=fontsize
                    )

                thisax.set_xlabel(
                    "Frequency (Hz)", fontname=fontname, fontsize=fontsize
                )
                thisax.set_xlim([-Fn / 2, Fn / 2])

                if fraction_labels:
                    thisax.set_xticks(ticks)
                    thisax.set_xticklabels(labels)

                # set axes to use scientific notation
                thisax.ticklabel_format(axis="y", style="sci", useMathText=True)

            # set tick font name
            for tick in thisax.get_xticklabels() + thisax.get_yticklabels():
                tick.set_fontname(labelname)

            # set the axes tick label size
            thisax.tick_params(which="both", labelsize=labelsize)

            # add a grid
            thisax.grid(True, linewidth=0.5, linestyle="--")
        except Exception as e:
            raise RuntimeError("Problem creating spectrogram: {}".format(e))

        fig.tight_layout()

        if ptype == "spectrogram":
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

        if self.outlier_mask is None and remove_outliers:
            idx = self._not_outliers(thresh=thresh)
            times = self.times.value[idx]
            data = self.data[idx]
        else:
            times = self.times.value
            data = self.data

        # check diff of times
        if len(times) < 2:
            raise ValueError("There must be at least two samples!")

        dts = np.diff(times).astype(
            np.float32
        )  # convert to float32 due to precision errors

        if np.all(dts == self.dt.value):
            # no zero padding required as data is evenly sampled
            return data

        # get the greatest common divisor of the deltaTs
        gcd = gcd_array(dts)

        # get the "new" padded time stamps
        tottime = times[-1] - times[0]
        newtimes = np.linspace(times[0], times[-1], 1 + int(tottime) // gcd)

        # get indices of original times im new times
        tidxs = np.where(np.in1d(newtimes, times))[0]

        # get zero array and add data
        padded = np.zeros(len(newtimes), dtype=complex)
        padded[tidxs] = data

        return padded

    @property
    def include_ssb(self):
        """
        A boolean stating whether the heterodyne included Solar System
        barycentring.
        """

        try:
            return self._include_ssb
        except AttributeError:
            return False

    @include_ssb.setter
    def include_ssb(self, incl):
        self._include_ssb = bool(incl)

    @property
    def include_bsb(self):
        """
        A boolean stating whether the heterodyne included Binary System
        barycentring.
        """

        try:
            return self._include_bsb
        except AttributeError:
            return False

    @include_bsb.setter
    def include_bsb(self, incl):
        self._include_bsb = bool(incl)

    @property
    def include_glitch(self):
        """
        A boolean stating whether the heterodyne included corrections for any
        glitch phase evolution.
        """

        try:
            return self._include_glitch
        except AttributeError:
            return False

    @include_glitch.setter
    def include_glitch(self, incl):
        self._include_glitch = bool(incl)

    @property
    def include_fitwaves(self):
        """
        A boolean stating whether the heterodyne included corrections for any
        red noise FITWAVES parameters.
        """

        try:
            return self._include_fitwaves
        except AttributeError:
            return False

    @include_fitwaves.setter
    def include_fitwaves(self, incl):
        self._include_fitwaves = bool(incl)

    def as_timeseries(self):
        """
        Return the data as a :class:`gwpy.timeseries.TimeSeries`.
        """

        return TimeSeries(self.data, times=self.times, channel=self.channel)

    @property
    def filter_history(self):
        """
        An array with the "history" of any filters used during the
        heterodyning.
        """

        try:
            return self._filter_history
        except AttributeError:
            return None

    @filter_history.setter
    def filter_history(self, history):
        self._filter_history = np.array(history)

    def heterodyne(self, phase, stride=1, singlesided=False, datasegments=None):
        """
        Heterodyne the data (see :meth:`gwpy.timeseries.TimeSeries.heterodyne`)
        for details). Unlike :meth:`gwpy.timeseries.TimeSeries.heterodyne` this
        will heterodyne unevenly sampled data, although contiguous segments of
        data will be truncated if they do not contain an integer number of
        strides. Additional parameters are given below:

        Parameters
        ----------
        datasegments: list
            A list of pairs of times within which to include data. Data outside
            these times will be removed.
        """

        try:
            phaselen = len(phase)
        except Exception as exc:
            raise TypeError("Phase is not array_like: {}".format(exc))
        if phaselen != len(self):
            raise ValueError(
                "Phase array must be the same length as the HeterodynedData"
            )

        dt = self.dt.value
        samplerate = 1.0 / dt
        stridesamp = int(stride * samplerate)

        # find contiguous stretches of data
        if not np.allclose(
            np.diff(self.times.value), self.dt.value * np.ones(len(self) - 1)
        ):
            breaks = np.argwhere(np.diff(self.times.value) != self.dt.value)[
                :, 0
            ].tolist()
            segments = SegmentList()
            breaks = [-1] + breaks + [len(self) - 1]
            for i in range(len(breaks) - 1):
                segments.append(
                    (
                        self.times.value[breaks[i] + 1] - dt / 2,
                        self.times.value[breaks[i + 1]] + dt / 2,
                    )
                )
        else:
            segments = SegmentList(
                [(self.times.value[0] - dt / 2, self.times.value[-1] + dt / 2)]
            )

        if datasegments is not None:
            # get times within segments and data time span
            segments = (segments & SegmentList(datasegments)) & SegmentList(
                [(self.times.value[0] - dt / 2, self.times.value[-1] + dt / 2)]
            )

        # check that some data is within the segments
        if len(segments) == 0:
            return None

        # heterodyne the data
        hetdata = np.exp(-1j * np.asarray(phase)) * self.data

        times = []
        # downsample the data
        counter = 0
        for seg in segments:
            segsize = seg[1] - seg[0]
            nsteps = int(segsize // stride)  # number of steps within segment

            idx = np.argwhere(
                (self.times.value >= seg[0]) & (self.times.value < seg[1])
            ).flatten()

            if len(idx) == 0:
                continue

            for step in range(nsteps):
                istart = int(stridesamp * step)
                iend = istart + stridesamp
                stepidx = idx[istart:iend]
                hetdata[counter] = (
                    2 * hetdata[stepidx].mean()
                    if singlesided
                    else hetdata[stepidx].mean()
                )
                times.append(self.times.value[stepidx].mean())
                counter += 1

        # resize
        hetdata.resize((counter,), refcheck=False)

        # create new object (will not contain, e.g., par file, injection info,
        # etc)
        out = type(self)(hetdata, times=times, window=0, bbminlength=len(self))
        out.bbminlength = None
        out.__array_finalize__(self)

        return out

    @property
    def heterodyne_arguments(self):
        """
        The dictionary of arguments passed to
        :class:`cwinpy.heterodyne.Heterodyne` if it was used to create the
        current :class:`~cwinpy.data.HeterodynedData` object. If the
        :class:`~cwinpy.data.HeterodynedData` object was created via a merge of
        many individual :class:`~cwinpy.data.HeterodynedData` objects then this
        will be a list with a dictionary from each of the merged objects.

        .. note::

           This attribute will only be saved if writing the data to a HDF5
           file, but not if writing to an ascii text file.
        """

        if hasattr(self, "_heterodyne_arguments"):
            return self._heterodyne_arguments
        else:
            return None

    @heterodyne_arguments.setter
    def heterodyne_arguments(self, args):
        if not isinstance(args, (dict, list)):
            raise TypeError("heterodyne arguments must be a dictionary or list")

        if self.heterodyne_arguments is None:
            self._heterodyne_arguments = args
        elif isinstance(self.heterodyne_arguments, dict):
            # convert into list and append
            self._heterodyne_arguments = [self._heterodyne_arguments]
            self._heterodyne_arguments.append(args)
        elif isinstance(self.heterodyne_arguments, list):
            self._heterodyne_arguments.append(args)

    @property
    def cwinpy_heterodyne_pipeline_config(self):
        """
        If the :class:`~cwinpy.data.HeterodynedData` object was created through
        :class:`~cwinpy.heterodyne.Heterodyne` being called within a HTCondor
        DAG, which itself was set up using the ``cwinpy_heterodyne_pipeline``
        script, then this attribute can contain the contents of the
        configuration file that created the DAG.
        """

        if hasattr(self, "_cwinpy_heterodyne_pipeline_config"):
            return self._cwinpy_heterodyne_pipeline_config
        else:
            return None

    @cwinpy_heterodyne_pipeline_config.setter
    def cwinpy_heterodyne_pipeline_config(self, config):
        # check config can be converted into a valid ConfigParser object
        try:
            import configparser

            cf = configparser.ConfigParser()
            cf.read_string(config)
        except (TypeError, configparser.MissingSectionHeaderError):
            raise TypeError("Configuration file is not valid")

        self._cwinpy_heterodyne_pipeline_config = config

    def generate_roq(self, priors, **kwargs):
        """
        Generate reduced order quadrature interpolators for likelihood
        calculations for each data chunk. See
        :class:`~cwinpy.pe.roq.GenerateROQ` for additional keyword arguments.
        The ``model`` argument is not required as that will always be
        set to ``"HeterodynedCWSimulator"``, while the ``par`` and ``det``
        arguments will default to be those in the
        :class:`~cwinpy.data.HeterodynedData` object.

        Parameters
        ----------
        priors: PriorDict
            A :class:`~bilby.core.prior.PriorDict` containing the parameters
            and prior distributions, which will be used to generate the model
            reduced basis set.
        """

        from .pe.roq import GenerateROQ

        roq = []

        roqkwargs = kwargs.copy()
        roqkwargs.setdefault("model", "HeterodynedCWSimulator")
        roqkwargs.setdefault("par", self.par)
        roqkwargs.setdefault("det", self.detector)

        count = 1
        for cpidx, cplen in zip(
            self.change_point_indices,
            self.chunk_lengths,
        ):
            if kwargs.get("verbose", False):
                print(
                    f"Generating Reduced Basis for chunk {count} "
                    f"[PSR: {get_psr_name(self.par)}, detector: {self.detector}]"
                )
                count += 1

            if roqkwargs.get("likelihood", "studentst") == "gaussian":
                roqkwargs["sigma"] = self.stds[cpidx : cpidx + cplen]

            chunkroq = GenerateROQ(
                self.data[cpidx : cpidx + cplen],
                self.times[cpidx : cpidx + cplen],
                priors,
                **roqkwargs,
            )

            roq.append(chunkroq)

            if kwargs.get("verbose", False):
                # print out ROQ basis sizes
                print(f"No. real basis nodes: {len(chunkroq._x_node_indices_real)}")
                print(
                    f"No. imaginary basis nodes: {len(chunkroq._x_node_indices_imag)}"
                )
                print(
                    f"No. squared model basis nodes: {len(chunkroq._x2_node_indices)}"
                )

        return roq

    def __len__(self):
        return len(self.data)


class PSDwrapper(object):
    """
    Wrapper for LALSimulation PSD functions that require a frequency series.

    Parameters
    ----------
    psdfunc: callable
        The function that generates the PSD.
    f0: float
        The frequency at which to extract the PSD.

    Returns
    -------
    psd: float
        The value to the PSD at ``f0``.
    """

    def __init__(self, psdfunc, f0=None):
        self.psdfunc = psdfunc
        self.f0 = f0

    def psd(self, f0=None):
        fval = f0 if f0 is not None else self.f0
        if fval is None:
            raise ValueError("No frequency has been supplied!")

        fs = lal.CreateREAL8FrequencySeries("", None, fval, 1.0, None, 2)
        try:
            _ = self.psdfunc(fs, fval)
        except Exception as e:
            raise RuntimeError("PSD function failed: {}".format(e))
        return fs.data.data[0]

    __call__ = psd
