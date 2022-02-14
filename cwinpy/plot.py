import logging

import numpy as np
from bilby.core.grid import Grid
from bilby.core.result import Result, read_in_result
from cwinpy.utils import lalinference_to_bilby_result
from gwpy.plot.colors import GW_OBSERVATORY_COLORS
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from pesummary.conf import colorcycle

from .parfile import PulsarParameters

#: dictionary of common parameters and equivalent LaTeX format strings
LATEX_LABELS = {
    "h0": r"$h_0$",
    "c21": r"$C_{21}$",
    "c22": r"$C_{22}$",
    "cosiota": r"$\cos{\iota}$",
    "siniota": r"$\sin{\iota}$",
    "iota": r"$\iota$ (rad)",
    "psi": r"$\psi$ (rad)",
    "phi0": r"$\phi_0$ (rad)",
    "phi21": r"$\Phi_{21}$ (rad)",
    "phi22": r"$\Phi_{22}$ (rad)",
    "q22": r"$Q_{22}$ ($\text{ kg}\,\text{m}^2$)",
}

#: dictionary of default parameter bounds
DEFAULT_BOUNDS = {
    "h0": {"low": 0.0},
    "c21": {"low": 0.0},
    "c22": {"low": 0.0},
    "cosiota": {"low": -1.0, "high": 1.0},
    "siniota": {"low": 0.0, "high": 1.0},
    "psi": {"low": 0.0, "high": np.pi / 2},
    "iota": {"low": 0.0, "high": np.pi, "method": "Transform"},
    "phi0": {"low": 0.0, "high": np.pi},
    "phi21": {"low": 0.0, "high": 2 * np.pi},
    "phi22": {"low": 0.0, "high": np.pi},
    "q22": {"low": 0.0},
}

#: dictionary mapping colors to color map names.
COLOR_MAP = {
    "b": "Blues",
    "r": "Reds",
    "g": "Greens",
    "k": "Greys",
    "m": "Purples",
    "c": "BuGn",
    "#222222": "Greys",
    "#ffb200": "YlOrBr",
    "#ee0000": "Reds",
    "#b0dd8b": "YlGn",
    "#4ba6ff": "Blues",
    "#9b59b6": "PuRd",
}

#: list of allowed 1d plot types
ONED_PLOT_TYPES = ["hist", "kde"]

#: list of allowed 2d plot types
TWOD_PLOT_TYPES = ["contour", "triangle", "reverse_triangle", "corner"]

#: list of allowed nd plot types
MULTID_PLOT_TYPES = ["corner"]


class Plot:
    def __init__(
        self,
        results,
        parameters=None,
        plottype="corner",
        latex_labels=None,
        kde=False,
        pulsar=None,
        untrig=None,
    ):
        """
        A class to plot individual or joint posterior distributions using a
        variety of plotting functions.

        .. note::

           In the case of a "corner" plot, the :class:`~bilby.core.grid.Grid`
           results can only be plotted together with samples from a
           :class:`~bilby.core.result.Result` object. If plotting
           :class:`~bilby.core.grid.Grid` results on their own, only single or
           pairs of parameters can be specified.

        Parameters
        ----------
        results: str, dict
            Pass the results to be plotted. This can either be in the form of
            a string giving the path to the results file(s), or directly
            passing bilby :class:`~bilby.core.result.Result` or
            :class:`~bilby.core.grid.Grid` objects. If passing a dictionary,
            the values can be file paths, :class:`~bilby.core.result.Result` or
            :class:`~bilby.core.grid.Grid` objects, while the keys may be, for
            example, detector names if wanting to overplot parameters estimated
            for different detectors. The keys will be used a legend labels for
            the plots.
        parameters: list, str
            A list of the parameters that you want to plot. If requesting a
            single parameter this can be a string with the parameter name. If
            this value is ``None`` (the default) then all parameters will be
            plotted.
        plottype: str
            The type of plot to produce. For 1d plots, this can be: "hist" -
            produce a histogram of the posterior or "kde" - produce a KDE plot of
            the posterior. For 2d plots, this can be: "contour", "triangle",
            "reverse_triangle", or "corner". For higher dimensional plots
            only "corner" can be used.
        latex_labels: dict
            A dictionary of LaTeX labels to be used for axes for the given
            parameters.
        kde: bool
            If plotting a histogram using ``"hist"``, set this to True to also
            plot the KDE. Use the ``"kde"`` `plottype` to only plot the KDE.
        pulsar: str, PulsarParameters
            A Tempo(2)-style pulsar parameter file containing the source
            parameters. If the requested `parameters` are in the parameter
            file, e.g., for a simulated signal, then if supplied these will be
            plotted with the posteriors.
        untrig: str, list
            A string, or list, of parameters that are defined as the
            trigonometric function of another parameters. If given in the list
            then those parameters will be inverted, e.g., if ``"cosiota"`` is
            present it will be changed to be "iota". This only works for result
            samples and not grid values. Default is None.
        """

        self.untrig = untrig
        self.results = results
        self.parameters = parameters
        self.plottype = plottype
        self.latex_labels = latex_labels
        self.kde = kde
        self.pulsar = pulsar

    @property
    def results(self):
        """
        A dictionary of results objects, where these can be either a
        :class:`~bilby.core.result.Result` or :class:`~bilby.core.grid.Grid`
        object. If a single result is present it will be stored in a dictionary
        with the default key ``"result"``.
        """

        return self._results

    @results.setter
    def results(self, results):
        self._results = {}

        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, (Grid, Result, str)):
                    self._results[key] = Plot._parse_result(value)
                else:
                    raise TypeError(f"result in '{key}' is not the correct type")
        elif isinstance(results, (Grid, Result, str)):
            self._results["result"] = Plot._parse_result(results)
        else:
            raise TypeError("results is not the correct type")

        # invert trigonometric parameters if given
        if isinstance(self.untrig, (str, list)):
            untrig = (
                [self.untrig] if isinstance(self.untrig, str) else list(self.untrig)
            )
            for key in self._results:
                if isinstance(self._results[key], Result):
                    for p in untrig:
                        if p in self._results[key].posterior.columns.values:
                            # add in new column
                            if p[0:3] == "cos":  # trig function
                                tf = np.arccos
                            elif p[0:3] == "sin":
                                tf = np.arcsin

                            self._results[key].posterior[p[3:]] = tf(
                                self._results[key].posterior[p]
                            )

                            # remove old column
                            self._results[key].posterior.drop(columns=p, inplace=True)
                elif isinstance(self._results[key], Grid):
                    raise TypeError("Cannot invert trigonometric parameter in Grid")

        # store the available parameters for each result object
        self._results_parameters = {}
        for key, value in self._results.items():
            if isinstance(self._results[key], Grid):
                self._results_parameters[key] = sorted(self._results[key].priors.keys())
            else:
                self._results_parameters[key] = sorted(
                    self._results[key].posterior.columns.values
                )

    @staticmethod
    def _parse_result(result):
        # try reading in a results file by iterating over it being a bibly
        # Result object, a bilby Grid object, or a LALInference produced file.
        if isinstance(result, (Grid, Result)):
            # no parsing required
            return result

        success = False

        for iofunc in [read_in_result, Grid.read, lalinference_to_bilby_result]:
            try:
                res = iofunc(result)
                success = True
                break
            except Exception:
                pass

        if not success:
            raise IOError(f"result file '{result}' is not a recognised format")

        return res

    @property
    def parameters(self):
        """
        The list of parameters to be plotted.
        """

        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        # if None the plot all parameters in the results files
        if parameters is None:
            # check for consistent parameters amoung results
            checkparams = list(self._results_parameters.values())[0]
            for params in self._results_parameters.values():
                if params != checkparams:
                    raise ValueError(
                        "results have inconsistent parameters, so all parameters cannot be plotted together"
                    )
        else:
            if isinstance(parameters, (str, list)):
                checkparams = (
                    [parameters] if isinstance(parameters, str) else list(parameters)
                )
            else:
                raise TypeError("parameters must be a string or list")

            # make sure values are all lower case
            checkparams = [param.lower() for param in checkparams]

            # check that requested parameters are available for all results
            for params in self._results_parameters.values():
                intersection = set(params) & set(checkparams)
                if intersection != set(checkparams):
                    badparams = list(intersection ^ set(checkparams))
                    raise ValueError(
                        f"Parameters '{badparams}' are not available in the results"
                    )

        self._parameters = checkparams
        self._num_parameters = len(checkparams)

    @property
    def plottype(self):
        """
        The plotting function type being used.
        """

        return self._plottype

    @plottype.setter
    def plottype(self, plottype):
        self._plottype = plottype
        if self._num_parameters == 1:
            if plottype not in ONED_PLOT_TYPES:
                raise TypeError(
                    f"Plot type '{plottype}' is not allowed for one parameter"
                )
        elif self._num_parameters == 2:
            if plottype not in TWOD_PLOT_TYPES:
                raise TypeError(
                    f"Plot type '{plottype}' is not allowed for two parameters"
                )
        else:
            if plottype not in MULTID_PLOT_TYPES:
                raise TypeError(
                    f"Plot type '{plottype}' is not allowed for multiple parameters"
                )

    @property
    def latex_labels(self):
        """
        Dictionary of LaTeX labels for each parameter.
        """

        return self._latex_labels

    @latex_labels.setter
    def latex_labels(self, labels):  # pragma: no cover
        self._latex_labels = {}
        for param in self.parameters:
            try:
                # check if label is supplied
                label = labels[param]
            except (TypeError, KeyError):
                if param in LATEX_LABELS:
                    # use existing defined label
                    label = LATEX_LABELS[param]
                else:
                    # create label from parameter name
                    label = param.replace("_", " ")  # remove _ characters

            self._latex_labels[param] = label

    @property
    def pulsar(self):
        """
        The :class:`~cwinpy.parfile.PulsarParameters` object containing the
        source parameters.
        """

        return self._pulsar

    @pulsar.setter
    def pulsar(self, pulsar):
        if pulsar is None:
            self._pulsar = None
            self._injection_parameters = None
        else:
            try:
                self._pulsar = PulsarParameters(pulsar)
            except ValueError:
                raise IOError(f"Could not parse pulsar parameter file '{pulsar}'")

            untrig = None
            if isinstance(self.untrig, (str, list)):
                untrig = (
                    [self.untrig] if isinstance(self.untrig, str) else list(self.untrig)
                )

            self._injection_parameters = {}
            for param in self.parameters:
                # check if inverted trigonometric function has been performed
                # and if so invert parameter value if required
                if untrig is not None:
                    for p in untrig:
                        if param == p[3:] and self._pulsar[param] is None:
                            if p[0:3] == "cos":  # trig function
                                self._pulsar[param] = np.arccos(self._pulsar[p])
                            elif p[0:3] == "sin":
                                self._pulsar[param] = np.arcsin(self._pulsar[p])
                            break

                self._injection_parameters[param] = self._pulsar[param]

    @property
    def injection_parameters(self):
        """
        A dictionary of simulated/true source parameters.
        """

        return self._injection_parameters

    @property
    def fig(self):
        """
        The :class:`matplotlib.figure.Figure` object created for the plot.
        """

        if hasattr(self, "_fig"):
            return self._fig
        else:
            return None

    @fig.setter
    def fig(self, fig):
        self._fig = fig[0] if isinstance(fig, tuple) else fig

    def savefig(self, fname, **kwargs):
        """
        An alias to run :meth:`matplotlib.figure.Figure.savefig` on the produced figure.
        """

        self.fig.savefig(fname, **kwargs)

    # alias savefig to save
    save = savefig

    def plot(self, **kwargs):
        """
        Create the plot of the data. Different keyword arguments can be passed
        for the different plot options as described below. By default, a
        particular colour scheme will be used, which if passed results for
        different detectors will use the
        `GWPy colour scheme <https://gwpy.github.io/docs/latest/plotter/colors.html>`_
        to differentiate them.

        For 1D plots using the "hist" option, keyword arguments to the
        :func:`matplotlib.pyplot.hist` function can be passed using a
        ``hist_kwargs`` dictionary keyword argument (by default the ``density``
        options, to plot the probability density, rather than bin counts will
        be ``True``). For 1D plots using the "kde" option, arguments for the
        KDE can be passed using a ``kde_kwargs`` dictionary keyword argument.

        When using the "contour" option the keyword arguments for
        :func:`corner.hist2d` can be used (note that here the
        ``plot_datapoints`` option will default to ``False``).

        For "corner" plots the keyword options for :func:`corner.corner` can be
        used.

        For "corner" plots, lines showing credible probability intervals can be
        set using the ``quantiles`` keyword. For other plots, lines showing the
        90% credible bounds (between the 5% and 95% percentiles) will be
        included if the ``plot_percentile`` keyword is set to ``True``.

        Parameters
        ----------
        colors: dict
            A dictionary of colour codes keyed to the keys of the ``results``.
            By default the GWpy colour scheme will be used for results keyed by
            GW detector prefixes. If keys are not known detector prefixes then
            the PESummary default color cycle will be used.
        grid2d: bool
            If plotting a `corner` plot, and overplotting Grid-based
            posteriors, set this to True to show the 2D Grid-based posterior
            densities rather than just the 1D posteriors. Default is False.

        Returns
        -------
        fig: Figure
            The :class:`matplotlib.figure.Figure` object.
        """

        # get colors
        colors = kwargs.get("colors", GW_OBSERVATORY_COLORS)

        # get Result samples
        self._samples = {
            label: value.posterior
            for label, value in self.results.items()
            if isinstance(value, Result)
        }

        # get Grid posteriors
        self._grids = {
            label: [value, value.ln_evidence]  # store grid and log evidence
            for label, value in self.results.items()
            if isinstance(value, Grid)
        }

        colordicts = []
        for j, res in enumerate([self._samples, self._grids]):
            colordicts.append({})
            for i, key in enumerate(res):
                if key in colors:
                    colordicts[-1][key] = colors[key]
                elif key.lower() == "joint":
                    # if using "Joint" as the multi-detector analysis key, set the color to black
                    colordicts[-1][key] = "k"
                else:
                    # use PESummary color cycle
                    colordicts[-1][key] = list(colorcycle)[
                        (j * 2 + i) % len(colorcycle)
                    ]

        # store original keywords arguments
        origkwargs = kwargs.copy()

        # plot samples
        fig = None
        if len(self._samples) > 0:
            kwargs["colors"] = list(colordicts[0].values())
            if self._num_parameters == 1:
                fig = self._1d_plot_samples(**kwargs)
            elif self._num_parameters == 2 and self.plottype != "corner":
                fig = self._2d_plot_samples(**kwargs)
            else:
                fig = self._nd_plot_samples(**kwargs)

        # restore keywords
        kwargs = origkwargs

        if len(self._grids) > 0:
            kwargs["colors"] = list(colordicts[1].values())
            if fig is not None and "fig" not in kwargs:
                kwargs["fig"] = fig
            if self._num_parameters == 1:
                fig = self._1d_plot_grid(**kwargs)
            elif self._num_parameters == 2 and self.plottype != "corner":
                fig = self._2d_plot_grid(**kwargs)
            else:
                fig = self._nd_plot_grid(**kwargs)

        # add further figure information
        if self._num_parameters == 1:
            ax = fig.gca()

            # set figure bounds if outside defaults
            if self.parameters[0] in DEFAULT_BOUNDS:
                _set_axes_limits(ax, self.parameters[0], axis="x")

            # add injection values
            if self.injection_parameters is not None:
                if self.injection_parameters[self.parameters[0]] is not None:
                    ax.axvline(
                        self.injection_parameters[self.parameters[0]],
                        color=kwargs.get("injection_color", "k"),
                        linewidth=1,
                    )
        elif self._num_parameters == 2:
            if "triangle" in self.plottype:
                a1, a2, a3 = fig[1:]
                order = ["x", "y"] if self.plottype == "triangle" else ["y", "x"]
                params = (
                    self.parameters[:2]
                    if self.plottype == "triangle"
                    else self.parameters[1::-1]
                )

                # set figure bounds if outside defaults
                for param, axes, axis in zip(params, [[a1, a2], [a2, a3]], order):
                    for ax in axes:
                        _set_axes_limits(ax, param, axis=axis)

        self.fig = fig
        return self.fig

    def _1d_plot_samples(self, **kwargs):
        """
        Create 1D plots from posterior samples.
        """

        from pesummary.core.plots.plot import _1d_comparison_histogram_plot

        param = self.parameters[0]

        if self.plottype == "kde":
            # set for plotting a KDE
            kwargs["kde"] = True
            kwargs["hist"] = False
        elif self.plottype == "hist":
            # set whether or not to also plot the KDE
            kwargs["kde"] = self.kde

        if "plot_percentile" not in kwargs:
            kwargs["plot_percentile"] = False

        kde_kwargs = kwargs.pop("kde_kwargs", {})
        if kwargs["kde"]:
            from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde

            # required to allow KDE generation when sample values are small (see
            # https://git.ligo.org/lscsoft/pesummary/-/merge_requests/604)
            kde_kwargs.update({"variance_atol": 0.0})

            # add bounds
            if param in DEFAULT_BOUNDS:
                kde_kwargs["xlow"] = (
                    DEFAULT_BOUNDS[param]["low"]
                    if "low" in DEFAULT_BOUNDS[param]
                    else None
                )
                kde_kwargs["xhigh"] = (
                    DEFAULT_BOUNDS[param]["high"]
                    if "high" in DEFAULT_BOUNDS[param]
                    else None
                )
                kde_kwargs["method"] = (
                    DEFAULT_BOUNDS[param]["method"]
                    if "method" in DEFAULT_BOUNDS[param]
                    else "Reflection"
                )
                kde_kwargs["kde_kernel"] = bounded_1d_kde

        singlesamps = [self._samples[key][param].values for key in self._samples]

        origkwargs = kwargs.copy()
        colors = kwargs.pop("colors")

        with DisableLogger():
            fig = _1d_comparison_histogram_plot(
                self.parameters[0],
                singlesamps,
                colors,
                self.latex_labels[param],
                list(self._samples.keys()),
                kde_kwargs=kde_kwargs,
                **kwargs,
            )

        # remove legend if this is the only result to plot
        if len(self.results) == 1:
            fig.axes[0].get_legend().remove()

        kwargs = origkwargs

        return fig

    def _1d_plot_grid(self, **kwargs):
        """
        Create 1D plots from posterior grids.
        """

        from pesummary.core.plots.plot import _1d_analytic_plot

        origkwargs = kwargs.copy()

        if "fig" not in kwargs:
            from pesummary.core.plots.figure import figure

            fig, ax = figure(gca=True)
            kwargs["fig"] = fig
            existingfig = False
        else:
            fig = kwargs["fig"]
            ax = fig.gca()
            existingfig = True

            # get existing legend info if using a KDE plot
            if self.plottype == "kde":
                texts = [
                    text.get_text() for text in fig.axes[0].get_legend().get_texts()
                ]
                leglines = fig.axes[0].get_legend().get_lines()

        colors = kwargs.pop("colors")

        if "plot_percentile" not in kwargs:
            # don't plot percentiles by default
            kwargs["plot_percentile"] = False

        for i, (label, grid) in enumerate(self._grids.items()):
            x = grid[0].sample_points[self.parameters[0]]

            pdf = np.exp(
                grid[0].marginalize_ln_posterior(not_parameters=self.parameters[0])
                - grid[1]
            )
            kwargs["label"] = label
            kwargs["color"] = colors[i]
            kwargs["title"] = False

            fig = _1d_analytic_plot(
                self.parameters[0],
                x,
                pdf,
                self.latex_labels[self.parameters[0]],
                ax=ax,
                **kwargs,
            )

        # update the legend
        if existingfig or len(self._grids) > 1:
            from matplotlib.lines import Line2D

            curhandles, labels = ax.get_legend_handles_labels()
            handles = []

            if self.plottype == "kde":
                # add in values from KDE plot
                curhandles = leglines + curhandles
                labels = texts + labels

            for handle, label in zip(curhandles, labels):
                # switch any Patches labels to Line2D objects
                legcolor = (
                    handle.get_color()
                    if isinstance(handle, Line2D)
                    else handle.get_edgecolor()
                )
                handles.append(Line2D([], [], color=legcolor, label=label))

            ax.legend(handles=handles)

        kwargs = origkwargs

        return fig

    def _2d_plot_samples(self, **kwargs):
        """
        Create 2D plots from posterior samples.
        """

        from pesummary.core.plots.bounded_2d_kde import Bounded_2d_kde

        # get bounds
        lows = []
        highs = []
        methods = []
        for param in self.parameters[0:2]:
            if param in DEFAULT_BOUNDS:
                lows.append(
                    DEFAULT_BOUNDS[param]["low"]
                    if "low" in DEFAULT_BOUNDS[param]
                    else None
                )
                highs.append(
                    DEFAULT_BOUNDS[param]["high"]
                    if "high" in DEFAULT_BOUNDS[param]
                    else None
                )
                methods.append(
                    DEFAULT_BOUNDS[param]["method"]
                    if "method" in DEFAULT_BOUNDS[param]
                    else "Reflection"
                )

        if self.plottype == "triangle":
            from pesummary.core.plots.publication import triangle_plot as plotfunc
        elif self.plottype == "reverse_triangle":
            from pesummary.core.plots.publication import (
                reverse_triangle_plot as plotfunc,
            )
        else:
            # contour plot
            from pesummary.core.plots.publication import (
                comparison_twod_contour_plot as plotfunc,
            )

            # set KDE information
            kwargs.update(
                {
                    "kde": Bounded_2d_kde,
                    "kde_kwargs": {
                        "xlow": lows[0],
                        "xhigh": highs[0],
                        "ylow": lows[1],
                        "yhigh": highs[1],
                    },
                }
            )

            # default to not showing data points
            if "plot_datapoints" not in kwargs:
                kwargs["plot_datapoints"] = False

        if "triangle" in self.plottype:
            from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde

            # set KDE informaiton
            kwargs.update(
                {
                    "kde_2d": Bounded_2d_kde,
                    "kde_2d_kwargs": {
                        "xlow": lows[0],
                        "xhigh": highs[0],
                        "ylow": lows[1],
                        "yhigh": highs[1],
                    },
                    "kde": bounded_1d_kde,
                }
            )

            kwargs["kde_kwargs"] = {
                "x_axis": {"xlow": lows[0], "xhigh": highs[0], "method": methods[0]},
                "y_axis": {"xlow": lows[1], "xhigh": highs[1], "method": methods[1]},
            }

        args = [
            [samps[self.parameters[0]].values for samps in self._samples.values()],
            [samps[self.parameters[1]].values for samps in self._samples.values()],
        ]

        if "xlabel" not in kwargs:
            kwargs["xlabel"] = self.latex_labels[self.parameters[0]]
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = self.latex_labels[self.parameters[1]]

        if "labels" not in kwargs and len(self.results) > 1:
            kwargs["labels"] = list(self._samples.keys())

        # set injection parameter values
        if self.injection_parameters is not None:
            if (
                self.injection_parameters[self.parameters[0]] is not None
                and self.injection_parameters[self.parameters[1]] is not None
            ):
                kwargname = "truths" if self.plottype == "corner" else "truth"
                kwargs[kwargname] = [
                    self.injection_parameters[self.parameters[0]],
                    self.injection_parameters[self.parameters[1]],
                ]

        # create plot
        with DisableLogger():
            fig = plotfunc(*args, **kwargs)

        return fig

    def _2d_plot_grid(self, **kwargs):
        """
        Create 2D plot from posterior grids.
        """

        origkwargs = kwargs.copy()
        colors = kwargs.pop("colors")

        # add no grid by default
        if "grid" not in kwargs:
            kwargs["grid"] = False

        if "shading" not in kwargs:
            kwargs["shading"] = "gouraud"

        if "levels" not in kwargs:
            kwargs["levels"] = None

        existingfig = False
        if self.plottype == "contour":
            from pesummary.core.plots.publication import (
                analytic_twod_contour_plot as plotfunc,
            )

            if "fig" in kwargs:
                ax = kwargs["fig"].gca()
                kwargs["ax"] = ax
                existingfig = True
        elif "triangle" in self.plottype:
            if self.plottype == "triangle":
                from pesummary.core.plots.publication import (
                    analytic_triangle_plot as plotfunc,
                )
            else:
                from pesummary.core.plots.publication import (
                    analytic_reverse_triangle_plot as plotfunc,
                )

            # get existing figure
            kwargs["existing_figure"] = kwargs.pop("fig", None)
            if kwargs["existing_figure"] is not None:
                existingfig = True

        for i, (label, grid) in enumerate(self._grids.items()):
            kwargs["level_kwargs"] = {
                "colors": [colors[i]]
            }  # set colour via level_kwargs
            kwargs["label"] = label

            if colors[i] in COLOR_MAP and "cmap" not in kwargs:
                kwargs["cmap"] = COLOR_MAP[colors[i]]

            if "zorder" not in kwargs:
                kwargs["zorder"] = -10  # try and plot 2D plot first

            if self.plottype == "contour":
                args = [
                    grid[0].sample_points[self.parameters[0]],
                    grid[0].sample_points[self.parameters[1]],
                    np.exp(
                        grid[0].marginalize_ln_posterior(not_parameters=self.parameters)
                        - grid[1]
                    ),
                ]
            else:
                args = [
                    grid[0].sample_points[self.parameters[0]],
                    grid[0].sample_points[self.parameters[1]],
                    np.exp(
                        grid[0].marginalize_ln_posterior(
                            not_parameters=self.parameters[0]
                        )
                        - grid[1]
                    ),
                    np.exp(
                        grid[0].marginalize_ln_posterior(
                            not_parameters=self.parameters[1]
                        )
                        - grid[1]
                    ),
                    np.exp(
                        grid[0].marginalize_ln_posterior(not_parameters=self.parameters)
                        - grid[1]
                    ),
                ]

            # set orientation of the 2D grid
            p1idx = grid[0].parameter_names.index(self.parameters[0])
            p2idx = grid[0].parameter_names.index(self.parameters[1])
            if p1idx < p2idx:
                # transpose density
                args[-1] = args[-1].T

            with DisableLogger():
                fig = plotfunc(*args, **kwargs)

        # update the legend
        if (existingfig or len(self._grids) > 1) and (
            "triangle" in self.plottype or "contour" == self.plottype
        ):
            from matplotlib.lines import Line2D

            if "triangle" in self.plottype:
                legax = fig[3] if self.plottype == "triangle" else fig[1]
            else:
                legax = fig.gca()
            curhandles, labels = legax.get_legend_handles_labels()
            handles = []
            for handle, label in zip(curhandles, labels):
                # switch any Patches labels to Line2D objects
                legcolor = (
                    handle.get_color()
                    if isinstance(handle, Line2D)
                    else handle.get_edgecolor()
                )
                handles.append(Line2D([], [], color=legcolor, label=label))

            # axis to output legend
            if "triangle" in self.plottype:
                # add any additional labels from grid
                kdeax = fig[1]
                for i, label in enumerate(self._grids):
                    for line in kdeax.get_lines():
                        linecolor = line.get_color()
                        # test that colours are the same
                        if linecolor == colors[i]:
                            handles.append(Line2D([], [], color=linecolor, label=label))
                            break
                    fig[2].legend(handles=handles, frameon=False, loc="best")
            else:
                leghandlemaps = legax.get_legend().get_legend_handler_map()

                for i, (label, grid) in enumerate(self._grids.items()):
                    if colors[i] in COLOR_MAP:
                        cmap = COLOR_MAP[colors[i]]
                    else:
                        cmap = kwargs.get("cmap", "viridis")

                    handles.append(Rectangle((0, 0), 1, 1, label=label))
                    leghandlemaps[handles[-1]] = HandlerColormap(cmap)

                # re-draw legend
                legax.legend(
                    handles=handles,
                    handler_map=leghandlemaps,
                    frameon=False,
                    loc="best",
                )

        kwargs = origkwargs

        return fig

    def _nd_plot_samples(self, **kwargs):
        """
        Create ND (where N > 2) plots from posterior samples.
        """

        from pesummary.core.plots.plot import (
            _make_comparison_corner_plot as plotfunc,
        )

        args = [self._samples]
        kwargs["corner_parameters"] = self.parameters
        if "latex_labels" not in kwargs:
            kwargs["latex_labels"] = self.latex_labels

        if "plot_percentile" not in kwargs:
            kwargs["plot_percentile"] = False

        # get ranges for each parameter to set figure axes extents
        if "range" not in kwargs:
            range = []
            for param in self.parameters:
                range.append(
                    [
                        np.min(
                            [samps[param].min() for samps in self._samples.values()]
                        ),
                        np.max(
                            [samps[param].max() for samps in self._samples.values()]
                        ),
                    ]
                )
            kwargs["range"] = range

        # default to not show quantile lines
        if "quantiles" not in kwargs:
            kwargs["quantiles"] = None

        # set default injection line color
        if "truth_color" not in kwargs:
            kwargs["truth_color"] = "k"

        # set injection parameter values
        if self.injection_parameters is not None:
            injpars = [
                self.injection_parameters[p]
                for p in self.parameters
                if self.injection_parameters[p] is not None
            ]
            if len(injpars) == self._num_parameters:
                kwargs["truths"] = injpars

        # create plot
        with DisableLogger():
            fig = plotfunc(*args, **kwargs)

        # turn frame off on legend
        fig.legends[0].set_frame_on(False)

        return fig

    def _nd_plot_grid(self, **kwargs):
        """
        Create ND (where N > 2) plots from posterior grids. These can only be
        added to existing corner plot figure axes.
        """

        from matplotlib.lines import Line2D
        from pesummary.core.plots.publication import pcolormesh

        # only add to corner plot if plotting on an existing figure
        if "fig" not in kwargs:
            raise TypeError(
                "Can only add Grid results to an existing corner plot showing samples"
            )

        colors = kwargs.pop("colors")

        fig = kwargs.pop("fig")
        ax = fig.axes

        quantiles = kwargs.pop("quantiles", None)

        grid2d = kwargs.pop("grid2d", False)

        for i, (label, grid) in enumerate(self._grids.items()):
            plotkwargs = {}
            plotkwargs["color"] = colors[i]
            plotkwargs["label"] = label

            axidx = 0
            for j, param in enumerate(self.parameters):
                x = grid[0].sample_points[param]
                pdf = np.exp(
                    grid[0].marginalize_ln_posterior(not_parameters=param) - grid[1]
                )

                ax[axidx].plot(x, pdf, **plotkwargs)

                if quantiles is not None:
                    low, high = self._credible_interval_grid(
                        grid[0], param, interval=quantiles
                    )
                    ax[axidx].axvline(low, color=colors[i], ls="--")
                    ax[axidx].axvline(high, color=colors[i], ls="--")

                # plot 2D posteriors
                if grid2d:
                    meshkwargs = {}
                    meshkwargs["zorder"] = kwargs.get("zorder", -10)
                    meshkwargs["shading"] = kwargs.get("shading", "gouraud")

                    if "cmap" not in kwargs:
                        if colors[i] in COLOR_MAP:
                            meshkwargs["cmap"] = COLOR_MAP[colors[i]]
                    else:
                        meshkwargs["cmap"] = kwargs["cmap"]

                    for k in range(j + 1, self._num_parameters):
                        y = grid[0].sample_points[self.parameters[k]]
                        density = np.exp(
                            grid[0].marginalize_ln_posterior(
                                not_parameters=[param, self.parameters[k]]
                            )
                            - grid[1]
                        )

                        # set orientation of the 2D grid
                        p1idx = grid[0].parameter_names.index(param)
                        p2idx = grid[0].parameter_names.index(self.parameters[k])
                        if p1idx < p2idx:
                            # transpose density
                            density = density.T

                        axyidx = axidx + (k - j) * self._num_parameters
                        pcolormesh(x, y, density, ax=ax[axyidx], **meshkwargs)

                axidx += self._num_parameters + 1

        # update the legend
        handles = []
        for legtext, leghandle in zip(
            fig.legends[0].texts, fig.legends[0].legendHandles
        ):
            label = legtext.get_text()
            legcolor = leghandle.get_color()

            handles.append(Line2D([], [], color=legcolor, label=label))

        for i, label in enumerate(self._grids):
            for line in ax[0].get_lines():
                linecolor = line.get_color()
                # test that colours are the same
                if linecolor == colors[i]:
                    handles.append(Line2D([], [], color=linecolor, label=label))
                    break

        # remove original legend
        fig.legends = []

        # re-add legend
        fig.legend(handles=handles, frameon=False, loc="upper right")

        return fig

    def credible_interval(self, parameter, interval=[0.05, 0.95]):
        """
        Calculate the credible intervals for a given parameter.

        Parameters
        ----------
        parameter: str
            The name of the parameter for which the credible interval is required.
        interval: list
            The credible interval to output. This defaults to ``[0.05, 0.95]``,
            i.e., the 90% credible interval bounded between the 5% and 95%
            percentiles.

        Returns
        -------
        intervals: dict, list
            If data contains multiple result objects a dictionary will be
            returned containing intervals for each result. If results is a single
            object, a single interval list will be returned.
        """

        if parameter not in self.parameters:
            raise ValueError(f"Parameter '{parameter}' is not available")

        intervals = {}
        for key, value in self.results.items():
            if isinstance(value, Grid):
                intervals[key] = Plot._credible_interval_grid(
                    value, parameter, interval
                )
            else:
                credint = value.posterior[parameter].quantile(interval).to_list()
                intervals[key] = credint[0] if len(interval) == 1 else credint

        return list(intervals.values())[0] if len(self.results) == 1 else intervals

    @staticmethod
    def _credible_interval_grid(grid, parameter, interval):
        """
        Calculate the credible intervals for a given parameter for a bilby
        :class:`~bilby.core.grid.Grid` object.
        """

        from pesummary.utils.array import Array

        margpost = grid.marginalize_posterior(not_parameters=parameter)
        intervals = Array.percentile(
            grid.sample_points[parameter],
            weights=margpost,
            percentile=[100 * val for val in interval],
        )

        return intervals if len(interval) > 1 else intervals[0]

    def upper_limit(self, parameter, bound=0.95):
        """
        Calculate an upper credible interval limit for a given parameter.

        Parameters
        ----------
        parameter: str
            The name of the parameter for which the credible interval is required.
        bound: float
            The quantile value between 0 and 1 at which to calculate the upper
            credible bound.

        Returns
        -------
        upperlimit: dict, list
            If data contains multiple result objects a dictionary will be
            returned containing upper limits for each result. If data is a single
            object, a single upper limit will be returned.
        """

        return self.credible_interval(parameter, interval=[bound])


def _set_axes_limits(ax, parameter, axis="x"):
    """
    Define the limits of an axis range using the current limits or the default
    bounds if current limits are outside those bounds.
    """

    lims = list(ax.get_xlim()) if axis == "x" else list(ax.get_ylim())

    if "low" in DEFAULT_BOUNDS[parameter]:
        low = DEFAULT_BOUNDS[parameter]["low"]
        if lims[0] < low:
            lims[0] = DEFAULT_BOUNDS[parameter]["low"]
    if "high" in DEFAULT_BOUNDS[parameter]:
        high = DEFAULT_BOUNDS[parameter]["high"]
        if lims[1] > high:
            lims[1] = DEFAULT_BOUNDS[parameter]["high"]

    if axis == "x":
        ax.set_xlim(lims)
    else:
        ax.set_ylim(lims)


class DisableLogger:
    """
    Context manager class to disable logging propagated from PESummary.
    See https://stackoverflow.com/a/20251235/1862861.
    """

    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


class HandlerColormap(HandlerBase):
    """
    A legend handler to create a legend patch to represent a colour map.
    This is taken from https://stackoverflow.com/a/55501861/1862861.
    """

    def __init__(self, cmap, num_stripes=16, **kw):
        from matplotlib.cm import get_cmap

        HandlerBase.__init__(self, **kw)
        self.cmap = get_cmap(cmap)
        self.num_stripes = num_stripes

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle(
                [xdescent + i * width / self.num_stripes, ydescent],
                width / self.num_stripes,
                height,
                fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                transform=trans,
            )
            stripes.append(s)

        return stripes
