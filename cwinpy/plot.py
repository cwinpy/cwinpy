"""
Classes/functions to plot posterior results from known pulsar analyses.
"""

import numpy as np
from bilby.core.grid import Grid
from bilby.core.result import Result, read_in_result
from cwinpy.utils import lalinference_to_bilby_result
from gwpy.plot.colors import GW_OBSERVATORY_COLORS
from pesummary.conf import colorcycle

# from pesummary.core.plots.bounded_1d_kde import (
#    ReflectionBoundedKDE,
#    TransformBoundedKDE,
# )

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
}

#: dictionary of default parameter bounds
DEFAULT_BOUNDS = {
    "h0": {"low": 0.0},
    "c21": {"low": 0.0},
    "c22": {"low": 0.0},
    "cosiota": {"low": -1.0, "high": 1.0},
    "siniota": {"low": 0.0, "high": 1.0},
    "psi": {"low": 0.0, "high": np.pi / 2},
    "iota": {"low": 0.0, "high": np.pi},
    "phi0": {"low": 0.0, "high": np.pi},
    "phi21": {"low": 0.0, "high": 2 * np.pi},
    "phi22": {"low": 0.0, "high": np.pi},
}

#: list of parameters for which to use the TransformBoundedKDE from PEsummary
TRANSFORM_KDE_PARAMS = ["iota"]

#: list of allowed 1d plot types
ONED_PLOT_TYPES = ["hist", "kde"]

#: list of allowed 2d plot types
TWOD_PLOT_TYPES = ["contour", "triangle", "reverse_triangle", "corner"]

#: list of allowed nd plot types
MULTID_PLOT_TYPES = ["corner"]


class Plot(object):
    def __init__(
        self,
        results,
        parameters=None,
        plottype="corner",
        latex_labels=None,
        kde=False,
        untrig=None,
    ):
        """
        A class to plot individual or joint posterior distributions using a
        variety of plotting functions.

        .. note::

           Results from a :class:`~bilby.core.grid.Grid` object can only be
           plotted on a "corner", "hist", or "kde" type plot. In the case of a
           "corner" plot, the :class:`~bilby.core.grid.Grid` results can only
           be plotted together with samples from a
           :class:`~bilby.core.result.Result` object, and the
           :class:`~bilby.core.result.Result` object must be before the
           :class:`~bilby.core.grid.Grid` results in the input ``results``
           dictionary. If plotting :class:`~bilby.core.grid.Grid` results on
           their own, only single parameters can be specified.

        Parameters
        ----------
        results: str, dict
            Pass the results to be plotted. This can either be in the form of
            a string giving the path to the results file, or directly passing
            bilby :class:`~bilby.core.result.Result` or
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
        parfile: str
            The path to a TEMPO(2)-style pulsar parameter file. If given, and
            containing values of the requested parameters (such as for a
            simulated signal), then these will be over-plotted on the output.
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
            If plotting a histogram using "hist", set this to True to also plot
            the KDE. Use the "kde" `plottype` to only plot the KDE.
        untrig: str, list
            A string, or list, of parameters that exist are defined as the
            trigonometric function of another parameters. If given in the list
            then those parameters will be inverted, e.g., if "cosiota" is
            present it will be changed to be "iota". This only works for result
            samples and not grid values. Default is None.
        """

        self.untrig = untrig
        self.results = results
        self.parameters = parameters
        self.plottype = plottype
        self.latex_labels = latex_labels
        self.kde = kde

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
            checkparams = list(self._results_parameters.values)[0]
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
    def latex_labels(self, labels):
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

    def plot(self, **kwargs):
        """
        Create the plot of the data.

        Parameters
        ----------
        colors: dict
            A dictionary of colour codes keyed to the keys of the ``results``.
            By default the GWpy colour scheme will be used for results keyed by
            GW detector prefixes. If keys are not known detector prefixes then
            the PESummary default color cycle will be used.
        """

        # don't add percentile title to figure by default
        if "title" not in kwargs:
            kwargs["title"] = False

        # get colors
        colors = kwargs.get("colors", GW_OBSERVATORY_COLORS)

        # get Result samples
        samples = {
            label: value.posterior
            for label, value in self.results.items()
            if isinstance(value, Result)
        }

        # get Grid posteriors
        grids = {
            label: [value, value.ln_evidence]  # store grid and log evidence
            for label, value in self.results.items()
            if isinstance(value, Grid)
        }

        colordicts = []
        for j, res in enumerate([samples, grids]):
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
        if len(samples) > 0:
            kwargs["colors"] = list(colordicts[0].values())
            if self._num_parameters == 1:
                fig = self._1d_plot_samples(samples, **kwargs)
            if self._num_parameters == 2:
                fig = self._2d_plot_samples(samples, **kwargs)

        # restore keywords
        kwargs = origkwargs

        if len(grids) > 0:
            kwargs["colors"] = list(colordicts[1].values())
            if fig is not None and "fig" not in kwargs:
                kwargs["fig"] = fig
            if self._num_parameters == 1:
                fig = self._1d_plot_grid(grids, **kwargs)
            if self._num_parameters == 2:
                fig = self._2d_plot_grid(grids, **kwargs)

        # for key, res in self.results.items():
        #    samps = isinstance(res, Result)
        #    if self._num_parameters == 1:
        #        if "latex_label" not in kwargs:
        #            kwargs["latex_label"] = list(self.latex_labels.values())

        # set function dependent on whether a Grid is used or not
        #        func = _1d_analytic_plot if not samps else self.plotfunction

        #        if not samps:
        #            kwargs["x"] = res.sample_points[self.parameters[0]]
        #            kwargs["pdf"] = res.marginalize_posterior(
        #                not_parameters=self.parameters[0]
        #            )
        #        else:
        #            kwargs["samples"] = res.posterior[self.parameters[0]]

        #        fig = func(fig=fig, param=self.parameters[0], **kwargs)

        #        if not samps:
        #            kwargs.pop("x")
        #            kwargs.pop("pdf")
        #        else:
        #            kwargs.pop("samples")
        #    else:
        #        if "latex_labels" not in kwargs and self.plottype == "corner":
        #            kwargs["latex_labels"] = self.latex_labels
        #        elif self.plottype != "corner":
        #            kwargs["xlabel"] = self.latex_labels[self.parameters[0]]
        #            kwargs["ylabel"] = self.latex_labels[self.parameters[1]]

        # create multi-D plots is using samples from a Result object
        #        if self.plottype == "corner" and samps:
        #            kwargs["samples"] = res.posterior

        #            if "hist_kwargs" not in kwargs:
        #                # make sure histograms are normalised
        #                kwargs["hist_kwargs"] = {"density": True}

        #            fig, _, _ = self.plotfunction(
        #                fig=fig, corner_parameters=self.parameters, **kwargs
        #            )
        #            kwargs.pop("samples")
        #        elif self.plottype != "corner" and samps:
        #            kwargs["x"] = res.posterior[self.parameters[0]].values
        #            kwargs["y"] = res.posterior[self.parameters[1]].values

        #            if self.plottype == "triangle":
        #                if isinstance(fig, tuple):
        #                    kwargs["existing_figure"] = fig
        #            else:
        #                kwargs["fig"] = fig

        #            fig = self.plotfunction(**kwargs)
        #            kwargs.pop("x")
        #            kwargs.pop("y")
        #        elif (
        #            self.plottype == "corner" and not samps and len(fig.get_axes()) > 1
        #        ):
        # add Grid to existing corner plot axes

        # loop over parameters
        #            ax = fig.get_axes()
        #            axidx = 0
        #            idxstep = self._num_parameters + 1
        #            for param in self.parameters:
        #                kwargs["x"] = res.sample_points[param]
        #                kwargs["pdf"] = res.marginalize_posterior(not_parameters=param)
        #                fig = _1d_analytic_plot(
        #                    fig=fig, ax=ax[axidx], param=param, **kwargs
        #                )
        #                kwargs.pop("x")
        #                kwargs.pop("pdf")
        #                axidx += idxstep
        #        else:
        #            raise RuntimeError("Problem creating plot")

        return fig

    def _1d_plot_samples(self, samples, **kwargs):
        """
        Create 1D plots from posterior samples.
        """

        from pesummary.core.plots.plot import _1d_comparison_histogram_plot

        if self.plottype == "kde":
            # set for plotting a KDE
            kwargs["kde"] = True
            kwargs["hist"] = False
        elif self.plottype == "hist":
            # set whether or not to also plot the KDE
            kwargs["kde"] = self.kde

        if kwargs["kde"]:
            # required to allow KDE generation when sample values are small (see
            # https://git.ligo.org/lscsoft/pesummary/-/merge_requests/604)
            # NOTE: this will only be available when PESummary >v0.12.1 is released
            if "kde_kwargs" in kwargs:
                kwargs["kde_kwargs"].update({"variance_atol": 0.0})
            else:
                kwargs["kde_kwargs"] = {"variance_atol": 0.0}

        singlesamps = [samples[key][self.parameters[0]].values for key in samples]

        origkwargs = kwargs.copy()
        colors = kwargs.pop("colors")
        kwargs.pop(
            "title"
        )  # already set to False in _1d_comparison_histogram_plot function

        fig = _1d_comparison_histogram_plot(
            self.parameters[0],
            singlesamps,
            colors,
            self.latex_labels[self.parameters[0]],
            list(samples.keys()),
            **kwargs,
        )

        kwargs = origkwargs

        return fig

    def _1d_plot_grid(self, grids, **kwargs):
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

        colors = kwargs.pop("colors")

        for i, (label, grid) in enumerate(grids.items()):
            x = grid[0].sample_points[self.parameters[0]]

            # NOTE: this will only be available when PESummary >v0.12.1 is released
            # following merge of https://git.ligo.org/lscsoft/pesummary/-/merge_requests/606
            pdf = np.exp(
                grid[0].marginalize_ln_posterior(not_parameters=self.parameters[0])
                - grid[1]
            )
            kwargs["label"] = label
            kwargs["color"] = colors[i]

            fig = _1d_analytic_plot(
                self.parameters[0],
                x,
                pdf,
                self.latex_labels[self.parameters[0]],
                ax=ax,
                **kwargs,
            )

        # update the legend
        if existingfig or len(grids) > 1:
            from matplotlib.lines import Line2D

            curhandles, labels = ax.get_legend_handles_labels()
            handles = []
            for handle, label in zip(curhandles, labels):
                # switch any Patches labels to Line2D objects
                legcolor = (
                    handle.get_color()
                    if isinstance(handle, Line2D)
                    else handle.get_edgecolor()
                )
                handles.append(
                    Line2D([], [], linewidth=1.75, color=legcolor, label=label)
                )

            ax.legend(handles=handles)

        kwargs = origkwargs

        return fig

    def _2d_plot_samples(self, samples, **kwargs):
        """
        Create 2D plots from posterior samples.
        """
        if self.plottype == "corner":
            from pesummary.core.plots.plot import (
                _make_comparison_corner_plot as plotfunc,
            )

            args = [samples]
            kwargs["corner_parameters"] = self.parameters
            if "latex_labels" not in kwargs:
                kwargs["latex_labels"] = self.latex_labels

            # make sure histograms are normalised
            if "hist_kwargs" not in kwargs:
                kwargs["hist_kwargs"] = {"density": True}

            # get ranges for each parameter to set figure axes extents
            if "range" not in kwargs:
                range = []
                for param in self.parameters:
                    range.append(
                        [
                            np.min([samps[param].min() for samps in samples.values()]),
                            np.max([samps[param].max() for samps in samples.values()]),
                        ]
                    )
                kwargs["range"] = range
        elif "triangle" in self.plottype or "contour" in self.plottype:
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

                # until https://git.ligo.org/lscsoft/pesummary/-/merge_requests/600 is merged
                # linestyles will need to be set
                if "linestyles" not in kwargs:
                    kwargs["linestyles"] = ["-"] * len(samples)

            args = [
                [samps[self.parameters[0]].values for samps in samples.values()],
                [samps[self.parameters[1]].values for samps in samples.values()],
            ]

            if "xlabel" not in kwargs:
                kwargs["xlabel"] = self.latex_labels[self.parameters[0]]
            if "ylabel" not in kwargs:
                kwargs["ylabel"] = self.latex_labels[self.parameters[1]]

        if "labels" not in kwargs and len(samples) > 1:
            kwargs["labels"] = list(samples.keys())

        # create plot
        fig = plotfunc(*args, **kwargs)

        return fig

    def _2d_plot_grid(self, grids, **kwargs):
        """
        Create 2D plot from posterior grids.
        """
        if self.plottype == "contour":
            from pesummary.core.plots.publication import (
                analytic_twod_contour_plot as plotfunc,
            )

            if "fig" in kwargs:
                ax = kwargs["fig"].gca()
                kwargs["ax"] = ax

            for label, grid in grids:
                x = grid[0].sample_points[self.parameters[0]]
                y = grid[0].sample_points[self.parameters[1]]
                density = np.exp(grid.ln_posterior - grid[1])

                fig = plotfunc(x, y, density, label=label, **kwargs)
        elif self.plottype == "triangle":
            from pesummary.core.plots.publication import (
                analytic_triangle_plot as plotfunc,
            )

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
        for key, value in self.results:
            if isinstance(value, Grid):
                intervals[key] = Plot._credible_interval_grid(
                    value, parameter, interval
                )
            else:
                intervals[key] = value.posterior[parameter].quantile(interval).to_list()

        return list(intervals.values())[0] if len(self.results) == 1 else intervals

    @staticmethod
    def _credible_interval_grid(grid, parameter, interval):
        """
        Calculate the credible intervals for a given parameter for a bilby
        :class:`~bilby.core.grid.Grid` object.
        """

        from pesummary.utils.array import Array

        margpost = grid.marginalize_posterior(not_parameters=parameter)
        intervals = Array(grid.sample_points[parameter], weights=margpost).percentile(
            [100 * val for val in interval]
        )

        return intervals

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
