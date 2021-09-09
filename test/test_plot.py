"""
Test code plotting results.
"""

import os
import shutil

import numpy as np
import pytest
from bilby.core.grid import Grid
from bilby.core.result import Result, read_in_result
from cwinpy import PulsarParameters
from cwinpy.plot import DEFAULT_BOUNDS, Plot
from matplotlib.figure import Figure


class TestPlotting(object):
    """
    Test the plotting class.
    """

    @classmethod
    def setup_class(cls):
        # set results files containing data to plot
        cls.lppen = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "plotting_test_lppen.hdf",
        )

        cls.cwinpy = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "plotting_test_cwinpy.json.gz",
        )

        cls.grid = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "plotting_test_grid.json.gz",
        )

        # labels for each results set
        cls.allresults = {
            "lppen": cls.lppen,
            "CWInPy": cls.cwinpy,
            "Grid": cls.grid,
        }

        cls.pulsar = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "plotting_test.par",
        )

        cls.outputdir = "plots"
        os.makedirs(cls.outputdir, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        """
        Remove test output plots.
        """

        shutil.rmtree(cls.outputdir)

    def test_plot_inputs(self):
        """
        Test inputs to plotting class.
        """

        with pytest.raises(TypeError):
            # fails with no positional arguments
            _ = Plot()

        with pytest.raises(TypeError):
            # fails with position argument of the wrong type
            _ = Plot(1)

        # test reading in different objects
        for res in [self.lppen, read_in_result(self.cwinpy), Grid.read(self.grid)]:
            plot = Plot(res)
            assert len(plot.results) == 1

        # test reading in dictionary - parameters is not specified, so should
        # fail with inconsistent parameter error
        with pytest.raises(ValueError):
            _ = Plot(
                {
                    "lppen": self.lppen,
                    "CWInPy": read_in_result(self.cwinpy),
                    "grid": Grid.read(self.grid),
                }
            )

        # try reading non-existent files
        with pytest.raises(IOError):
            _ = Plot({"one": "no_file.hdf", "two": "blah.json"})

        plot = Plot(self.allresults, parameters=["h0", "psi"])
        assert len(plot.results) == len(self.allresults)

        for label, restype in zip(self.allresults.keys(), [Result, Result, Grid]):
            assert isinstance(plot.results[label], restype)

    def test_parameters(self):
        """
        Test input parameters for plotting.
        """

        # pass parameter with wrong type
        with pytest.raises(TypeError):
            _ = Plot(self.cwinpy, parameters=1)

        # pass unavailable parameter
        with pytest.raises(ValueError):
            _ = Plot(self.cwinpy, parameters="blah")

        # pass real parameter - should fail for default plot type "corner",
        # which cannot be used for a single parameter
        with pytest.raises(TypeError):
            _ = Plot(self.cwinpy, parameters="h0")

        plot = Plot(self.cwinpy, parameters="h0", plottype="hist")
        assert plot.parameters == ["h0"]
        assert plot.plottype == "hist"
        assert plot.fig is None

        # pass two parameters with plottype "kde" - should fail as this can
        # only be used for 1 parameter
        with pytest.raises(TypeError):
            _ = Plot(self.cwinpy, parameters=["h0", "cosiota"], plottype="kde")

        plot = Plot(self.cwinpy, parameters=["h0", "cosiota"])
        assert plot.parameters == ["h0", "cosiota"]
        assert plot.plottype == "corner"
        assert len(plot.results) == 1

        # pass three parameters with plot type "hist" - should fail for as this
        # can only be used for 1 parameter
        with pytest.raises(TypeError):
            _ = Plot(self.cwinpy, parameters=["h0", "cosiota", "psi"], plottype="hist")

        plot = Plot(self.cwinpy, parameters=["h0", "cosiota", "psi"], plottype="corner")
        assert plot.parameters == ["h0", "cosiota", "psi"]
        assert plot.plottype == "corner"
        assert len(plot.results) == 1

    def test_plottype(self):
        """
        Test plot type is valid.
        """

        with pytest.raises(TypeError):
            _ = Plot(self.cwinpy, parameters="h0", plottype="skdfjkfda")

    def test_untrig(self):
        """
        Test inverting trigonometric function.
        """

        # fail as cannot invert value for Grid
        with pytest.raises(TypeError):
            _ = Plot(self.grid, parameters="cosiota", plottype="hist", untrig="cosiota")

        plot = Plot(self.cwinpy, parameters="iota", untrig="cosiota", plottype="hist")
        assert plot.parameters == ["iota"]
        assert len(plot.results) == 1
        assert "iota" in plot.results["result"].posterior.columns
        assert all(
            (DEFAULT_BOUNDS["iota"]["low"] <= plot.results["result"].posterior["iota"])
            & (
                DEFAULT_BOUNDS["iota"]["high"]
                > plot.results["result"].posterior["iota"]
            )
        )

    def test_pulsar_parameters(self):
        """
        Test reading in pulsar parameters.
        """

        # fail if parameter file is not right
        with pytest.raises(IOError):
            _ = Plot(
                self.cwinpy,
                parameters="iota",
                untrig="cosiota",
                plottype="hist",
                pulsar=985348,
            )

        plot = Plot(
            self.cwinpy,
            parameters="iota",
            untrig="cosiota",
            plottype="hist",
            pulsar=self.pulsar,
        )

        assert list(plot.injection_parameters.keys()) == ["iota"]
        assert plot.injection_parameters["iota"] == np.arccos(
            PulsarParameters(self.pulsar)["COSIOTA"]
        )

    def test_plot_1d(self):
        """
        Test 1D plots.
        """

        # histogram
        plot = Plot(self.allresults, parameters="h0", plottype="hist")
        assert len(plot.results) == len(self.allresults)
        assert plot.plottype == "hist"

        fig = plot.plot()  # create plot
        assert isinstance(fig, Figure) and isinstance(plot.fig, Figure)

        plot.save("oned_hist.png", dpi=150)
        fig.clf()
        fig.close()
        del plot

        # histogram with KDE
        plot = Plot(self.allresults, parameters="h0", plottype="hist", kde=True)
        assert len(plot.results) == len(self.allresults)
        assert plot.plottype == "hist"

        fig = plot.plot()  # create plot
        assert isinstance(fig, Figure) and isinstance(plot.fig, Figure)

        plot.save("oned_hist_with_kde.png", dpi=150)
        fig.clf()
        fig.close()
        del plot

        # KDE
        plot = Plot(self.allresults, parameters="h0", plottype="kde")
        assert len(plot.results) == len(self.allresults)
        assert plot.plottype == "kde"

        fig = plot.plot()  # create plot
        assert isinstance(fig, Figure) and isinstance(plot.fig, Figure)

        plot.save("oned_kde.png", dpi=150)
        fig.clf()
        fig.close()
        del plot

    def test_plot_2d(self):
        """
        Test 2D plots.
        """

        # corner
        plot = Plot(self.allresults, parameters=["h0", "cosiota"], plottype="corner")
        assert len(plot.results) == len(self.allresults)
        assert plot.plottype == "corner"

        fig = plot.plot()  # create plot
        assert isinstance(fig, Figure) and isinstance(plot.fig, Figure)
        assert len(plot.fig.axes) == 4

        plot.save("twod_corner.png", dpi=150)
        fig.clf()
        fig.close()
        del plot

        # triangle
        plot = Plot(self.allresults, parameters=["h0", "cosiota"], plottype="triangle")
        assert len(plot.results) == len(self.allresults)
        assert plot.plottype == "triangle"

        fig = plot.plot()  # create plot
        assert isinstance(fig, Figure) and isinstance(plot.fig, Figure)
        assert len(plot.fig.axes) == 4

        plot.save("twod_triangle.png", dpi=150)
        fig.clf()
        fig.close()
        del plot

        # reverse triangle
        plot = Plot(
            self.allresults, parameters=["h0", "cosiota"], plottype="reverse_triangle"
        )
        assert len(plot.results) == len(self.allresults)
        assert plot.plottype == "reverse_triangle"

        fig = plot.plot()  # create plot
        assert isinstance(fig, Figure) and isinstance(plot.fig, Figure)
        assert len(plot.fig.axes) == 4

        plot.save("twod_reverse_triangle.png", dpi=150)
        fig.clf()
        fig.close()
        del plot

        # contour
        plot = Plot(self.allresults, parameters=["h0", "cosiota"], plottype="contour")
        assert len(plot.results) == len(self.allresults)
        assert plot.plottype == "contour"

        fig = plot.plot()  # create plot
        assert isinstance(fig, Figure) and isinstance(plot.fig, Figure)
        assert len(plot.fig.axes) == 1

        plot.save("twod_contour.png", dpi=150)
        fig.clf()
        fig.close()
        del plot

    def test_plot_nd(self):
        """
        Test mulitple dimension plots.
        """

        # corner
        plot = Plot(
            self.allresults,
            parameters=["h0", "cosiota", "psi", "phi0"],
            plottype="corner",
        )
        assert len(plot.results) == len(self.allresults)
        assert plot.plottype == "corner"

        fig = plot.plot()  # create plot
        assert isinstance(fig, Figure) and isinstance(plot.fig, Figure)
        assert len(plot.fig.axes) == 16

        plot.save("nd_corner.png", dpi=150)
        fig.clf()
        fig.close()
        del plot

    def test_credible_interval(self):
        """
        Test credible interval calculation.
        """

        plot = Plot(
            self.allresults["CWInPy"],
            parameters=["h0", "cosiota", "psi", "phi0"],
            plottype="corner",
        )

        with pytest.raises(ValueError):
            # invalid parameter
            _ = plot.credible_interval(parameter="skdg")

        interval = plot.credible_interval("psi")
        assert len(interval) == 2
        assert interval[0] < interval[1]
        assert (
            interval[0] > DEFAULT_BOUNDS["psi"]["low"]
            and interval[1] < DEFAULT_BOUNDS["psi"]["high"]
        )

        # test upper limit
        ul = plot.upper_limit("h0")

        assert isinstance(ul, float)
        assert ul > 0.0

        plot = Plot(
            self.allresults,
            parameters=["h0", "cosiota", "psi", "phi0"],
            plottype="corner",
        )
        interval = plot.credible_interval(parameter="cosiota")
        assert len(interval) == len(self.allresults)

        for label in self.allresults:
            credint = interval[label]
            assert (
                credint[0] > DEFAULT_BOUNDS["cosiota"]["low"]
                and credint[1] < DEFAULT_BOUNDS["cosiota"]["high"]
            )

        uls = plot.upper_limit("h0")
        for label in self.allresults:
            assert isinstance(uls[label], float)
            assert uls[label] > 0.0
