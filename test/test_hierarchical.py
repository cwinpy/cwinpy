"""
Test script for the hierarchical classes.
"""

import os

import numpy as np
import pytest
from bilby.core.grid import Grid
from bilby.core.prior import Uniform
from bilby.core.result import Result, ResultList
from cwinpy.hierarchical import (
    BaseDistribution,
    BoundedGaussianDistribution,
    ExponentialDistribution,
    MassQuadrupoleDistribution,
    create_distribution,
)


class TestDistributionObjects(object):
    """
    Tests for the distribution objects.
    """

    def test_base_distribution(self):
        """
        Test the BaseDistribution object.
        """

        name = "test"

        # test failure for unknown distribution
        with pytest.raises(ValueError):
            BaseDistribution(name, "kjsgdkdgkjgsda")

        # test failure for inappropriate bounds
        with pytest.raises(ValueError):
            BaseDistribution(name, "gaussian", low=0.0, high=-1.0)

        # test failure for unknown hyperparameter name
        with pytest.raises(KeyError):
            hyper = {"mu": [1], "dkgwkufd": [2]}
            BaseDistribution(name, "gaussian", hyperparameters=hyper)

        # test failure with invalid hyperparameter type
        with pytest.raises(TypeError):
            BaseDistribution(name, "gaussian", hyperparameters="blah")

        with pytest.raises(TypeError):
            hyper = "blah"
            BaseDistribution(name, "exponential", hyperparameters=hyper)

        # test default log_pdf is NaN
        hyper = {"mu": 2.0}
        dist = BaseDistribution(name, "exponential", hyperparameters=hyper)

        assert dist["mu"] == hyper["mu"]
        assert np.isnan(dist.log_pdf({}, 0))
        assert dist.sample({}) is None

        # test failure when getting unknown item
        with pytest.raises(KeyError):
            _ = dist["kgksda"]

        del dist

        # test setter failure
        dist = BaseDistribution(name, "exponential")

        with pytest.raises(KeyError):
            dist["madbks"] = Uniform(0.0, 1.0, "mu")

        # test setter
        dist["mu"] = Uniform(0.0, 1.0, "mu")
        assert isinstance(dist["mu"], Uniform)

    def test_bounded_gaussian(self):
        """
        Test the BoundedGaussianDistribution class.
        """

        name = "test"

        # test failures
        with pytest.raises(TypeError):
            BoundedGaussianDistribution(name, mus="blah")

        with pytest.raises(TypeError):
            BoundedGaussianDistribution(name, mus=[1], sigmas="blah")

        with pytest.raises(TypeError):
            BoundedGaussianDistribution(name, mus=[1], sigmas=[1], weights="blah")

        with pytest.raises(ValueError):
            BoundedGaussianDistribution(name, mus=[1.0], sigmas=[1.0, 2.0])

        with pytest.raises(ValueError):
            BoundedGaussianDistribution(
                name, mus=[1.0, 2.0], sigmas=[1.0, 2.0], weights=[1]
            )

        with pytest.raises(ValueError):
            BoundedGaussianDistribution(name)

        dist = BoundedGaussianDistribution(name, mus=[1.0, 2.0], sigmas=[1.0, 2.0])

        assert dist.nmodes == 2
        assert np.all(np.array(dist["weight"]) == 1.0)

        del dist
        # test log pdf
        dist = BoundedGaussianDistribution(
            name,
            mus=[Uniform(0.0, 1.0, "mu0"), 2.0],
            sigmas=[Uniform(0.0, 1.0, "sigma0"), 2.0],
            weights=[Uniform(0.0, 1.0, "weight0"), 2.0],
        )

        value = 1.0
        hyper = {"mu8": 0.5, "sigma0": 0.5, "weight0": 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(hyper, value)

        hyper = {"mu0": 0.5, "sigma8": 0.5, "weight0": 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(hyper, value)

        hyper = {"mu0": 0.5, "sigma0": 0.5, "weight8": 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(hyper, value)

        hyper = {"mu0": 0.5, "sigma0": 0.5, "weight0": 0.5}
        assert np.isfinite(dist.log_pdf(hyper, value))
        assert np.exp(dist.log_pdf(hyper, value)) == dist.pdf(hyper, value)

        # check negative values give -inf by default
        value = -1.0
        assert dist.log_pdf(hyper, value) == -np.inf

        # check drawn sample is within bounds
        assert dist.low < dist.sample(hyper) < dist.high

        # draw multiple samples
        N = 100
        samples = dist.sample(hyper, size=N)
        assert len(samples) == N
        assert np.all((samples > dist.low) & (samples < dist.high))

        # test with multiple modes
        del dist
        dist = BoundedGaussianDistribution(
            name,
            mus=[Uniform(0.0, 1.0, "mu0"), Uniform(0.0, 1.0, "mu1")],
            sigmas=[Uniform(0.0, 1.0, "sigma0"), Uniform(0.0, 1.0, "sigma1")],
            weights=[0.25, 0.75],
        )

        hyper = {"mu0": 0.5, "sigma0": 0.5, "mu1": 0.7, "sigma1": 0.8}

        N = 100
        samples = dist.sample(hyper, size=N)
        assert len(samples) == N
        assert np.all((samples > dist.low) & (samples < dist.high))

    def test_exponential(self):
        """
        Test the ExponentialDistribution class.
        """

        name = "test"

        with pytest.raises(TypeError):
            ExponentialDistribution(name, mu=1.0)

        dist = ExponentialDistribution(name, mu=Uniform(0.0, 1.0, "mu"))

        value = -1.0
        hyper = {"mu": 0.5}
        assert dist.log_pdf(hyper, value) == -np.inf
        assert np.exp(dist.log_pdf(hyper, value)) == dist.pdf(hyper, value)

        # check drawn sample is within bounds
        assert dist.low < dist.sample(hyper) < dist.high

        # draw multiple samples
        N = 100
        samples = dist.sample(hyper, size=N)
        assert len(samples) == N
        assert np.all((samples > dist.low) & (samples < dist.high))

        value = 1.0
        hyper = {"kgsdg": 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(hyper, value)

    def test_create_distribution(self):
        """
        Test the create_distribution() function.
        """

        name = "test"
        with pytest.raises(ValueError):
            create_distribution(name, "kjbskdvakvkd")

        with pytest.raises(TypeError):
            create_distribution(name, 2.3)

        gausskwargs = {"mus": [1.0, 2.0], "sigmas": [1.0, 2.0]}
        dist = create_distribution(name, "Gaussian", gausskwargs)

        assert isinstance(dist, BoundedGaussianDistribution)
        assert (
            dist["mu0"] == gausskwargs["mus"][0]
            and dist["mu1"] == gausskwargs["mus"][1]
        )
        assert (
            dist["sigma0"] == gausskwargs["sigmas"][0]
            and dist["sigma1"] == gausskwargs["sigmas"][1]
        )
        del dist

        expkwargs = {"mu": Uniform(0.0, 1.0, "mu")}
        dist = create_distribution(name, "Exponential", expkwargs)
        assert isinstance(dist, ExponentialDistribution)
        assert dist["mu"] == expkwargs["mu"]

        newdist = create_distribution(name, dist)
        assert isinstance(newdist, ExponentialDistribution)
        assert newdist["mu"] == dist["mu"]


class TestMassQuadrupoleDistribution(object):
    """
    Test the MassQuadrupoleDistribution object.
    """

    def test_mass_quadrupole_distribution(self):
        # test data sets from bilby
        testdata1 = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "hierarchical_test_set_0_result.json",
        )
        testdata2 = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "hierarchical_test_set_1_result.json",
        )

        # test invalid q22range (lower bounds is less than upper bound)
        with pytest.raises(ValueError):
            MassQuadrupoleDistribution(q22range=[100.0, 1.0])

        # test invalid q22range (only one value passed)
        with pytest.raises(ValueError):
            MassQuadrupoleDistribution(q22range=[100.0])

        # test invalid data type
        with pytest.raises(TypeError):
            MassQuadrupoleDistribution(data=1)

        res = ResultList([testdata1, testdata2])

        # remove Q22 from results to test error
        del res[0].posterior["q22"]
        with pytest.raises(RuntimeError):
            MassQuadrupoleDistribution(data=res)

        # distribution with wrong name (i.e., not 'Q22')
        pdist = ExponentialDistribution("Blah", mu=Uniform(0.0, 1e37, "mu"))
        with pytest.raises(ValueError):
            MassQuadrupoleDistribution(data=[testdata1, testdata2], distribution=pdist)

        # distribution with no priors to infer
        pdist = BoundedGaussianDistribution("Q22", mus=[0.0], sigmas=[1e34])
        with pytest.raises(ValueError):
            MassQuadrupoleDistribution(data=[testdata1, testdata2], distribution=pdist)

        # unknown sampler type
        pdist = ExponentialDistribution("Q22", mu=Uniform(0.0, 1e32, "mu"))
        with pytest.raises(ValueError):
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2], distribution=pdist, sampler="akgkfsfd"
            )

        # unknown bandwidth type for KDE
        bw = "lkgadkgds"
        with pytest.raises(RuntimeError):
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2], distribution=pdist, bw=bw
            )

        # wrong type for q22grid
        q22grid = "lsgdgkavbc"
        with pytest.raises(TypeError):
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2], distribution=pdist, q22grid=q22grid
            )

        # test sampler
        mdist = MassQuadrupoleDistribution(
            data=[testdata1, testdata2], distribution=pdist
        )
        res = mdist.sample(**{"Nlive": 100, "save": False})

        assert isinstance(res, Result)

        del res
        del mdist

        # test grid sampler
        grid = "Blah"
        with pytest.raises(TypeError):
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2], distribution=pdist, grid=grid
            )

        grid = {"mu": 100}
        mdist = MassQuadrupoleDistribution(
            data=[testdata1, testdata2], distribution=pdist, grid=grid
        )

        res = mdist.sample()
        assert isinstance(res, Grid)
