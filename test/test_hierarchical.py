"""
Test script for the hierarchical classes.
"""

import os

import numpy as np
import pytest
from bilby.core.grid import Grid
from bilby.core.prior import DirichletPriorDict, Uniform
from bilby.core.result import Result, ResultList
from cwinpy.hierarchical import (
    BaseDistribution,
    BoundedGaussianDistribution,
    DeltaFunctionDistribution,
    ExponentialDistribution,
    MassQuadrupoleDistribution,
    PowerLawDistribution,
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
            BaseDistribution(name, "exponential", hyperparameters="blah")

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

        # test failure with invalid hyperparameter type
        with pytest.raises(TypeError):
            hyper = "blah"
            BaseDistribution(name, "deltafunction", hyperparameters=hyper)

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

        with pytest.raises(TypeError):
            BoundedGaussianDistribution(name, mus=[1], sigmas=[1], weights=23.4)

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
            weights=DirichletPriorDict(n_dim=2, label="weight"),
        )

        value = 1.0
        hyper = {"mu8": 0.5, "sigma0": 0.5, "weight0": 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(value, hyper)

        hyper = {"mu0": 0.5, "sigma8": 0.5, "weight0": 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(value, hyper)

        hyper = {"mu0": 0.5, "sigma0": 0.5, "weight8": 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(value, hyper)

        hyper = {"mu0": 0.5, "sigma0": 0.5, "weight0": 0.5}
        assert np.isfinite(dist.log_pdf(value, hyper))
        assert np.exp(dist.log_pdf(value, hyper)) == dist.pdf(value, hyper)

        # check negative values give -inf by default
        value = -1.0
        assert dist.log_pdf(value, hyper) == -np.inf

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

        dist = ExponentialDistribution(name, mu=1.0)
        assert dist["mu"] == 1.0
        assert dist.fixed["mu"] is True

        dist = ExponentialDistribution(name, mu=Uniform(0.0, 1.0, "mu"))

        value = -1.0
        hyper = {"mu": 0.5}
        assert isinstance(dist["mu"], Uniform)
        assert dist.fixed["mu"] is False
        assert dist.log_pdf(value, hyper) == -np.inf
        assert np.exp(dist.log_pdf(value, hyper)) == dist.pdf(value, hyper)

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
            dist.log_pdf(value, hyper)

    def test_deltafunction(self):
        """
        Test the DeltaFunctionDistribution class.
        """

        name = "test"

        dist = DeltaFunctionDistribution(name, peak=1.0)
        assert dist["peak"] == 1.0
        assert dist.fixed["peak"] is True
        assert np.all(dist.sample(size=10) == 1.0)

        dist = DeltaFunctionDistribution(name, peak=Uniform(0.0, 1.0, "peak"))

        value = 0.1
        hyper = {"peak": 0.5}
        assert isinstance(dist["peak"], Uniform)
        assert dist.fixed["peak"] is False
        assert dist.log_pdf(value, hyper) == -np.inf
        assert np.exp(dist.log_pdf(value, hyper)) == dist.pdf(value, hyper)

        value = 0.5
        assert dist.log_pdf(value, hyper) == 0.0
        assert np.exp(dist.log_pdf(value, hyper)) == dist.pdf(value, hyper)

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
            dist.log_pdf(value, hyper)

    def test_powerlaw(self):
        """
        Test the PowerLawDistribution class.
        """

        name = "test"

        dist = PowerLawDistribution(name, alpha=1.0, minimum=0.1, maximum=10.0)
        assert dist["alpha"] == 1.0
        assert dist.fixed["alpha"] is True
        assert dist["minimum"] == 0.1
        assert dist.fixed["minimum"] is True
        assert dist["maximum"] == 10.0
        assert dist.fixed["maximum"] is True

        # test out of bounds
        with pytest.raises(ValueError):
            PowerLawDistribution(name, alpha=1.0, minimum=-1.0, maximum=10.0)

        with pytest.raises(ValueError):
            PowerLawDistribution(name, alpha=1.0, minimum=-np.inf, maximum=10.0)

        with pytest.raises(ValueError):
            PowerLawDistribution(name, alpha=1.0, minimum=1.0, maximum=0.5)

        with pytest.raises(ValueError):
            PowerLawDistribution(name, alpha=1.0, minimum=1.0, maximum=-np.inf)

        minimum = 0.001
        maximum = 10.0
        dist = PowerLawDistribution(
            name, alpha=Uniform(0.0, 1.0, "alpha"), minimum=minimum, maximum=maximum
        )

        value = -1.0
        hyper = {"alpha": 0.5}
        assert isinstance(dist["alpha"], Uniform)
        assert dist.fixed["alpha"] is False
        assert dist.log_pdf(value, hyper) == -np.inf
        assert np.exp(dist.log_pdf(value, hyper)) == dist.pdf(value, hyper)

        # check drawn sample is within bounds
        assert minimum < dist.sample(hyper) < maximum

        # draw multiple samples
        N = 100
        samples = dist.sample(hyper, size=N)
        assert len(samples) == N
        assert np.all((samples > minimum) & (samples < maximum))

        value = 1.0
        hyper = {"kgsdg": 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(value, hyper)

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

        deltakwargs = {"peak": Uniform(0.0, 1.0, "peak")}
        dist = create_distribution(name, "DeltaFunction", deltakwargs)
        assert isinstance(dist, DeltaFunctionDistribution)
        assert dist["peak"] == deltakwargs["peak"]

        powerlawkwargs = {
            "alpha": Uniform(-1, 1, name="alpha"),
            "minimum": 0.00001,
            "maximum": 1000.0,
        }
        dist = create_distribution(name, "PowerLaw", powerlawkwargs)
        assert isinstance(dist, PowerLawDistribution)
        assert dist["alpha"] == powerlawkwargs["alpha"]
        assert dist["minimum"] == powerlawkwargs["minimum"]
        assert dist["maximum"] == powerlawkwargs["maximum"]


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

        # test invalid gridrange (lower bounds is less than upper bound)
        with pytest.raises(ValueError):
            MassQuadrupoleDistribution(gridrange=[100.0, 1.0])

        # test invalid gridrange (only one value passed)
        with pytest.raises(ValueError):
            MassQuadrupoleDistribution(gridrange=[100.0])

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
        pdist = ExponentialDistribution(
            "Q22", mu=Uniform(name="mu", minimum=0.0, maximum=1e32)
        )
        with pytest.raises(ValueError):
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2], distribution=pdist, sampler="akgkfsfd"
            )

        # unknown bandwidth type for KDE
        bw = "lkgadkgds"
        with pytest.raises(RuntimeError):
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2], distribution=pdist, bw=bw
            ).sample()

        # wrong type for q22grid
        q22grid = "lsgdgkavbc"
        with pytest.raises(TypeError):
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2], distribution=pdist, grid=q22grid
            )

        # test sampler
        mdist = MassQuadrupoleDistribution(
            data=[testdata1, testdata2], distribution=pdist
        )

        # make sure result is None before sampling
        assert mdist.result is None
        with pytest.raises(RuntimeError):
            _ = [_ for _ in mdist.posterior_predictive([0, 1])]

        res = mdist.sample(**{"Nlive": 100, "save": False, "label": "test1"})

        assert isinstance(res, Result)
        assert np.all((res.posterior["mu"] > 0.0) & (res.posterior["mu"] < 1e32))

        # check posterior predictive errors
        with pytest.raises(TypeError):
            _ = [_ for _ in mdist.posterior_predictive(0)]

        # check error when requesting too many samples
        with pytest.raises(ValueError):
            _ = [_ for _ in mdist.posterior_predictive([0, 1], nsamples=10000000000000)]

        points = [1e29, 1e30, 1e31]
        nsamples = 2
        for tfunc in [list, tuple, np.array]:
            assert np.array(
                [
                    values
                    for values in mdist.posterior_predictive(tfunc(points), nsamples)
                ]
            ).shape == (nsamples, len(points))

        del res
        del mdist

        # test using expectation values
        postsamples = ResultList([testdata1, testdata2])

        with pytest.raises(TypeError):
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2],
                distribution=pdist,
                integration_method="expectation",
                nsamples="blah",
            )

        with pytest.raises(ValueError):
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2],
                distribution=pdist,
                integration_method="expectation",
                nsamples=0,
            )

        mdist = MassQuadrupoleDistribution(
            data=[testdata1, testdata2],
            distribution=pdist,
            integration_method="expectation",
        )

        assert len(mdist._posterior_samples[0]) == len(postsamples[0].posterior)
        assert len(mdist._posterior_samples[1]) == len(postsamples[1].posterior)

        del mdist

        # use all samples (but test passing nsamples a larger number)
        mdist = MassQuadrupoleDistribution(
            data=[testdata1, testdata2],
            distribution=pdist,
            integration_method="expectation",
            nsamples=(
                1
                + int(
                    np.max(
                        [len(postsamples[0].posterior), len(postsamples[1].posterior)]
                    )
                )
            ),
        )

        assert len(mdist._posterior_samples[0]) == len(postsamples[0].posterior)
        assert len(mdist._posterior_samples[1]) == len(postsamples[1].posterior)

        del mdist

        # use 500 samples
        nsamples = 500
        mdist = MassQuadrupoleDistribution(
            data=[testdata1, testdata2],
            distribution=pdist,
            integration_method="expectation",
            nsamples=nsamples,
        )

        assert len(mdist._posterior_samples[0]) == nsamples
        assert len(mdist._posterior_samples[1]) == nsamples

        # try sampler
        res = mdist.sample(**{"Nlive": 100, "save": False, "label": "test2"})

        assert isinstance(res, Result)
        assert np.all((res.posterior["mu"] > 0.0) & (res.posterior["mu"] < 1e32))

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

        # make sure result is None before sampling
        assert mdist.result is None

        res = mdist.sample()
        assert isinstance(res, Grid)

        del res
        del mdist

        # test using ellipticity
        pdist = ExponentialDistribution("Q22", mu=Uniform(0.0, 1e-5, "mu"))
        with pytest.raises(ValueError):
            # pdist does not contain ELL
            MassQuadrupoleDistribution(
                data=[testdata1, testdata2], distribution=pdist, use_ellipticity=True
            )

        pdist = ExponentialDistribution("ELL", mu=Uniform(0.0, 1e-5, "mu"))
        mdist = MassQuadrupoleDistribution(
            data=[testdata1, testdata2], distribution=pdist, use_ellipticity=True
        )
        res = mdist.sample(**{"Nlive": 100, "save": False, "label": "test3"})

        assert isinstance(res, Result)
        assert "mu" in res.posterior.columns
        assert np.all((res.posterior["mu"] > 0.0) & (res.posterior["mu"] < 1e-5))

        del res
        del mdist
