"""
Test script for the hierarchical classes.
"""

import pytest
import numpy as np
from cwinpy.hierarchical import (BaseDistribution,
                                 BoundedGaussianDistribution,
                                 ExponentialDistribution,
                                 create_distribution)
from bilby.core.prior import Uniform


class TestDistributionObjects(object):
    """
    Tests for the distribution objects.
    """

    def test_base_distribution(self):
        """
        Test the BaseDistribution object.
        """

        name = 'test'

        # test failure for unknown distribution
        with pytest.raises(ValueError):
            BaseDistribution(name, 'kjsgdkdgkjgsda')
        
        # test failure for inappropriate bounds
        with pytest.raises(ValueError):
            BaseDistribution(name, 'gaussian', low=0., high=-1.)

        # test failure for unknown hyperparameter name
        with pytest.raises(KeyError):
            hyper = {'mu': [1], 'dkgwkufd': [2]}
            BaseDistribution(name, 'gaussian', hyperparameters=hyper)

        # test failure with invalid hyperparameter type
        with pytest.raises(TypeError):
            BaseDistribution(name, 'gaussian', hyperparameters='blah')
        
        with pytest.raises(TypeError):
            hyper = 'blah'
            BaseDistribution(name, 'exponential', hyperparameters=hyper)

        # test default log_pdf is NaN
        hyper = {'mu': 2.}
        dist = BaseDistribution(name, 'exponential',
                                hyperparameters=hyper)

        assert dist['mu'] == hyper['mu']
        assert np.isnan(dist.log_pdf({}, 0))

        # test failure when getting unknown item
        with pytest.raises(KeyError):
            val = dist['kgksda']

        del dist

        # test setter failure
        dist = BaseDistribution(name, 'exponential')

        with pytest.raises(KeyError):
            dist['madbks'] = Uniform(0., 1., 'mu')

        # test setter
        dist['mu'] = Uniform(0., 1., 'mu')
        assert isinstance(dist['mu'], Uniform)

    def test_bounded_gaussian(self):
        """
        Test the BoundedGaussianDistribution class.
        """

        name = 'test'

        # test failures
        with pytest.raises(TypeError):
            BoundedGaussianDistribution(name, mus='blah')

        with pytest.raises(TypeError):
            BoundedGaussianDistribution(name, mus=[1], sigmas='blah')
        
        with pytest.raises(ValueError):
            BoundedGaussianDistribution(name, mus=[1.], sigmas=[1., 2.])
        
        with pytest.raises(ValueError):
            BoundedGaussianDistribution(name)
        
        dist = BoundedGaussianDistribution(name, mus=[1., 2.], sigmas=[1., 2.])

        assert dist.nmodes == 2

        del dist
        # test log pdf
        dist = BoundedGaussianDistribution(name, mus=[Uniform(0., 1., 'mu0'), 2.],
                                           sigmas=[Uniform(0., 1., 'sigma0'), 2.])

        value = 1.
        hyper = {'mu8': 0.5, 'sigma0': 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(hyper, value)
        
        hyper = {'mu0': 0.5, 'sigma8': 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(hyper, value)
        
        hyper = {'mu0': 0.5, 'sigma0': 0.5}
        assert np.isfinite(dist.log_pdf(hyper, value))

        # check negative values give -inf by default
        value = -1.
        assert dist.log_pdf(hyper, value) == -np.inf

    def test_exponential(self):
        """
        Test the ExponentialDistribution class.
        """

        name = 'test'

        with pytest.raises(TypeError):
            ExponentialDistribution(name, mu=1.)

        dist = ExponentialDistribution(name, mu=Uniform(0., 1., 'mu'))

        value = -1.
        hyper = {'mu': 0.5}
        assert dist.log_pdf(hyper, value) == -np.inf

        value = 1.
        hyper = {'kgsdg': 0.5}
        with pytest.raises(KeyError):
            dist.log_pdf(hyper, value)

    def test_create_distribution(self):
        """
        Test the create_distribution() function.
        """

        name = 'test'
        with pytest.raises(ValueError):
            create_distribution(name, 'kjbskdvakvkd')
        
        with pytest.raises(TypeError):
            create_distribution(name, 2.3)

        gausskwargs = {'mus': [1., 2.], 'sigmas': [1., 2.]}
        dist = create_distribution(name, 'Gaussian', gausskwargs)

        assert isinstance(dist, BoundedGaussianDistribution)
        assert (dist['mu0'] == gausskwargs['mus'][0] and
                dist['mu1'] == gausskwargs['mus'][1])
        assert (dist['sigma0'] == gausskwargs['sigmas'][0] and
                dist['sigma1'] == gausskwargs['sigmas'][1])
        del dist
        
        expkwargs = {'mu': Uniform(0., 1., 'mu')}
        dist = create_distribution(name, 'Exponential', expkwargs)
        assert isinstance(dist, ExponentialDistribution)
        assert dist['mu'] == expkwargs['mu']

        newdist = create_distribution(name, dist)
        assert isinstance(newdist, ExponentialDistribution)
        assert newdist['mu'] == dist['mu']
