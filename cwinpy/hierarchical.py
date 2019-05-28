"""
Classes for hierarchical parameter inference.
"""

import numpy as np
from scipy.stats import (gaussian_kde, truncnorm, expon)
from scipy.interpolate import interp1d
from collections import OrderedDict
from itertools import compress
import bilby


# allowed distributions and their required hyperparameters
DISTRIBUTION_REQUIREMENTS = {'exponential': ['mu'],
                             'gaussian': ['mu', 'sigma']}


class BaseDistribution(object):
    """
    The base class for the distribution, as defined by a set of
    hyperparameters, that you want to fit.

    Parameters
    ----------
    name: str
        The parameter for which this distribution is the prior.
    disttype: str
        The type of distribution, e.g., 'exponential', 'gaussian' 
    hyperparameters: dict
        A dictionary of hyperparameters for the distribution with the keys
        giving the parameter names, and values giving their fixed value, or
        a :class:`bilby.core.prior.Prior` for values that are to be inferred.
    low: float
        The lower bound of the distribution
    high: float
        The upper bound of the distribution
    """

    def __init__(self, name, disttype, hyperparameters={}, low=-np.inf,
                 high=np.inf):
        self.name = name  # the parameter name
        self.disttype = disttype
        self.hyperparameters = hyperparameters
        self.low = low
        self.high = high

    @property
    def disttype(self):
        return self._disttype

    @disttype.setter
    def disttype(self, disttype):
        if disttype.lower() not in DISTRIBUTION_REQUIREMENTS.keys():
            raise ValueError('Distribution name "{}" is not '
                             'known'.format(disttype))
        else:
            self._disttype = disttype.lower()

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        if isinstance(hyperparameters, dict):
            # check is contains the required parameter names
            for key in hyperparameters.keys():
                if key.lower() not in DISTRIBUTION_REQUIREMENTS[self.disttype]:
                    raise KeyError('Unknown parameter "{}" for distribution '
                                   '"{}"'.format(key, self.disttype))
            self._hyperparameters = {key.lower(): value
                                     for key, value in hyperparameters.items()}
        else:
            raise TypeError("hyperparameters must be a dictionary")

        # set fixed values
        self.fixed = self._hyperparameters

    @property
    def parameters(self):
        return list(self.hyperparameters.keys())

    @property
    def values(self):
        return list(self.hyperparameters.values())

    @property
    def unpacked_parameters(self):
        params = []
        for key, value in self.hyperparameters.items():
            if isinstance(value, (list, np.ndarray)):
                for i in range(len(value)):
                    params.append('{0}_{{{1:d}}}'.format(key, i))
            else:
                params.append(key)
        return params

    @property
    def unpacked_values(self):
        values = []
        for key, value in self.hyperparameters.items():
            if isinstance(value, (list, np.ndarray)):
                for i in range(len(value)):
                    values.append(value[i])
            else:
                values.append(value)
        return values

    def __getitem__(self, item):
        if item.lower() in self.parameters:
            return self.hyperparameters[item.lower()]
        elif item.lower() in self.unpacked_parameters:
            return self.unpacked_values[
                self.unpacked_parameters.index(item.lower())]
        elif item.lower() in DISTRIBUTION_REQUIREMENTS[self.disttype]:
            return None
        else:
            raise KeyError('"{}" is not a parameter in this '
                           'distribution'.format(item))

    def __setitem__(self, item, value):
        if item.lower() not in self.hyperparameters.keys():
            if item.lower() in DISTRIBUTION_REQUIREMENTS[self.disttype]:
                self._hyperparameters[item.lower()] = value
            else:
                raise KeyError('"{}" is not a parameter in this '
                               'distribution'.format(item))
        else:
            self._hyperparameters[item.lower()] = value

    @property
    def fixed(self):
        """
        Return a dictionary keyed to parameter names and with boolean values
        indicating whether the parameter is fixed (True), or to be inferred
        (False).
        """

        return self._fixed

    @fixed.setter
    def fixed(self, hyperparameters):
        self._fixed = OrderedDict()

        for param, value in hyperparameters.items():
            if isinstance(value, bilby.core.prior.Prior):
                self._fixed[param] = False
            elif isinstance(value, (list, np.ndarray)):
                self._fixed[param] = []
                for i in range(len(value)):
                    if isinstance(value[i], bilby.core.prior.Prior):
                        self._fixed[param].append(False)
                    elif isinstance(value[i], (int, float)):
                        self._fixed[param.append(True)]
                    else:
                        raise TypeError("Hyperparameter type is not valid")
            elif isinstance(value, (int, float)):
                self._fixed[param] = True
            else:
                raise TypeError("Hyperparameter type is not valid")

    @property
    def unpacked_fixed(self):
        """
        Return a flattened version of ``fixed``, with multivalued parameters
        indexed.
        """

        fixed = OrderedDict()

        for param, value in zip(self.unpacked_parameters, self.unpacked_values):
            if isinstance(value, bilby.core.prior.Prior):
                fixed[param] = False
            elif isinstance(value, (int, float)):
                fixed[param] = True
            else:
                raise TypeError("Hyperparameter type is not valid")

        return fixed

    @property
    def unknown_parameters(self):
        """
        A list of the parameters that are to be inferred.
        """

        return list(compress(self.unpacked_parameters,
                             ~np.array(self.unpacked_fixed)))

    @property
    def unknown_priors(self):
        """
        A list of the :class:`~bilby.core.prior.Prior`s for the parameters
        that are to be inferred.
        """

        return list(compress(self.unpacked_values,
                             ~np.array(self.unpacked_fixed)))
    
    def log_pdf(self, hyperparameters, value):
        """
        The natural logarithm of the distribution's probability density
        function at the given value.

        Parameters
        ----------
        hyperparameters: dict
            A dictionary of the hyperparameter values that define the current
            state of the distribution.
        value: float
            The value at which to evaluate the probability.

        Returns
        -------
        lnpdf:
            The natural logarithm of the probability.
        """

        return np.nan


class BoundedGaussianDistribution(BaseDistribution):
    """
    A distribution to define estimating the parameters of a (potentially
    multi-modal) bounded Gaussian distribution.
    
    Parameters
    ----------
    name: str
        See :class:`~cwinpy.hierarchical.BaseDistribution`
    mus: array_like
        A list of values of the means of each mode of the Gaussian.
    sigmas: array_like
        A list of values of the standard deviations of each mode of the
        Gaussian.
    low: float
        The lower bound of the distribution (defaults to 0, i.e., only positive
        values are allowed)
    high: float
        The upper bound of the distribution (default to infinity)
    """

    def __init__(self, name, mus=[], sigmas=[], low=0., high=np.nan):
        gaussianparameters = {'mu': [], 'sigma': []}

        if isinstance(mus, (int, float, bilby.core.prior.Prior)):
            mus = [mus]
        elif not isinstance(mus, (list, np.ndarray)):
            raise TypeError("Unknown type fot 'mus'")
        
        if isinstance(sigmas, (int, float, bilby.core.prior.Prior)):
            sigmas = [sigmas]
        elif not isinstance(sigmas, (list, np.ndarray)):
            raise TypeError("Unknown type fot 'sigmas'")

        # set the number of modes
        self.nmodes = len(mus)

        if len(mus) != len(sigmas):
            raise ValueError("'mus' and 'sigmas' must be the same length")

        if self.nmodes < 1:
            raise ValueError('Gaussian must have at least one mode')

        for i in range(self.nmodes):
            gaussianparameters['mu'].append(mus[i])
            gaussianparameters['sigma'].append(sigmas[i])

        # initialise
        super().__init__(name, 'gaussian', hyperparameters=gaussianparameters,
                         low=low, high=high)

    def log_pdf(self, hyperparameters, value):
        """
        The natural logarithm of the pdf of a 1d (potentially multi-modal)
        Gaussian probability distribution.

        Parameters
        ----------
        hyperparameters: dict
            A dictionary containing the current values of the hyperparameters
            that need to be inferred.
        value: float
            The value at which the probability is to be evaluated.

        Returns
        -------
        logpdf:
            The natural logarithm of the probability at the given value.
        """

        if value < self.low or value > self.high:
            return -np.inf

        mus = self['mu']
        sigmas = self['sigma']

        # get current mus and sigmas from values
        for i in range(self.nmodes):
            if not self.fixed['mu'][i]:
                param = 'mu_{}'.format(i)
                try:
                    mus[i] = hyperparameters[param]
                except KeyError:
                    raise KeyError("Cannot calculate log probability when "
                                   "value '{}' is not given".format(param))
            
            if not self.fixed['sigma'][i]:
                param = 'sigma_{}'.format(i)
                try:
                    sigmas[i] = hyperparameters[param]
                except KeyError:
                    raise KeyError("Cannot calculate log probability when "
                                   "value '{}' is not given".format(param))

        # get log pdf
        logpdf = -np.inf
        for mu, sigma in zip(mus, sigmas):
            logpdf = np.logaddexp(logpdf, truncnorm.logpdf(value, self.low,
                                                           self.high, loc=mu,
                                                           scale=sigma))

        # properly normalise
        logpdf -= np.log(self.nmodes)

        return logpdf


class ExponentialDistribution(BaseDistribution):
    """
    A distribution to define estimating the parameters of an
    exponential distribution.

    Parameters
    ----------
    name: str
        See :class:`~cwinpy.hierarchical.BaseDistribution`
    mu: array_like
        The mean of the exponential distribution.
    """

    def __init__(self, name, mu):
        if not isinstance(mu, bilby.core.prior.Prior):
            raise TypeError('Mean must be a Prior distribution')

        # initialise
        super().__init__(name, 'exponential', hyperparameters=dict(mu=mu),
                         low=0., high=np.inf)

    def log_pdf(self, hyperparameters, value):
        """
        The natural logarithm of the pdf of an exponential distribution.

        Parameters
        ----------
        hyperparameters: dict
            A dictionary containing the current values of the hyperparameters
            that need to be inferred.
        value: float
            The value at which the probability is to be evaluated.

        Returns
        -------
        logpdf:
            The natural logarithm of the probability at the given value.
        """

        if value < self.low or value > self.high:
            return -np.inf

        try:
            mu = hyperparameters['mu']
        except KeyError:
            raise KeyError("Cannot evaluate the probability when mu is not "
                           "given")

        # get log pdf
        logpdf = expon.logpdf(value, scale=mu)

        # properly normalise
        logpdf -= np.log(self.nmodes)

        return logpdf


def create_distribution(name, distribution, distkwargs={}):
    """
    Function to create a distribution.

    Parameters
    ----------
    name: str
        The name of the parameter which the distribution represents.
    distribution: str, :class:`cwinpy.hierarchical.BaseDistribution`
        A string giving a valid distribution name. This is the distribution for
        which the hyperparameters are going to be inferred. If using a string,
        the distribution keyword arguments must be passed using ``distkwargs``.
    distkwargs: dict
        A dictionary of keyword arguments for the distribution that is being
        inferred.

    Returns
    -------
    distribution
        The distribution class.
    """

    if isinstance(distribution, BaseDistribution):
        return distribution
    elif isinstance(distribution, str):
        if distribution.lower() not in DISTRIBUTION_REQUIREMENTS.keys():
            raise ValueError('Unknown distribution type '
                             '"{}"'.format(distribution))

        if distribution.lower() == 'gaussian':
            return GaussianDistribution(name, **distkwargs)
        elif distribution.lower() == 'exponential':
            return ExponentialDistribution(name, **distkwargs)
    else:
        raise TypeError("Unknown distribution")


class MassQuadrupoleDistribution(object):
    """
    A class infer the hyperparameters of the :math:`l=m=2` mass quadrupole
    distribution for a given selection of known pulsars (see, for example,
    [1]_).

    The class currently can attempt to fit the hyperparameters for the
    following distributions:

    * a :math:`n`-mode Gaussian distribution defined by either fixed or unknown
      means and standard deviations
    * an exponential distribution defined by an unknown mean.

    All distributions do not allow the quadrupole value to become negative.

    Parameters
    ----------
    data: :class:`bilby.core.result.ResultList`
        A :class:`bilby.core.result.ResultList` of outputs from running source
        parameter estimation using bilby for a set of individual CW sources.
        These can be from MCMC or nested sampler runs, but only the latter can
        be used if requiring a properly normalised evidence value.
    q22range: array_like
        A list of values at which the :math:`Q_{22}` parameter posteriors
        should be interpolated, or a lower and upper bound in the range of
        values, which will be split into ``q22bins`` points spaced linearly in
        log-space.
    q22bins: int
        The number of bins in :math:`Q_{22}` at which the posterior will be
        interpolated.
    distribution: :class:`cwinpy.hierarchical.BaseDistribution`, str
        A predefined distribution, or string giving a valid distribution name.
        This is the distribution who's hyperparameters that are going to be
        inferred. If using a string, the distribution keyword arguments must be
        passed using ``distkwargs``.
    distkwargs: dict
        A dictionary of keyword arguments for the distribution that is being
        inferred.
    bw: str, scalar, callable
        See the ``bw_method`` argument for :class:`scipy.stats.gaussian_kde`.
    sampler: str
        The name of the stochastic sampler method used by ``bilby`` for
        sampling the posterior. This defaults to use 'dynesty'.
    sampler_kwargs: dict
        A dictionary of arguments required by the given sampler.
    grid: dict
        A dictionary of values that define a grid in the parameter and
        hyperparameter space that can be used by a
        :class:`bilby.core.grid.Grid`. If given sampling will be performed on
        the grid, rather than using the stochastic sampler.

    .. todo::

    Distributions the could be added include:

    * a power law distribution with an unknown spectral index, or a (single)
      broken power law with two unknown indices and a known or unknown break
      point;
    * a Student's t-distributions with unknown mean and number of degrees of
      freedom.

    References
    ----------

    .. [1] M. Pitkin, C. Messenger & X. Fan, Phys. Rev. D, 98, 063001, 2018
       (`arXiv:1807.06726 <https://arxiv.org/abs/1807.06726>`_)
    """

    def __init__(self, data=None, q22range=None, q22bins=100,
                 distribution=None, distkwargs=None, bw='scott',
                 sampler='dynesty', sampler_kwargs={}, grid=None):
        self._posterior_samples = []
        self._posterior_kdes = []
        self._likelihood_kdes_interp = []

        # set the values of q22 at which to calculate the KDE interpolator
        self.set_q22range(q22range, q22bins)

        # set the data
        self.add_data(data, bw=bw)

        # set the distribution
        self.set_distribution(distribution, distkwargs)

        # set the sampler
        if grid is None:
            self.set_sampler(sampler, sampler_kwargs)
        else:
            self.set_grid(grid)

    def set_q22range(self, q22range, q22bins=100, prependzero=True):
        """
        Set the values of :math:`Q_{22}`, either directly, or as a set of
        points linear in log-space defined by a lower and upper bounds and
        number of bins, at which to evaluate the posterior samples via their
        KDE to make an interpolator.

        Parameters
        ----------
        q22range: array_like
            If this array contains two values it is assumed that these are the
            lower and upper bounds of a range, and the ``q22bins`` parameter
            sets the number of bins in log-space that the range will be split
            into. Otherwise, if more than two values are given it is assumed
            these are the values for :math:`Q_{22}`.
        q22bins: int
            The number of bins the range is split into.
        prependzero: bool
            If setting an upper and lower range, this will prepend zero at the
            start of the range. Default is True.
        """

        if q22range is None:
            self._q22_interp_values = None
            return 

        if len(q22range) == 2:
            if q22range[1] < q22range[0]:
                raise ValueError('Q22 range is badly defined')
            self._q22_interp_values = np.logspace(q22range[0], q22range[1],
                                                  q22bins)

            if prependzero:
                self._q22_interp_values = np.insert(self._q22_interp_values,
                                                    0, 0)
        elif len(q22range) > 2:
            self._q22_interp_values = q22range
        else:
            raise ValueError('Q22 range is badly defined')

    @property
    def interpolated_log_kdes(self):
        """
        Return the list of interpolation functions for the natural logarithm of
        the :math:`Q_{22}` likelihood functions after a Gaussian KDE has been
        applied.
        """

        return self._likelihood_kdes_interp

    def add_data(self, data, bw='scott'):
        """
        Set the data, i.e., the individual source posterior distributions, on
        which the hierarchical analysis will be performed.

        The posterior samples must include the ``Q22`` :math:`l=m=2` parameter
        for this inference to be performed. The samples will be converted to
        a KDE (reflected about zero to avoid edge effects, and re-normalised),
        using :meth:`scipy.stats.gaussian_kde`, which ultimately can be used as
        the data for hierarchical inference. For speed, interpolation functions
        of the natural logarithm of the KDEs, are stored. If the posterior
        samples come with a Bayesian evidence value, and the prior is present,
        then these are used to convert the posterior distribution into a
        likelihood, which is what is then stored in the interpolation function.

        Parameters
        ----------
        data: :class:`bilby.core.result.ResultList`
            A list, or single, results from bilby containing posterior samples
            for a set of sources, or individual source.
        bw: str, scale, callable
            The Gaussian KDE bandwidth calculation method as required by
            :class:`scipy.stats.gaussian_kde`. The default is the 'scott'
            method.
        """

        # check the data is a ResultList
        if not isinstance(data, bilby.core.result.ResultList):
            if isinstance(data, (bilby.core.Result, str)):
                # convert to a ResultList
                data = bilby.core.result.ResultList([data])
            elif isinstance(data, list):
                data = bilby.core.result.ResultList(data)
            elif data is None:
                return
            else:
                raise TypeError('Data is not a known type')

        # create KDEs
        for result in data:
            self._bw = bw

            # check all posteriors contain Q22
            if 'Q22' not in result.search_parameter_keys:
                raise RuntimeError("Results do not contain Q22")

            try:
                samples = result.samples['Q22']

                # get reflected samples
                samps = np.concatenate((samples, -samples))[:, np.newaxis]

                # calculate KDE
                kde = gaussian_kde(samps, bw_method=bw)
                self._posterior_kdes.append(kde)

                # use log pdf for the kde
                interpvals = (kde.logpdf(self._q22_interp_values)
                    + np.log(2.))  # multiply by 2 so pdf normalises to 1
            except:
                raise RuntimeError("Problem creating KDE")

            # convert posterior to likelihood (if possible)
            if np.isfinite(result.log_evidence):
                # multiply by evidence
                interpvals += result.log_evidence

            # divide by Q22 prior
            if 'Q22' not in result.priors:
                raise KeyError('Prior contains no Q22 value')
            prior = result.priors['Q22']
            interpvals -= prior.ln_prob(self._q22_interp_values)

            # create and add interpolator
            self._likelihood_kdes_interp.append(
                interp1d(self._q22_interp_values), interpvals)

            # append samples
            self._posteriors_samples.append(samples)

    def set_distribution(self, distribution, distkwargs={}):
        """
        Set the distribution who's hyperparameters are going to be inferred.

        Parameters
        ----------
        distribution: :class:`cwinpy.hierarchical.BaseDistribution`, str
            A predefined distribution, or string giving a valid distribution
            name. If using a string, the distribution keyword arguments must be
            passed using ``distkwargs``.
        distkwargs: dict
            A dictionary of keyword arguments for the distribution that is being
            inferred. 
        """

        self._distribution = None
        self._prior = None
        self._likelihood = None

        if distribution is None:
            return

        if isinstance(distribution, BaseDistribution):
            if distribution.name.upper() is not 'Q22':
                raise ValueError("Distribution name must be 'Q22'")
            elif distribution.disttype not in DISTRIBUTION_REQUIREMENTS.keys():
                raise ValueError("Distribution type '{}' is not "
                                 "known".format(distribution.disttype))
            else:
                self._distribution = distribution
        elif isinstance(distribution, str):
            self._distribution = create_distribution('Q22',
                                                     distribution.lower(),
                                                     **distkwargs)

        # set the priors from the distribution
        self._set_priors()

        # set the likelihood function
        self._set_likelihood()

    def _set_priors(self):
        """
        Set the priors based on those supplied via the distribution class.
        """

        # get the priors from the distribution
        if len(self._distribution.unknown_parameters) < 1:
            raise ValueError("Distribution has no parameters to infer")
        
        # add priors as PriorDict
        self._prior = {param: prior
                       for param, prior in zip(self._distribution.unknown_parameters,
                                               self._distribution.unknown_priors)}

    def _set_likelihood(self):
        """
        Set the likelihood.
        """

        self._likelihood = MassQuadrupoleDistributionLikelihood(
            self._distribution, self._likelihood_kdes_interp)
    
    def set_sampler(self, sampler='dynesty', sampler_kwargs={}):
        """
        Set the stochastic sampling method for ``bilby`` to use when sampling
        the parameter and hyperparameter posteriors.

        Parameters
        ----------
        sampler: str
            The name of the stochastic sampler method used by ``bilby`` for
            sampling the posterior. This defaults to use 'dynesty'.
        sampler_kwargs: dict
            A dictionary of arguments required by the given sampler.
        """

        self._sampler = sampler
        if self._sampler not in bilby.core.sampler.IMPLEMENTED_SAMPLERS:
            raise ValueError('Sampler "{}" is not implemented in '
                             'bilby'.format(self._sampler))
        self._sampler_kwargs = sampler_kwargs
        self._use_grid = False  # set to not use the Grid sampling

    def set_grid(self, grid):
        """
        Set a grid on which to evaluate the parameter and hyperparameter
        posterior, as used by :class:`bilby.core.grid.Grid`.

        Parameters
        ----------
        grid: dict
            A dictionary of values that define a grid in the parameter and
            hyperparameter space that can be used by a
            :class:`bilby.core.grid.Grid` class.
        """

        self._grid = grid
        self._use_grid = True

    @property
    def result(self):
        """
        Return the ``bilby`` object containing the results. If evaluating the
        posterior over a grid this is a :class:`bilby.core.grid.Grid` object.
        If sampling using a stochastic sampler, this is a
        :class:`bilby.core.result.Result` object.
        """

        if self._use_grid:
            return self._grid_result
        else:
            return self._result

    def sample(self, **run_kwargs):
        """
        Sample the posterior distribution using ``bilby``. This can take
        keyword argument required by the bilby ``run sampler()` <https://lscsoft.docs.ligo.org/bilby/samplers.html#bilby.run_sampler>`_ method.
        """

        if self._use_grid:
            # set the grid
            self._grid_result = bilby.core.grid.Grid(self._likelihood,
                                                     self._prior,
                                                     grid_size=self._grid)
        else:
            self._result = bilby.run_sampler(likelihood=self._likelihood,
                                             priors=self._prior,
                                             sampler=self._sampler,
                                             **self._sampler_kwargs,
                                             **run_kwargs)

        return self.result


class MassQuadrupoleDistributionLikelihood(bilby.core.likelihood.Likelihood):
    """
    The likelihood function for the inferring the hyperparameters of the
    mass quadrupole, :math:`Q_{22}`, distribution.
    
    Parameters
    ----------
    distribution: :class:`cwinpy.hierarchical.BaseDistribution`
        The probability distribution for which the hyperparameters are going
        to be inferred.
    likelihoods: list
        A list of interpolation functions each of which gives the likelihood
        function for a single source.

    """

    def __init__(self, distribution, likelihoods):
        if not isinstance(distribution, BaseDistribution):
            raise TypeError("Distribution is not the correct type")
        
        # check that the distribution contains parameters to be inferred
        if len(distribution.unknown_parameters) < 1:
            raise ValueError("Distribution has no parameters to infer")

        inferred_parameters = {param: None
                               for param in distribution.unknown_parameters}
        inferred_parameters[distribution.name] = None
        self.distribution = distribution
        self.likelihoods = likelihoods

        super().__init__(parameters=inferred_parameters)

    def log_likelihood(self):
        """
        Evaluate the log likelihood.
        """

        value = self.parameters[self.distribution.name]

        # evaluate the hyperparameter distribution
        log_prior = self.distribution.log_pdf(self.parameters, value)

        # evaluate the log likelihood (sum of each separate log likelihood)
        log_like = 0.
        for intfunc in self.likelihoods:
            log_like += intfunc(value)

        return log_like + log_prior
