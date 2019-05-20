"""
Classes for hierarchical parameter inference.
"""

import numpy as np
import bilby
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d


# allowed distributions and their required hyperparameters
DISTRIBUTION_REQUIREMENTS = {'exponential': ['mu'],
                             'gaussian': ['mu', 'sigma']}


class BaseDistribution(object):
    """
    The base class the distribution, as defined by a set of hyperparameters,
    that you want to fit.

    Parameters
    ----------
    name: str
        The parameter for which this distribution is the prior.
    disttype: str
        The type of distribution, e.g., 'exponential', 'gaussian' 
    hyperparameters: dict
        A dictionary of hyperparameters for the distribution with the keys
        giving the parameter names, and values giving their (fixed) values.
    """

    def __init__(self, name, disttype, hyperparameters={}):
        self.name = name  # the parameter name
        self.disttype = disttype
        self.hyperparameters = hyperparameters

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
                    params.append('{0}{1:02d}'.format(key, i))
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

    def known(self, item):
        """
        Check if a given hyperparameter is known, i.e., it has a set value, or
        is unknown, i.e., it is not set, or it has a value of None. For
        hyperparameters that are defined as lists return a boolean list.

        Parameters
        ----------
        item: str
            The name of the hyperparameter to check

        Returns
        -------
        bool:
            A single boolean, or list of boolean values, stating whether the
            parameter is defined (known) or None (unknown and to be estimated).
        """
        
        value = self[item]
        if value is not None:
            if isinstance(value, (list, np.ndarray)):
                isknown = [True] * len(value)
                for i in range(len(value)):
                    if value[i] is None:
                        isknown[i] = False
                return isknown
            else:
                return True
        else:
            return False


class GaussianDistribution(BaseDistribution):
    """
    A distribution to define estimating the parameters of a (potentially
    multi-modal) Gaussian distribution.
    
    Parameters
    ----------
    name: str
        See :class:`~cwinpy.hierarchical.BaseDistribution`
    mus: array_like
        A list of values of the means of each mode of the Gaussian.
    sigmas: array_like
        A list of values of the standard deviations of each mode of the
        Gaussian.
    nmodes: int (optional)
        The number of modes of the Gaussian distribution. If this is not set
        then the number is taken from the maximum of the lengths of either
        `means` or `sigmas`. If this is longer than either `means` or `sigmas`
        then those additional values are filled in a None, i.e., they are
        unknown.
    """

    def __init__(self, name, mus, sigmas, nmodes=1):
        gaussianparameters = {'mu': [], 'sigma': []}

        if nmodes < 1:
            raise ValueError('Gaussian must have at least one mode')

        if isinstance(mus, (int, float)):
            mus = [mus]
        
        if isinstance(sigmas, (int, float)):
            sigmas = [sigmas]

        if (not isinstance(mus, (list, np.ndarray)) or not
                isinstance(sigmas, (list, np.ndarray))):
            raise TypeError('Means and standard deviations must be lists')

        self.nmodes = np.max([nmodes, len(mus), len(sigmas)])

        for i in range(self.nmodes):
            if len(mus) < (i + 1):
                gaussianparameters['mu'].append(None)
            else:
                gaussianparameters['mu'].append(mus[i])

            if len(sigmas) < (i + 1):
                gaussianparameters['sigma'].append(None)
            else:
                gaussianparameters['sigma'].append(sigmas[i])

        # initialise
        super().__init__(name, 'gaussian', gaussianparameters)


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
        if mu is not None:
            if not isinstance(mu, (int, float)):
                raise TypeError('Mean must be a number')
            
            if mu <= 0.:
                raise ValueError('Mean must be a positive number')

        # initialise
        super().__init__(name, 'exponential', dict(mu=mu))


def create_distribution(distribution, name=None, distkwargs={}):
    """
    Create a distribution.

    Parameters
    ----------
    distribution: :class:`cwinpy.hierarchical.BaseDistribution`, str
        A predefined distribution, or string giving a valid distribution name.
        This is the distribution who's hyperparameters that are going to be
        inferred. If using a string, the distribution keyword arguments must be
        passed using ``distkwargs``.
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
                 distribution=None, distkwargs=None, bw='scott'):
        self._posterior_samples = []
        self._posterior_kdes = []
        self._likelihood_kdes_interp = []

        # set the values of q22 at which to calculate the KDE interpolator
        self.set_q22range(q22range, q22bins)

        # set the data
        self.add_data(data, bw=bw)

        # set the distribution
        self.set_distribution(distribution, distkwargs)

    def set_q22range(self, q22range, q22bins=100):
        """
        Set the values of :math:`Q_{22}`, either directly, or as a set of
        points linear in log-space defined by a lower and upper bounds and
        number of bins, at which to evaluate the posterior samples via their
        KDE to make an interpolator.

        Parameters
        ----------
        q22range: array_like
            If this array contains two values it is assumed that these are the
            lower and upper bounds of a range, and the Q22
        """

        if q22range is None:
            self._q22_interp_values = None
            return 

        if len(q22range) == 2:
            if q22range[1] < q22range[0]:
                raise ValueError('Q22 range is badly defined')
            self._q22_interp_values = np.logspace(q22range[0], q22range[1],
                                                  q22bins)
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
        bw: str, 'scott'
            The Gaussian KDE bandwidth calculation method as required by
            :meth:`scipy.stats.gaussian_kde`.
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

        if distribution is None:
            return

        if isinstance(distribution, BaseDistribution):
            self._distribution = distribution
        elif isinstance(distribution, str):
            if distribution.lower() in DISTRIBUTION_REQUIREMENTS.values():
                if 
