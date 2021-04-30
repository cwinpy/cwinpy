"""
Classes for hierarchical parameter inference.
"""

from itertools import compress

import bilby
import numpy as np
from lintegrate import logtrapz
from scipy.interpolate import splev, splrep
from scipy.stats import expon, gaussian_kde, truncnorm

from .utils import ellipticity_to_q22, q22_to_ellipticity

#: Allowed distributions and their required hyperparameters
DISTRIBUTION_REQUIREMENTS = {
    "exponential": ["mu"],
    "gaussian": ["mu", "sigma", "weight"],
    "deltafunction": ["peak"],
    "powerlaw": ["alpha", "minimum", "maximum"],
    "histogram": ["weight"],
}


class BaseDistribution(object):
    """
    The base class for the distribution, as defined by a set of
    hyperparameters, that you want to fit.

    Parameters
    ----------
    name: str
        The parameter for which this distribution is the prior.
    disttype: str
        The type of distribution, e.g., 'exponential', 'gaussian'.
    hyperparameters: dict
        A dictionary of hyperparameters for the distribution with the keys
        giving the parameter names, and values giving their fixed value, or
        a :class:`bilby.core.prior.Prior` for values that are to be inferred.
    low: float
        The lower bound of the distribution
    high: float
        The upper bound of the distribution
    """

    def __init__(self, name, disttype, hyperparameters={}, low=-np.inf, high=np.inf):
        self.name = name  # the parameter name
        self.disttype = disttype
        self.hyperparameters = hyperparameters
        self.low = low
        self.high = high

        if self.low >= self.high:
            raise ValueError("Lower bound is higher than upper bound!")

    @property
    def disttype(self):
        return self._disttype

    @disttype.setter
    def disttype(self, disttype):
        if disttype.lower() not in DISTRIBUTION_REQUIREMENTS.keys():
            raise ValueError('Distribution name "{}" is not known'.format(disttype))
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
                    raise KeyError(
                        'Unknown parameter "{}" for distribution '
                        '"{}"'.format(key, self.disttype)
                    )
            self._hyperparameters = {
                key.lower(): value for key, value in hyperparameters.items()
            }
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
                    params.append("{0}{1:d}".format(key, i))
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
            return self.unpacked_values[self.unpacked_parameters.index(item.lower())]
        elif item.lower() in DISTRIBUTION_REQUIREMENTS[self.disttype]:
            return None
        else:
            raise KeyError('"{}" is not a parameter in this distribution'.format(item))

    def __setitem__(self, item, value):
        if item.lower() not in self.hyperparameters.keys():
            if item.lower() in DISTRIBUTION_REQUIREMENTS[self.disttype]:
                self._hyperparameters[item.lower()] = value
            else:
                raise KeyError(
                    '"{}" is not a parameter in this distribution'.format(item)
                )
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
        self._fixed = dict()

        for param, value in hyperparameters.items():
            if isinstance(value, (bilby.core.prior.Prior, bilby.core.prior.PriorDict)):
                self._fixed[param] = False
            elif isinstance(value, (list, np.ndarray)):
                self._fixed[param] = []
                for i in range(len(value)):
                    if isinstance(
                        value[i], (bilby.core.prior.Prior, bilby.core.prior.PriorDict)
                    ):
                        self._fixed[param].append(False)
                    elif isinstance(value[i], (int, float)):
                        self._fixed[param].append(True)
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

        fixed = dict()

        for param, value in zip(self.unpacked_parameters, self.unpacked_values):
            if isinstance(value, (bilby.core.prior.Prior, bilby.core.prior.PriorDict)):
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

        return list(
            compress(
                self.unpacked_parameters, ~np.array(list(self.unpacked_fixed.values()))
            )
        )

    @property
    def unknown_priors(self):
        """
        A list of the :class:`~bilby.core.prior.Prior` for the parameters
        that are to be inferred.
        """

        return list(
            compress(
                self.unpacked_values, ~np.array(list(self.unpacked_fixed.values()))
            )
        )

    def log_pdf(self, value, hyperparameters):
        """
        The natural logarithm of the distribution's probability density
        function at the given value.

        Parameters
        ----------
        value: float
            The value at which to evaluate the probability.
        hyperparameters: dict
            A dictionary of the hyperparameter values that define the current
            state of the distribution.

        Returns
        -------
        lnpdf:
            The natural logarithm of the probability.
        """

        return np.nan

    def pdf(self, value, hyperparameters):
        """
        The distribution's probability density function at the given value.

        Parameters
        ----------
        value: float
            The value at which to evaluate the probability.
        hyperparameters: dict
            A dictionary of the hyperparameter values that define the current
            state of the distribution.

        Returns
        -------
        pdf:
            The probability density.
        """

        return np.exp(self.log_pdf(value, hyperparameters))

    def sample(self, hyperparameters, size=1):
        """
        Draw a sample from the distribution as defined by the given
        hyperparameters.

        Parameters
        ----------
        hyperparameters: dict
            A dictionary of the hyperparameter values that define the current
            state of the distribution.
        size: int
            The number of samples to draw from the distribution.

        Returns
        -------
        sample:
            A sample, or set of samples, from the distribution.
        """

        return None


class BoundedGaussianDistribution(BaseDistribution):
    """
    A distribution to define estimating the parameters of a (potentially
    multi-modal) bounded Gaussian distribution.

    An example of using this distribution for a two component Gaussian
    distribution bounded at zero and with unknown mean, standard deviations and
    weights would be:

    >>> from bilby.core.prior import HalfNormal, LogUniform, DirichletPriorDict
    >>> # set priors for means (half-Normal distributions with mode at 0)
    >>> mus = [HalfNormal(10.0, name="mu0"), HalfNormal(10.0, name="mu1")]
    >>> # set priors for standard deviations (log uniform distributions)
    >>> sigmas = [LogUniform(name="sigma0", minimum=0.0001, maximum=100.0),
                  LogUniform(name="sigma1", minimum=0.0001, maximum=100.0)]
    >>> # set a Dirichlet prior on the weights (i.e., they must add up to 1)
    >>> weights = DirichletPriorDict(n_dim=2, label="weight")
    >>> dist = BoundedGaussianDistribution("x", mus=mus, sigmas=sigmas, weights=weights)

    Note that if usind a Dirichlet prior on the weights all weights must be
    included and none can be set as fixed.

    Parameters
    ----------
    name: str
        See :class:`~cwinpy.hierarchical.BaseDistribution`
    mus: array_like
        A list of values of the means of each mode of the Gaussian.
    sigmas: array_like
        A list of values of the standard deviations of each mode of the
        Gaussian.
    weights: array_like
        A list of values of the weights (relative probabilities) of
        each mode. This will default to equal weights if not given. If wanting
        to estimate multiple weights a DirichletPriorDict should be used as in
        the example above.
    low: float
        The lower bound of the distribution (defaults to 0, i.e., only positive
        values are allowed)
    high: float
        The upper bound of the distribution (default to infinity)

    """

    def __init__(self, name, mus=[], sigmas=[], weights=None, low=0.0, high=np.inf):
        gaussianparameters = {"mu": [], "sigma": [], "weight": []}

        if isinstance(mus, (int, float, bilby.core.prior.Prior)):
            mus = [mus]
        elif not isinstance(mus, (list, np.ndarray)):
            raise TypeError("Unknown type for 'mus'")

        if isinstance(sigmas, (int, float, bilby.core.prior.Prior)):
            sigmas = [sigmas]
        elif not isinstance(sigmas, (list, np.ndarray)):
            raise TypeError("Unknown type for 'sigmas'")

        if weights is None:
            weights = [1] * len(mus)
        elif not isinstance(
            weights, (list, np.ndarray, bilby.core.prior.DirichletPriorDict)
        ):
            raise TypeError("Unknown type for 'weights'")

        if isinstance(weights, bilby.core.prior.DirichletPriorDict):
            # DirichletPriorDict has length one less than the number of weights
            nweights = len(weights) + 1
            for wv in weights.values():
                gaussianparameters["weight"].append(wv)
        else:
            nweights = len(weights)

        # set the number of modes
        self.nmodes = len(mus)

        if len(mus) != len(sigmas) or nweights != len(mus):
            raise ValueError("'mus', 'sigmas' and 'weights' must be the same length")

        if self.nmodes < 1:
            raise ValueError("Gaussian must have at least one mode")

        for i in range(self.nmodes):
            gaussianparameters["mu"].append(mus[i])
            gaussianparameters["sigma"].append(sigmas[i])

            if isinstance(weights, (list, np.ndarray)):
                gaussianparameters["weight"].append(weights[i])

        # initialise
        super().__init__(
            name, "gaussian", hyperparameters=gaussianparameters, low=low, high=high
        )

    def log_pdf(self, value, hyperparameters={}):
        """
        The natural logarithm of the pdf of a 1d (potentially multi-modal)
        Gaussian probability distribution.

        Parameters
        ----------
        value: float
            The value at which the probability is to be evaluated.
        hyperparameters: dict
            A dictionary containing the current values of the hyperparameters
            that need to be inferred. If there are multiple modes and weights
            are not fixed then the hyperparameters should include ``n-1``
            weights values, where ``n`` is the number of modes.

        Returns
        -------
        logpdf:
            The natural logarithm of the probability density at the given
            value.
        """

        if np.any((value < self.low) | (value > self.high)):
            return -np.inf

        mus = self["mu"]
        sigmas = self["sigma"]

        if isinstance(self.fixed["weight"], (list, np.ndarray)):
            if np.any(np.asarray(self.fixed["weight"]) == True):  # noqa: E712
                weights = self["weight"]
            else:
                # all should be False for Dirichlet priors
                weights = np.zeros(self.nmodes)
        else:
            weights = np.zeros(self.nmodes)

        # get current mus and sigmas from values
        for i in range(self.nmodes):
            if not self.fixed["mu"][i]:
                param = "mu{}".format(i)
                try:
                    mus[i] = hyperparameters[param]
                except KeyError:
                    raise KeyError(
                        "Cannot calculate log probability when "
                        "value '{}' is not given".format(param)
                    )

            if not self.fixed["sigma"][i]:
                param = "sigma{}".format(i)
                try:
                    sigmas[i] = hyperparameters[param]
                except KeyError:
                    raise KeyError(
                        "Cannot calculate log probability when "
                        "value '{}' is not given".format(param)
                    )

            if i < (self.nmodes - 1):
                if not self.fixed["weight"][i]:
                    param = "weight{}".format(i)
                    try:
                        weights[i] = hyperparameters[param]
                    except KeyError:
                        raise KeyError(
                            "Cannot calculate log probability when "
                            "value '{}' is not given".format(param)
                        )

        if weights[self.nmodes - 1] == 0.0:
            # set final weight
            weights[self.nmodes - 1] = 1.0 - np.sum(weights[:-1])

        if np.any(np.asarray(sigmas) <= 0.0):
            return -np.inf

        if np.any(np.asarray(weights) <= 0.0):
            return -np.inf

        # normalise weights
        lweights = np.log(np.asarray(weights) / np.sum(weights))

        # get log pdf
        if isinstance(value, (float, int)):
            logpdf = -np.inf
        elif isinstance(value, (list, np.ndarray)):
            logpdf = np.full_like(value, -np.inf)
        else:
            raise TypeError("value must be a float or array-like")

        for mu, sigma, lweight in zip(mus, sigmas, lweights):
            lpdf = lweight + truncnorm.logpdf(
                value,
                (self.low - mu) / sigma,
                (self.high - mu) / sigma,
                loc=mu,
                scale=sigma,
            )
            logpdf = np.logaddexp(logpdf, lpdf)

        return logpdf

    def sample(self, hyperparameters={}, size=1):
        """
        Draw a sample from the bounded Gaussian distribution as defined by the
        given hyperparameters.

        Parameters
        ----------
        hyperparameters: dict
            A dictionary of the hyperparameter values that define the current
            state of the distribution. If there are multiple modes and weights
            are not fixed then the hyperparameters should include ``n-1``
            weights values, where ``n`` is the number of modes.
        size: int
            The number of samples to draw. Default is 1.

        Returns
        -------
        sample:
            A sample, or set of samples, from the distribution.
        """

        mus = self["mu"]
        sigmas = self["sigma"]

        if isinstance(self.fixed["weight"], (list, np.ndarray)):
            if np.any(np.asarray(self.fixed["weight"]) == True):  # noqa: E712
                weights = self["weight"]
            else:
                # all should be False for Dirichlet priors
                weights = np.zeros(self.nmodes)
        else:
            weights = np.zeros(self.nmodes)

        # get current mus and sigmas from values
        for i in range(self.nmodes):
            if not self.fixed["mu"][i]:
                param = "mu{}".format(i)
                try:
                    mus[i] = hyperparameters[param]
                except KeyError:
                    raise KeyError(
                        "Cannot calculate log probability when "
                        "value '{}' is not given".format(param)
                    )

            if not self.fixed["sigma"][i]:
                param = "sigma{}".format(i)
                try:
                    sigmas[i] = hyperparameters[param]
                except KeyError:
                    raise KeyError(
                        "Cannot calculate log probability when "
                        "value '{}' is not given".format(param)
                    )

            if i < (self.nmodes - 1):
                if not self.fixed["weight"][i]:
                    param = "weight{}".format(i)
                    try:
                        weights[i] = hyperparameters[param]
                    except KeyError:
                        raise KeyError(
                            "Cannot calculate log probability when "
                            "value '{}' is not given".format(param)
                        )

        if weights[self.nmodes - 1] == 0.0:
            # set final weight
            weights[self.nmodes - 1] = 1.0 - np.sum(weights[:-1])

        # cumulative normalised weights
        cweights = np.cumsum(np.asarray(weights) / np.sum(weights))

        # pick mode and draw sample
        if self.nmodes == 1:
            sample = truncnorm.rvs(
                (self.low - mus[0]) / sigmas[0],
                (self.high - mus[0]) / sigmas[0],
                loc=mus[0],
                scale=sigmas[0],
                size=size,
            )
        else:
            sample = np.zeros(size)
            for i in range(size):
                mode = np.argwhere(cweights - np.random.rand() > 0)[0][0]

                sample[i] = truncnorm.rvs(
                    (self.low - mus[mode]) / sigmas[mode],
                    (self.high - mus[mode]) / sigmas[mode],
                    loc=mus[mode],
                    scale=sigmas[mode],
                    size=1,
                )

            if size == 1:
                sample = sample[0]

        return sample


class ExponentialDistribution(BaseDistribution):
    """
    A distribution to define estimating the parameters of an exponential distribution.

    Parameters
    ----------
    name: str
        See :class:`~cwinpy.hierarchical.BaseDistribution`
    mu: float, Prior
        The mean of the exponential distribution.
    """

    def __init__(self, name, mu):
        # initialise
        super().__init__(
            name, "exponential", hyperparameters=dict(mu=mu), low=0.0, high=np.inf
        )

    def log_pdf(self, value, hyperparameters={}):
        """
        The natural logarithm of the pdf of an exponential distribution.

        Parameters
        ----------
        value: float
            The value at which the probability is to be evaluated.
        hyperparameters: dict
            A dictionary containing the current values of the hyperparameters
            that need to be inferred.

        Returns
        -------
        logpdf:
            The natural logarithm of the probability at the given value.
        """

        if np.any((value < self.low) | (value > self.high)):
            return -np.inf

        mu = self["mu"]
        if not self.fixed["mu"]:
            try:
                mu = hyperparameters["mu"]
            except KeyError:
                raise KeyError("Cannot evaluate the probability when mu is not given")

        if mu <= 0.0:
            return -np.inf

        # get log pdf
        logpdf = expon.logpdf(value, scale=mu)

        return logpdf

    def sample(self, hyperparameters={}, size=1):
        """
        Draw a sample from the exponential distribution as defined by the
        given hyperparameters.

        Parameters
        ----------
        hyperparameters: dict
            A dictionary of the hyperparameter values (``mu``) that define the
            current state of the distribution.
        size: int
            The number of samples to draw from the distribution.

        Returns
        -------
        sample:
            A sample, or set of samples, from the distribution.
        """

        mu = self["mu"]
        if not self.fixed["mu"]:
            try:
                mu = hyperparameters["mu"]
            except KeyError:
                raise KeyError("Cannot evaluate the probability when mu is not given")

        samples = expon.rvs(scale=mu, size=size)

        while 1:
            idx = (samples > self.low) & (samples < self.high)
            nvalid = np.sum(idx)

            if nvalid != size:
                sample = expon.rvs(scale=mu, size=(size - nvalid))
                samples[~idx] = sample
            else:
                break

        if size == 1:
            sample = samples[0]
        else:
            sample = samples

        return sample


class PowerLawDistribution(BaseDistribution):
    """
    A distribution to define estimating the parameters of a power law distribution.

    Parameters
    ----------
    name: str
        See :class:`~cwinpy.hierarchical.BaseDistribution`
    alpha: float, Prior
        The power law index of the distribution.
    minimum: float
        A positive finite value giving the lower cutoff of the distribution.
    maximum: float
        A positive finite value giving the upper cutoff of the distribution.
    """

    def __init__(self, name, alpha, minimum, maximum):
        if isinstance(minimum, float):
            if minimum <= 0 or not np.isfinite(minimum):
                raise ValueError(
                    "Minimum of distribution must be positive finite value"
                )

            if isinstance(maximum, float):
                if maximum < minimum:
                    raise ValueError(
                        "Maximum of distribution must be smaller than minimum!"
                    )

        if isinstance(maximum, float):
            if maximum <= 0 or not np.isfinite(maximum):
                raise ValueError(
                    "Maximum of distribution must be positive finite value"
                )

        # initialise
        super().__init__(
            name,
            "powerlaw",
            hyperparameters=dict(alpha=alpha, minimum=minimum, maximum=maximum),
        )

    def log_pdf(self, value, hyperparameters={}):
        """
        The natural logarithm of the pdf of a power law distribution.

        Parameters
        ----------
        value: float
            The value at which the probability is to be evaluated.
        hyperparameters: dict
            A dictionary containing the current values of the hyperparameters
            that need to be inferred.

        Returns
        -------
        logpdf:
            The natural logarithm of the probability at the given value.
        """

        alpha = self["alpha"]
        if not self.fixed["alpha"]:
            try:
                alpha = hyperparameters["alpha"]
            except KeyError:
                raise KeyError(
                    "Cannot evaluate the probability when alpha is not given"
                )

        minimum = self["minimum"]
        if not self.fixed["minimum"]:
            try:
                minimum = hyperparameters["minimum"]
            except KeyError:
                raise KeyError(
                    "Cannot evaluate the probability when minimum is not given"
                )
        elif np.any(value < minimum):
            return -np.inf

        maximum = self["maximum"]
        if not self.fixed["maximum"]:
            try:
                maximum = hyperparameters["maximum"]
            except KeyError:
                raise KeyError(
                    "Cannot evaluate the probability when maximum is not given"
                )
        elif np.any(value > maximum):
            return -np.inf

        # get log pdf
        logpdf = bilby.core.prior.PowerLaw(alpha, minimum, maximum).ln_prob(value)

        return logpdf

    def sample(self, hyperparameters={}, size=1):
        """
        Draw a sample from the exponential distribution as defined by the
        given hyperparameters.

        Parameters
        ----------
        hyperparameters: dict
            A dictionary of the hyperparameter values (``alpha``, ``minimum``
            and ``maximum``) that define the current state of the distribution.
        size: int
            The number of samples to draw from the distribution.

        Returns
        -------
        sample:
            A sample, or set of samples, from the distribution.
        """

        alpha = self["alpha"]
        if not self.fixed["alpha"]:
            try:
                alpha = hyperparameters["alpha"]
            except KeyError:
                raise KeyError(
                    "Cannot evaluate the probability when alpha is not given"
                )

        minimum = self["minimum"]
        if not self.fixed["minimum"]:
            try:
                minimum = hyperparameters["minimum"]
            except KeyError:
                raise KeyError(
                    "Cannot evaluate the probability when minimum is not given"
                )

        maximum = self["maximum"]
        if not self.fixed["maximum"]:
            try:
                maximum = hyperparameters["maximum"]
            except KeyError:
                raise KeyError(
                    "Cannot evaluate the probability when maximum is not given"
                )

        samples = bilby.core.prior.PowerLaw(alpha, minimum, maximum).sample(size=size)

        if size == 1:
            sample = samples[0]
        else:
            sample = samples

        return sample


class DeltaFunctionDistribution(BaseDistribution):
    """
    A distribution defining a delta function (useful if wanting to fix a
    parameter at a specific value if creating signals, or use as a null model).

    Parameters
    ----------
    name: str
        See :class:`~cwinpy.hierarchical.BaseDistribution`
    peak: float
        The value at which the delta function is non-zero.
    """

    def __init__(self, name, peak):
        # initialise
        super().__init__(name, "deltafunction", hyperparameters=dict(peak=peak))

    def log_pdf(self, value, hyperparameters={}):
        """
        The natural logarithm of the pdf of a delta function distribution.

        Parameters
        ----------
        value: float
            The value at which the probability is to be evaluated.
        hyperparameters: dict
            A dictionary containing the current values of the hyperparameters
            that need to be inferred.

        Returns
        -------
        logpdf:
            The natural logarithm of the probability at the given value.
        """

        peak = self["peak"]
        if not self.fixed["peak"]:
            try:
                peak = hyperparameters["peak"]
            except KeyError:
                raise KeyError("Cannot evaluate the probability when peak is not given")

        if value != peak:
            return -np.inf
        return 0.0

    def sample(self, hyperparameters={}, size=1):
        """
        Return the position of the delta function.

        Parameters
        ----------
        hyperparameters: dict
            A dictionary of the hyperparameter values (``peak``) that define
            the current state of the distribution.
        size: int
            The number of samples to draw from the distribution.

        Returns
        -------
        sample:
            A sample, or set of samples, from the distribution.
        """

        peak = self["peak"]
        if not self.fixed["peak"]:
            try:
                peak = hyperparameters["peak"]
            except KeyError:
                raise KeyError("Cannot evaluate the probability when peak is not given")

        if size == 1:
            return peak
        else:
            return peak * np.ones(size)


class HistogramDistribution(BaseDistribution):
    """
    A distribution to define estimating the bin weights of a non-parameteric
    histogram-type distribution. The priors for the bin weights will be a
    Dirichlet prior.

    An example of using this distribution for a 10 bin histogram would be:

    >>> # set the number of bins and bounds
    >>> nbins = 20
    >>> lowerbound = 0.0
    >>> upperbound = 10.0
    >>> dist = HistogramDistribution("x", low=lowerbound, high=upperbound, nbins=nbins)

    Parameters
    ----------
    name: str
        See :class:`~cwinpy.hierarchical.BaseDistribution`
    low: float
        The lower bound of the distribution (required).
    high: float
        The upper bound of the distribution (required).
    nbins: int
        An integer number of histogram bins to use (defaults to 10).

    """

    def __init__(self, name, low, high, nbins=10):
        binparameters = {"weight": []}

        if isinstance(nbins, int):
            if nbins < 1:
                raise ValueError("Histogram must have at least one bin.")
            self.nbins = nbins
        else:
            raise TypeError("Number of bins must be an integer")

        # set the histogram bin edges (add small buffer on upper bin to allow
        # points on the edge)
        self.binedges = np.linspace(low, high + 1e-8 * high, nbins + 1)

        # set Dirichlet priors on weights
        binparameters["weight"] = list(
            bilby.core.prior.DirichletPriorDict(
                n_dim=self.nbins,
                label="weight",
            ).values()
        )

        # initialise
        super().__init__(
            name, "histogram", hyperparameters=binparameters, low=low, high=high
        )

    def log_pdf(self, value, hyperparameters={}):
        """
        The natural logarithm of the pdf of a histogrammed distribution.

        Parameters
        ----------
        value: float
            The value at which the probability is to be evaluated.
        hyperparameters: dict
            A dictionary containing the current values of the hyperparameters
            that need to be inferred. For a histogram with ``n`` bins, the
            hyperparameters should include ``n-1`` weights values.

        Returns
        -------
        logpdf:
            The natural logarithm of the probability density at the given
            value.
        """

        if np.any((np.asarray(value) < self.low) | (np.asarray(value) > self.high)):
            return -np.inf

        weights = np.zeros(self.nbins)

        for i in range(self.nbins):
            param = "weight{}".format(i)
            if i < (self.nbins - 1):
                weights[i] = hyperparameters[param]
            else:
                # set final weight
                weights[i] = 1.0 - np.sum(weights[:-1])

        if np.any(weights <= 0.0):
            return -np.inf

        # get log of weights
        lweights = np.log(weights)

        # get log pdf
        if isinstance(value, (float, int)):
            logpdf = -np.inf
        elif isinstance(value, (list, np.ndarray)):
            logpdf = np.full_like(value, -np.inf)
        else:
            raise TypeError("value must be a float or array-like")

        binidxs = np.digitize(value, self.binedges)

        if isinstance(value, (float, int)):
            logpdf = lweights[binidxs - 1]
        else:
            for i in range(len(value)):
                logpdf[i] = lweights[binidxs[i] - 1]

        return logpdf

    def sample(self, hyperparameters={}, size=1):
        """
        Draw a sample from the histogram distribution as defined by the
        given hyperparameters.

        Parameters
        ----------
        hyperparameters: dict
            A dictionary of the hyperparameter values that define the current
            state of the distribution. If there are ``n`` bins in the
            histogram, then the hyperparameters should include ``n-1`` weights
            values.
        size: int
            The number of samples to draw. Default is 1.

        Returns
        -------
        sample:
            A sample, or set of samples, from the distribution.
        """

        rng = np.random.default_rng()
        weights = np.zeros(self.nbins)

        # get current weights
        for i in range(self.nbins):
            param = "weight{}".format(i)
            if i < (self.nbins - 1):
                weights[i] = hyperparameters[param]
            else:
                # set final weight
                weights[i] = 1.0 - np.sum(weights[:-1])

        # cumulative normalised weights
        cweights = np.cumsum(np.asarray(weights) / np.sum(weights))

        # pick bin and draw sample
        if self.nbins == 1:
            sample = rng.uniform(self.low, self.high, size=size)
        else:
            sample = np.zeros(size)
            for i in range(size):
                bin = np.argwhere(cweights - rng.uniform() > 0)[0][0]

                sample[i] = rng.uniform(
                    self.binedges[bin], self.binedges[bin + 1], size=1
                )

            if size == 1:
                sample = sample[0]

        return sample


def create_distribution(name, distribution, distkwargs={}):
    """
    Function to create a distribution.

    An example of creating an exponential distribution, with a half-Gaussian
    prior on the mean would be:

    >>> from bilby.core.prior import HalfGaussian
    >>> sigma = 1e34  # width of half-Gaussian prior on mu
    >>> distkwargs = {"mu": HalfGaussian(name="mu", sigma=sigma)}
    >>> expdist = create_distribution("q22", "exponential", distkwargs)

    An example of creating a bimodal-Gaussian distribution, with modes fixed at
    particular values, fixed weights, but log-uniform priors on the mode
    widths, would be:

    >>> from bilby.core.prior import LogUniform
    >>> min = 1e28  # minimum of the width prior
    >>> max = 1e38  # maximum of the width prior
    >>> modes = [0.0, 1e32]  # fixed modes
    >>> weights = [0.7, 0.3]  # fixed weights
    >>> sigmas = [
    >>>     LogUniform(name="sigma0", minimum=min, maximum=max),
    >>>     LogUniform(name="sigma1", minimum=min, maximum=max),
    >>> ]
    >>> distkwargs = {
    >>>     "mu": modes,  # set "mu" for the modes
    >>>     "sigma": sigmas,  # set "sigma" for the widths
    >>>     "weight": weights,  # set "weight" for the weights
    >>> }
    >>> gaussdist = create_distribution("q22", "gaussian", distkwargs)

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
            raise ValueError('Unknown distribution type "{}"'.format(distribution))

        if distribution.lower() == "gaussian":
            return BoundedGaussianDistribution(name, **distkwargs)
        elif distribution.lower() == "exponential":
            return ExponentialDistribution(name, **distkwargs)
        elif distribution.lower() == "deltafunction":
            return DeltaFunctionDistribution(name, **distkwargs)
        elif distribution.lower() == "powerlaw":
            return PowerLawDistribution(name, **distkwargs)
        elif distribution.lower() == "histogram":
            return HistogramDistribution(name, **distkwargs)
    else:
        raise TypeError("Unknown distribution")


class MassQuadrupoleDistribution(object):
    """
    A class to infer the hyperparameters of the :math:`l=m=2` mass quadrupole
    distribution (or fiducial ellipticity :math:`\\varepsilon`) for a given
    selection of known pulsars (see, for example, [2]_).

    The class currently can attempt to fit the hyperparameters for the
    following distributions:

    * a :math:`n`-mode bounded Gaussian distribution defined by either fixed or
      unknown means, standard deviations and weights;
    * an exponential distribution defined by an unknown mean.
    * a power law distribution defined by an unknown power law index and fixed
      or unknown bounds.

    All distributions do not allow the quadrupole value to become negative.

    Parameters
    ----------
    data: :class:`bilby.core.result.ResultList`
        A :class:`bilby.core.result.ResultList` of outputs from running source
        parameter estimation using bilby for a set of individual CW sources.
        These can be from MCMC or nested sampler runs, but only the latter can
        be used if requiring a properly normalised evidence value.
    gridrange: array_like
        A list of values at which the :math:`Q_{22}` parameter posteriors
        should be interpolated, or a lower and upper bound in the range of
        values, which will be split into ``bins`` points spaced linearly in
        log-space (unless ``gridtype'' is set to a value other than ``"log"``).
        If not supplied this will instead be set using the posterior samples,
        with a minimum value at zero and a maximum given by the maximum of all
        posterior samples.
    bins: int
        The number of bins at which the posterior will be interpolated.
    gridtype: str
        This sets the grid bin spacing used for assigning the interpolation
        grid. It defaults to spacings that are uniform in log-space for
        distributions other than
        :class:`cwinpy.hierarchical.HistogramDistribution` for which case the
        spacing defaults to linear. Values can either be ``"log"`` or
        ``"linear"`` to force one or other spacing.
    distribution: :class:`cwinpy.hierarchical.BaseDistribution`, str
        A predefined distribution, or string giving a valid distribution name.
        This is the distribution for which the hyperparameters are going to be
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
    integration_method: str
        The method to use for integration over the :math:`Q_{22}` parameter for
        each source. Default is 'numerical' to perform trapezium rule
        integration. Other allowed values are: 'sample' to sample over each
        individual :math:`Q_{22}` parameter for each source; or, 'expectation',
        which uses the :math:`Q_{22}` posterior samples to approximate the
        expectation value of the hyperparameter distribution. At the moment,
        these two additional methods may not be correct/reliable.
    nsamples: int
        This sets the number of posterior samples to store for either
        estimating KDEs or calculating expectation values from those passed in
        the data. This allows downsampling of large numbers of samples by
        randomly drawing a subsection of samples. If the number given is larger
        than the total number of samples for a given pulsar, then all samples
        will be used in that case. The default will be to use all samples, but
        this may lead to memory issues when using large numbers of pulsars.
    use_ellipticity: bool
        If True, work with fiducial ellipticity :math:`\\varepsilon` rather
        than mass quadrupole.

    To do
    -----

    Distributions that could be added include:

    * a Student's t-distributions with unknown mean and number of degrees of
      freedom.
    """

    def __init__(
        self,
        data=None,
        gridrange=None,
        bins=100,
        gridtype=None,
        distribution=None,
        distkwargs=None,
        bw="scott",
        sampler="dynesty",
        sampler_kwargs={},
        grid=None,
        integration_method="numerical",
        nsamples=None,
        use_ellipticity=False,
    ):
        self._posterior_samples = []
        self._pulsar_priors = []
        self._log_evidence = []
        self._likelihood_kdes_interp = []
        self._distribution = None

        # set whether to use ellipticity rather than mass quadrupole
        self.use_ellipticity = use_ellipticity

        # set the values of q22/ellipticity at which to calculate the KDE
        # interpolator
        self.set_range(gridrange, bins, gridtype=gridtype)

        # set integration method
        self.set_integration_method(integration_method)

        # set the data
        self.add_data(data, bw=bw, nsamples=nsamples)

        # set the sampler
        if grid is None:
            self.set_sampler(sampler, sampler_kwargs)
        else:
            self.set_grid(grid)

        # set the distribution
        self.set_distribution(distribution, distkwargs)

    def set_range(self, gridrange, bins=100, gridtype=None):
        """
        Set the values of :math:`Q_{22}`, or ellipticity :math:`\\varepsilon`,
        either directly, or as a set of points linear in log-space defined by
        a lower and upper bounds and number of bins, at which to evaluate the
        posterior samples via their KDE to make an interpolator.

        Parameters
        ----------
        gridrange: array_like
            If this array contains two values it is assumed that these are the
            lower and upper bounds of a range, and the ``bins`` parameter sets
            the number of bins in log-space that the range will be split into.
            Otherwise, if more than two values are given it is assumed these
            are the values for :math:`Q_{22}` or :math:`\\varepsilon`.
        bins: int
            The number of bins the range is split into.
        gridtype: str
            Set whether to have grid-spacing be ``"linear"`` or linear in
            log-10 space (``"log"``). By default, for distribution's other than
            :class:`cwinpy.hierarchical.HistogramDistribution` the default will
            be linear in log-10 space.
        """

        self._bins = bins

        if gridrange is None:
            self._grid_interp_values = None
            return

        if len(gridrange) == 2:
            if gridrange[1] < gridrange[0]:
                raise ValueError("Grid range is badly defined")

            # set grid spacing (either linear or linear in log10-space)
            lower, upper = gridrange
            if (
                gridtype is None
                and not isinstance(self._distribution, HistogramDistribution)
            ) or gridtype == "log":
                self._grid_interp_values = np.logspace(
                    np.log10(gridrange[0]), np.log10(gridrange[1]), self._bins
                )
            else:
                self._grid_interp_values = np.linspace(
                    gridrange[0], gridrange[1], self._bins
                )
        elif len(gridrange) > 2:
            self._grid_interp_values = gridrange
        else:
            raise ValueError("Grid range is badly defined")

    @property
    def interpolated_log_kdes(self):
        """
        Return the list of interpolation functions for the natural logarithm of
        the :math:`Q_{22}` likelihood functions after a Gaussian KDE has been
        applied.
        """

        return self._likelihood_kdes_interp

    def add_data(self, data, bw="scott", nsamples=None):
        """
        Set the data, i.e., the individual source posterior distributions, on
        which the hierarchical analysis will be performed.

        The posterior samples must include the ``Q22`` :math:`l=m=2` parameter,
        or the fiducial ellipticity parameter ``ELL``, for this inference to be
        performed.

        If using the "numerical" integration method, upon running the
        :meth:`~cwinpy.hierarchical.MassQuadrupoleDistribution.sample` method,
        these samples will be converted to a KDE (reflected about zero
        to avoid edge effects, and re-normalised, although the bandwidth will
        be calculated using the unreflected samples), using
        :class:`scipy.stats.gaussian_kde`, which will be used as the
        data for hierarchical inference. If the posterior
        samples come with a Bayesian evidence value, and the prior is present,
        then these are used to convert the posterior distribution into a
        likelihood, which is what is then stored in the interpolation function.

        If using the "expectation" integration method, and if the posterior
        samples were not estimated using a uniform prior on ``Q22``/``ELL``,
        then the samples will be resampled from a uniform prior to attempt to
        generate samples from the likelihood.

        Parameters
        ----------
        data: :class:`bilby.core.result.ResultList`
            A list, or single, results from bilby containing posterior samples
            for a set of sources, or individual source.
        bw: str, scale, callable
            The Gaussian KDE bandwidth calculation method as required by
            :class:`scipy.stats.gaussian_kde`. The default is the 'scott'
            method.
        nsamples: int
            This sets the number of posterior samples to store and use from
            those passed in the data. This allows downsampling of large numbers
            of samples by randomly drawing a subsection of samples. If the
            number given is larger than the total number of samples for a given
            pulsar, then all samples will be used in that case. The default
            will be to use all samples, but this may lead to memory issues when
            using large numbers of pulsars.
        """

        # check the data is a ResultList
        if not isinstance(data, bilby.core.result.ResultList):
            if isinstance(data, (bilby.core.result.Result, str)):
                # convert to a ResultList
                data = bilby.core.result.ResultList([data])
            elif isinstance(data, list):
                data = bilby.core.result.ResultList(data)
            elif data is None:
                return
            else:
                raise TypeError("Data is not a known type")

        self._bw = bw

        for result in data:
            # check all posteriors contain Q22 or ellipticity
            if (
                "Q22" not in result.posterior.columns
                and "q22" not in result.posterior.columns
            ):
                if (
                    "ELL" in result.posterior.columns
                    or "ell" in result.posterior.columns
                ):
                    priorkey = "ell" if "ell" in result.posterior.columns else "ELL"
                    if not self.use_ellipticity:
                        # convert ellipticity into q22
                        result.posterior["q22"] = ellipticity_to_q22(
                            result.posterior[priorkey]
                        )
                else:
                    raise RuntimeError("Results do not contain Q22")
            else:
                priorkey = "q22" if "q22" in result.posterior.columns else "Q22"
                if self.use_ellipticity:
                    result.posterior["ell"] = q22_to_ellipticity(
                        result.posterior[priorkey]
                    )

            self._pulsar_priors.append(result.priors[priorkey])

        if nsamples is not None:
            if not isinstance(nsamples, int):
                raise TypeError("nsamples must be a positive integer")
            elif nsamples < 1:
                raise ValueError("nsamples must be a positive integer")

        # set number of samples to use
        if not hasattr(self, "_nsamples") and nsamples is not None:
            self._nsamples = nsamples
            numsamps = nsamples
        elif hasattr(self, "_nsamples") and nsamples is None:
            numsamps = self._nsamples
        else:
            numsamps = nsamples

        # create KDEs/add samples
        iniidx = len(self._posterior_samples)
        for i, result in enumerate(data):
            if self.use_ellipticity:
                keystr = "ell" if "ell" in result.posterior.columns else "ELL"
            else:
                keystr = "q22" if "q22" in result.posterior.columns else "Q22"

            samples = result.posterior[keystr]

            # reweight samples back to equivalent likelihood samples if prior
            # on Q22/ELL for PE was not uniform
            prior = self._pulsar_priors[iniidx + i]
            if self._integration_method == "expectation" and not isinstance(
                prior, bilby.core.prior.Uniform
            ):
                # resample to uniform prior
                possamps = result.posterior[keystr]
                lnweights = prior.log_prob(possamps)
                weights = np.exp(lnweights - np.max(lnweights))
                samples = possamps[weights > np.random.rand(len(weights))]

            # append samples
            if numsamps is None:
                self._posterior_samples.append(np.array(samples))
            else:
                if len(samples) < numsamps:
                    self._posterior_samples.append(np.array(samples))
                else:
                    # generate random choice of samples to store
                    sidx = np.random.default_rng().choice(
                        len(samples), numsamps, replace=False
                    )
                    self._posterior_samples.append(np.array(samples)[sidx])

            self._log_evidence.append(result.log_evidence)

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

        self._prior = None
        self._likelihood = None

        if distribution is None:
            return

        if isinstance(distribution, BaseDistribution):
            if self.use_ellipticity:
                if distribution.name.upper() != "ELL":
                    raise ValueError("Distribution name must be 'ELL'")
                else:
                    self._distribution = distribution
            else:
                if distribution.name.upper() != "Q22":
                    raise ValueError("Distribution name must be 'Q22'")
                else:
                    self._distribution = distribution
        elif isinstance(distribution, str):
            if self.use_ellipticity:
                self._distribution = create_distribution(
                    "ELL", distribution.lower(), distkwargs
                )
            else:
                self._distribution = create_distribution(
                    "Q22", distribution.lower(), distkwargs
                )

        # set the priors from the distribution
        self._set_priors()

    def _set_priors(self):
        """
        Set the priors based on those supplied via the distribution class.
        """

        # get the priors from the distribution
        if len(self._distribution.unknown_parameters) < 1:
            raise ValueError("Distribution has no parameters to infer")

        # add priors as PriorDict
        self._prior = None

        # check for Dirichlet priors
        for param, prior in zip(
            self._distribution.unknown_parameters, self._distribution.unknown_priors
        ):
            if isinstance(prior, bilby.core.prior.DirichletElement):
                self._prior = bilby.core.prior.DirichletPriorDict(
                    n_dim=prior.n_dimensions, label=prior.label
                )
                break

        if self._prior is None:
            self._prior = bilby.core.prior.ConditionalPriorDict()

        for param, prior in zip(
            self._distribution.unknown_parameters, self._distribution.unknown_priors
        ):
            if param not in self._prior:
                self._prior[param] = prior

    def _set_likelihood(self):
        """
        Set the likelihood.
        """

        samples = None
        grid = None
        likelihoods = None

        # set the grid
        if self._integration_method == "expectation":
            samples = self._posterior_samples
        else:
            if self._grid_interp_values is None:
                # set parameter range from data
                minmax = [np.inf, -np.inf]
                for psamples in self._posterior_samples:
                    minval = psamples.min()
                    maxval = psamples.max()
                    if minval < minmax[0]:
                        minmax[0] = minval
                    if maxval > minmax[1]:
                        minmax[1] = maxval

                self.set_range(minmax, self._bins)

            grid = self._grid_interp_values

            # generate KDEs from samples and create spline interpolants
            nkdes = len(self._likelihood_kdes_interp)
            if len(self._posterior_samples) > nkdes:
                for i in range(nkdes, len(self._posterior_samples)):
                    psamples = self._posterior_samples[i]
                    try:
                        # get reflected samples
                        samps = np.concatenate((psamples, -psamples))

                        # calculate the KDE initially using the unreflected
                        # samples to get a better bandwidth and prevent
                        # artificially broadened distributions
                        kdeorig = gaussian_kde(psamples, bw_method=self._bw)

                        # calculate KDE (using new bandwidth equivalent to that
                        # for unreflected samples)
                        bw = np.sqrt(kdeorig.covariance[0][0] / np.var(samps))
                        kde = gaussian_kde(samps, bw_method=bw)

                        # use log pdf for the kde
                        interpvals = kde.logpdf(self._grid_interp_values) + np.log(
                            2.0
                        )  # multiply by 2 so pdf normalises to 1

                        # replace any infinity values with small number (logpdf
                        # returns inf rather than -inf, so we need to flip the
                        # sign)
                        infvals = ~np.isfinite(interpvals)
                        if np.any(infvals):
                            interpvals[infvals] = -np.inf
                            interpvals = np.nan_to_num(interpvals)

                    except Exception as e:
                        raise RuntimeError("Problem creating KDE: {}".format(e))

                    # convert posterior to likelihood (if possible)
                    if np.isfinite(self._log_evidence[i]):
                        # multiply by evidence
                        interpvals += self._log_evidence[i]

                    # divide by prior
                    interpvals -= self._pulsar_priors[i].ln_prob(
                        self._grid_interp_values
                    )

                    # create and add interpolator (the tck tuple for a B-spline)
                    self._likelihood_kdes_interp.append(
                        splrep(self._grid_interp_values, interpvals)
                    )

            likelihoods = self._likelihood_kdes_interp

        self._likelihood = MassQuadrupoleDistributionLikelihood(
            self._distribution, likelihoods=likelihoods, samples=samples, grid=grid
        )

    def set_sampler(self, sampler="dynesty", sampler_kwargs={}):
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
            raise ValueError(
                'Sampler "{}" is not implemented in "bilby"'.format(self._sampler)
            )
        self._sampler_kwargs = sampler_kwargs
        self._use_grid = False  # set to not use the Grid sampling

    def set_grid(self, grid):
        """
        Set a grid on which to evaluate the hyperparameter posterior, as used
        by :class:`bilby.core.grid.Grid`.

        Parameters
        ----------
        grid: dict
            A dictionary of values that define a grid in the hyperparameter
            space that can be used by a :class:`bilby.core.grid.Grid` class.
        """

        if not isinstance(grid, dict):
            raise TypeError("Grid must be a dictionary")

        self._grid = grid
        self._use_grid = True

    def set_integration_method(self, integration_method="numerical"):
        """
        Set the method to use for integration over the :math:`Q_{22}` parameter
        for each source.

        Parameters
        ----------
        integration_method: str
            Default is 'numerical' to perform trapezium rule integration.
            The ther allowed value is 'expectation', which uses the
            :math:`Q_{22}` posterior samples to approximate the expectation
            value of the hyperparameter distribution.
        """

        if not isinstance(integration_method, str):
            raise TypeError("integration method must be a string")

        if integration_method.lower() not in ["numerical", "expectation"]:
            raise ValueError(
                "Unrecognised integration method type "
                "'{}'".format(integration_method)
            )

        self._integration_method = integration_method.lower()

    @property
    def result(self):
        """
        Return the ``bilby`` object containing the results. If evaluating the
        posterior over a grid this is a :class:`bilby.core.grid.Grid` object.
        If sampling using a stochastic sampler, this is a
        :class:`bilby.core.result.Result` object. If sampling has not yet been
        performed this returns ``None``.
        """

        if self._use_grid:
            if hasattr(self, "_grid_result"):
                return self._grid_result
            else:
                return None
        else:
            if hasattr(self, "_result"):
                return self._result
            else:
                return None

    def sample(self, **run_kwargs):
        """
        Sample the posterior distribution using ``bilby``. This can take
        keyword arguments required by the bilby
        :func:`~bilby.core.sampler.run_sampler` function.
        """

        # set up the likelihood function
        self._set_likelihood()

        # set use_ratio to False by default, i.e., don't use the likelihood
        # ratio
        run_kwargs.setdefault("use_ratio", False)

        if self._use_grid:
            self._grid_result = bilby.core.grid.Grid(
                self._likelihood, self._prior, grid_size=self._grid
            )
        else:
            self._result = bilby.run_sampler(
                likelihood=self._likelihood,
                priors=self._prior,
                sampler=self._sampler,
                **self._sampler_kwargs,
                **run_kwargs
            )

        return self.result

    def posterior_predictive(self, points, nsamples=100):
        """
        Return an iterator that will draw samples from the distribution
        hyperparameter posterior (once
        :meth:`~cwinpy.hierarchical.MassQuadrupoleDistribution.sample` has been
        run) and returns the associated distribution evaluated at a set of
        points.

        Currently this is only implemented to work using samples from a
        stochastic sampling method rather than posteriors evaluated on a grid.

        Parameters
        ----------
        points: array_like
            An array of Q22/ellipticity values at which to evaluate the
            distribution.
        nsamples: int
            The number of samples to draw from the distribution. This defaults
            to 100, but must be less than the number of posterior samples.
        """

        if self.result is None:
            raise RuntimeError("Sampling has not yet been performed")

        if self._use_grid:
            raise RuntimeError("Posterior predictive check can only use samples")

        # check grid
        if not isinstance(points, (tuple, list, np.ndarray)):
            raise TypeError("points must be array_like")

        if nsamples > len(self.result.posterior):
            raise ValueError(
                "Requested number of samples is greater than the number of posterior samples"
            )

        # chose random indexes of samples
        idx = np.random.choice(len(self.result.posterior), nsamples, replace=False)

        for i in idx:
            # get parameters of distribution for each sample
            hyper = {
                key: self.result.posterior[key][i]
                for key in self._distribution.unpacked_parameters
                if key in self.result.posterior.columns
            }

            # evaluate the distribution
            yield self._distribution.pdf(np.asarray(points), hyper)


class MassQuadrupoleDistributionLikelihood(bilby.core.likelihood.Likelihood):
    """
    The likelihood function for the inferring the hyperparameters of the
    mass quadrupole, :math:`Q_{22}`, distribution (or equivalently the
    fiducial ellipticity distribution).

    Parameters
    ----------
    distribution: :class:`cwinpy.hierarchical.BaseDistribution`
        The probability distribution for which the hyperparameters are going
        to be inferred.
    likelihoods: list
        A list of interpolation functions each of which gives the likelihood
        function for a single source.
    grid: array_like
        If given, the integration over the mass quadrupole distribution for
        each source is performed numerically on at these grid points. If not
        given, individual samples from :math:`Q_{22}` will be drawn from each
        source (i.e., equivalent to having a new :math:`Q_{22}` parameter for
        each source in the sampler).
    samples: list
        A list of arrays of :math:`Q_{22}` samples for each source. If this is
        given then these samples will be used to approximate the integral over
        independent :math:`Q_{22}` variables for each source.
    """

    def __init__(self, distribution, likelihoods=None, grid=None, samples=None):
        if not isinstance(distribution, BaseDistribution):
            raise TypeError("Distribution is not the correct type")

        # check that the distribution contains parameters to be inferred
        if len(distribution.unknown_parameters) < 1:
            raise ValueError("Distribution has no parameters to infer")

        # set parameters to be inferred
        inferred_parameters = {param: None for param in distribution.unknown_parameters}
        self.distribution = distribution
        self.grid = grid
        self.likelihoods = likelihoods
        self.samples = samples

        super().__init__(parameters=inferred_parameters)

    @property
    def likelihoods(self):
        return self._likelihoods

    @likelihoods.setter
    def likelihoods(self, like):
        if like is None:
            self._likelihoods = None
            self._nsources = 0
        elif not isinstance(like, list):
            raise TypeError("Likelihoods must be a list")
        else:
            if self.grid is not None:
                # evaluate the interpolated (log) likelihoods on the grid
                self._likelihoods = []
                for ll in like:
                    self._likelihoods.append(splev(self.grid, ll))
                self._nsources = len(like)
            else:
                raise ValueError("Grid must be set to evaluate likelihoods")

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        if isinstance(grid, (list, np.ndarray)):
            self._grid = np.asarray(grid)
        elif grid is None:
            self._grid = None
        else:
            raise TypeError("Grid must be array-like")

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        if samples is not None:
            if not isinstance(samples, (list, np.ndarray)):
                raise TypeError("samples value must be a list")

            if isinstance(samples, np.ndarray):
                if len(samples.shape) != 2:
                    raise ValueError("Samples must be a 2D array")

            for samplelist in samples:
                if not isinstance(samplelist, (list, np.ndarray)):
                    raise TypeError("Samples must be a list")

                if len(np.asarray(samplelist).shape) != 1:
                    raise ValueError("Source samples must be a 1d list")

            self._nsources = len(samples)

        self._samples = samples

    def log_likelihood(self):
        """
        Evaluate the log likelihood.
        """

        log_like = 0.0  # initialise the log likelihood

        if self.samples is not None:
            # log-likelihood using expectation value from samples
            for samps in self.samples:
                with np.errstate(divide="ignore"):
                    log_like += np.log(
                        np.mean(self.distribution.pdf(samps, self.parameters))
                    )
            log_like = np.nan_to_num(log_like)
        else:
            # evaluate the hyperparameter distribution
            logp = self.distribution.log_pdf(self.grid, self.parameters)

            # log-likelihood numerically integrating over grid
            for logl in self.likelihoods:
                log_like += logtrapz(logp + logl, self.grid, disable_checks=True)

        return log_like

    def noise_log_likelihood(self):
        """
        The log-likelihood for the unknown hyperparameters being equal to
        with zero.

        Note
        ----

        For distributions with hyperparameter priors that exclude zero this
        will always given :math:`-\\infty`.
        """

        for p in self.parameters:
            self.parameters[p] = 0.0

        return self.log_likelihood()

    def __len__(self):
        return self._nsources
