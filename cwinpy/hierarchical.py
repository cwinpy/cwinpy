"""
Classes for hierarchical parameter inference.
"""

import numpy
import bilby


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

    @type.setter
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

    def __getitem__(self, item):
        if item.lower() in self.parameters:
            return self.hyperparameters[item.lower()]
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
    means: array_like
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
    distribution: :class:`cwinpy.hierarchical.BaseDistribution`

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

    def __init__(self, data, q22range, q22bins=100, distribution=None):
        self._data = None
        self._posterior_kdes = None
        self._posterior_kdes_interp = None

        # set the values of q22 at which to calculate the KDE interpolator
        self.set_q22range(q22range, q22bins)

        self.add_data(data)  # set the data

    def set_q22range(self, q22range, q22bins=100):
        """
        Set the values of :math:`Q_{22}`, either directly, or as a set of
        points linear in
        log-space defined by a lower and upper bounds and number of bins, at
        which to evaluate the posterior samples via their KDE to make an
        interpolator.

        Parameters
        ----------
        q22range: array_like
            If this array contains two values it is assumed that these are the
            lower and upper bounds of a range, and the Q22
        """

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
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        """
        Set the data, i.e., the individual source posterior distributions, on
        which the hierarchical analysis will be performed.

        The posterior samples must include the ``Q22`` :math:`l=m=2` parameter
        for this inference to be performed. The samples will be converted to
        a KDE (reflected about zero to avoid edge effects, and re-normalised)
        which can be used as the data for hierarchical inference.
        """

        # check the data is a ResultList
        if not isinstance(data, bilby.core.result.ResultList):
            if isinstance(data, (bilby.core.Result, str)):
                # convert to a ResultList
                data = bilby.core.result.ResultList([data])
            elif isinstance(data, list):
                data = bilby.core.result.ResultList(data)
            else:
                raise TypeError('Data is not a known type')

        # create KDEs
        for result in data:
            # check all posteriors contain Q22
            if 'Q22' not in result.search_parameter_keys:
                raise RuntimeError("Results do not contain Q22")

            try:
                from statsmodels.nonparametric.bandwidths import bw_scott
                from sklearn.neighbors import KernelDensity

                samples = result.samples['Q22']

                # get the Gaussian kernel bandwidth
                bw = bw_scott(samples)

                # get reflected samples
                samps = np.concatenate((samples, -samples))[:, np.newaxis]

                # calculate KDE
                kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(samps)

                # use sklearn KDE as it returns natural logarithms of KDE (and so doesn't go to zero)
                interpvals = kde.score_samples(self._q22_interp_values[:, np.newaxis]) + np.log(2.) # multiply by 2 to make sure pdf normalises to 1
            except:
                raise RuntimeError("Problem creating KDE")
            
            # create interpolator

            # append interpolator

        # append data
        if self._data is None:
            self._data = data
        else:
            self._data.append(data)

        
