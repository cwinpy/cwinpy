"""
Classes providing likelihood functions.
"""

import numpy as np
from bilby import Likelihood, PriorDict, DeltaFunction
from .data import HeterodynedData, MultiHeterodynedData
import lalpulsar
from lalpulsar.simulateHeterodynedCW import HeterodynedCWSimulator


class TargetedPulsarLikelihood(Likelihood):
    """
    A likelihood, based on the :class:`bilby.core.likelihood.Likelihood`, for
    use in source parameter estimation for continuous wave signals, with
    particular focus on emission from known pulsars.
    
    Parameters
    ----------
    data: (str, HeterodynedData, MultiHeterodynedData)
        A :class:`~cwinpy.data.HeterodynedData` or
        :class:`~cwinpy.data.MultiHeterodynedData` object containing all
        required data.
    priors: bilby.PriorDict
        A :class:`bilby.core.prior.PriorDict` containing the parameters that
        are to be sampled. This is required to be set before this class is
        initialised.
    likelihood: str
        A string setting which likelihood function to use. This can either be
        'studentst' (the default), or 'gaussian' ('roq' should be added in the
        future).
    """

    # a set of parameters that define the "amplitude" of the signal (i.e.,
    # they do not effect the phase evolution of the signal)
    AMPLITUDE_PARAMS = ['H0', 'PHI0', 'PSI', 'IOTA', 'COSIOTA', 'C21', 'C22',
                        'PHI21', 'PHI22']

    # the set of potential non-GR "amplitude" parameters
    NONGR_AMPLITUDE_PARAM = ['PHI01TENSOR', 'PHI02TENSOR',
                             'PHI01VECTOR', 'PHI02VECTOR',
                             'PHI01SCALAR', 'PHI02SCALAR',
                             'PSI1TENSOR', 'PSI2TENSOR',
                             'PSI1VECTOR', 'PSI2VECTOR',
                             'PSI1SCALAR', 'PSI2SCALAR',
                             'H1PLUS', 'H2PLUS', 'H1CROSS', 'H2CROSS',
                             'H1VECTORX', 'H2VECTORX',
                             'H1VECTORY', 'H2VECTORY',
                             'H1SCALARB', 'H2SCALARB',
                             'H1SCALARL', 'H2SCALARL']

    def __init__(self, data, priors, par=None, det=None,
                 likelihood='studentst'):

        Likelihood.__init__(self, dict())  # initialise likelihood class

        # set the data
        if isinstance(data, HeterodynedData):
            # convert HeterodynedData to MultiHeterodynedData object
            if data.detector is None:
                raise ValueError("'data' must contain a named detector.")

            if data.par is None:
                raise ValueError("'data' must contain a heterodyne parameter "
                                 "file.")

            self.data = MultiHeterodynedData(data)
        elif isinstance(data, MultiHeterodynedData):
            if None in data.pars:
                raise ValueError("A data set does not have an associated "
                                 "heterodyned parameter set.")

            self.data = data
        else:
            raise TypeError("'data' must be a HeterodynedData or "
                            "MultiHeterodynedData object.")

        if not isinstance(priors, PriorDict):
            raise TypeError("Prior must be a bilby PriorDict")
        else:
            self.priors = priors

        # set the likelihood function
        self.likelihood = likelihood

        # check if prior includes (non-DeltaFunction) non-amplitude parameters,
        # i.e., will the phase evolution have to be included in the search
        self.include_phase = False
        for key in self.priors:
            if (key.upper() not in self.AMPLITUDE_PARAMS and
                    not isinstance(self.priors[key], DeltaFunction)):
                self.include_phase = True
        
        # check if any non-GR "amplitude" parameters are set
        self.nonGR = False
        for key in self.priors:
            if key.upper() in self.NONGR_AMPLITUDE_PARAM:
                self.nonGR = True

        # set up signal model classes
        self.models = []
        for het in self.data:
            self.models.append(HeterodynedCWSimulator(het.par, het.detector,
                                                      het.times))

        # if phase evolution is not in the model set the pre-summed quadratures
        # of the data and antenna patterns
        if not self.include_phase:
            self.presummed_quadratures()

    @property
    def likelihood(self):
        """
        The the string containing the 'type' of likelihood function to use:
        'studentst' or 'gaussian'.
        """

        return self.__likelihood

    @likelihood.setter
    def likelihood(self, likelihood):
        # make sure the likelihood is of the allowed type
        if likelihood.lower() not in ['studentst', 'students-t', 'studentt',
                                      'gaussian', 'normal']:
            raise ValueError("Likelihood must be 'studentst' or 'gaussian'.")

        if likelihood.lower() in ['studentst', 'students-t', 'studentt']:
            self.__likelihood = 'studentst'
        else:
            self.__likelihood = 'gaussian'

    def presummed_quadratures(self):
        """
        Calculate the pre-summed quadratures of the data and the antenna
        pattern functions. These are:

        .. math::

           \sum \\frac{dd^*}{\sigma^2}
        """

        self.quadratures = []  # list of quadratures for each data set

        # loop over HeterodynedData and model functions
        for data, model in zip(self.data, self.models):
            self.quadratures.append(dict())

            sumdata = np.zeros(data.num_chunks)

            # loop over "chunks" into which each data set has been split
            for i, cpidx, cplen in zip(range(data.num_chunks),
                                       data.change_point_indices,
                                       data.chunk_lengths):
                B = data.data[cpidx:cpidx + cplen]
                if self.likelihood == 'gaussian':
                    var = data.vars[cpidx:cpidx + cplen]
                else:
                    var = 1.

                # get the sum of the data squared
                sumdata[i] = np.sum(B.real / var) + np.sum(B.imag / var)

                # get the response functions