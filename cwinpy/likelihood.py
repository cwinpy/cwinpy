"""
Classes providing likelihood functions.
"""

import numpy as np
from bilby import Likelihood, PriorDict
from .data import HeterodynedData, MultiHeterodynedData
import lalpulsar
from lalpulsar.simulateHeterodynedCW import HeterodynedCWSimulator
from collections import OrderedDict
import copy
import re


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
                        'PHI21', 'PHI22', 'I21', 'I31', 'LAMBDA', 'COSTHETA',
                        'THETA', 'Q22', 'DIST']
    
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

    # the set of positional parameters
    POSITIONAL_PARAMETERS = ['RAJ', 'DECJ', 'RA', 'DEC', 'PMRA', 'PMDEC',
                             'POSEPOCH']

    # the set of potential binary parameters
    BINARY_PARAMS = ["PB", "ECC", "EPS1", "EPS2", "T0", "TASC", "A1", "OM",
                     "PB_2", "ECC_2", "T0_2", "A1_2", "OM_2", "PB_3", "ECC_3",
                     "T0_3", "A1_3", "OM_3", "XPBDOT", "EPS1DOT", "EPS2DOT",
                     "OMDOT", "GAMMA", "PBDOT", "XDOT", "EDOT", "SINI", "DR",
                     "DTHETA", "A0", "B0", "MTOT", "M2", "FB"]

    # the parameters that are held as vectors
    VECTOR_PARAMS = ['F', 'GLEP', 'GLPH', 'GLF0', 'GLF1', 'GLF2', 'GLF0D',
                     'GLTD', 'FB']

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
        # i.e., will the phase evolution have to be included in the search,
        # or binary parameters
        self.include_phase = False
        self.include_binary = False
        self.update_ssb = False
        for key in self.priors:
            if key.upper() not in self.AMPLITUDE_PARAMS:
                if self.priors[key].is_fixed:
                    # check if it's the same value as the par files
                    for het in self.data:
                        if het.par[key.upper()] != self.priors[key].peak:
                            self.include_phase = True
                            break
                else:
                    self.include_phase = True
            if key.upper() in self.BINARY_PARAMS:
                if self.priors[key].is_fixed:
                    # check if it's the same value as the par files
                    for het in self.data:
                        if het.par[key.upper()] != self.priors[key].peak:
                            self.include_binary = True
                            break
                else:
                    self.include_binary = True
            if key.upper() in self.POSITIONAL_PARAMETERS:
                self.update_ssb = True

        # check if any non-GR "amplitude" parameters are set
        self.nonGR = False
        for key in self.priors:
            if key.upper() in self.NONGR_AMPLITUDE_PARAM:
                self.nonGR = True

        # set up signal model classes
        self.models = []
        self.basepars = []
        for het in self.data:
            self.models.append(HeterodynedCWSimulator(het.par, het.detector,
                                                      het.times))
            # copy of heterodyned parameters
            self.basepars.append(copy.deepcopy(het.par))

        # if phase evolution is not in the model set the pre-summed products
        # of the data and antenna patterns
        if not self.include_phase:
            self.dot_products()

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

    def dot_products(self):
        """
        Calculate the (noise-weighted) dot products of the data and the
        antenna pattern functions. E.g., for the data this is the real value

        .. math::

           (d/\sigma) \cdot (d*/\sigma) = \sum_i \\frac{d_id_i^*}{\sigma_i^2} \equiv \sum_i \\frac{\Re{(d)}^2 + \Im{(d)}^2}{\sigma_i_^2}.
        
        For the antenna patterns, for example :math:`F_+` and
        :math:`F_{\\times}`, we would have

        .. math::

           (F_+/\sigma) \cdot (F_+/\sigma) = \sum_i \\frac{{F_+}_i^2}{\sigma_i^2},
           
           (F_\\times/\sigma) \cdot (F_\\times/\sigma) = \sum_i \\frac{{F_\\times}_i^2}{\sigma_i^2},
           
           (F_+/\sigma) \cdot (F_{\\times}/\sigma) = \sum_i \\frac{{F_+}_i{F_\\times}_i}{\sigma_i^2},
           
           (d/\sigma) \cdot (F_+/\sigma) = \sum_i \\frac{d_i{F_+}_i}{\sigma_i^2},

           (d/\sigma) \cdot (F_\\times/\sigma) = \sum_i \\frac{d_i{F_\\times}_i}{\sigma_i^2},
        
        For non-GR signals, also involving the vector and scalar modes, there
        are similar products.
        """

        self.products = []  # list of products for each data set

        # loop over HeterodynedData and model functions
        for data, model in zip(self.data, self.models):
            self.products.append(dict())

            # initialise arrays in dictionary (Tp and Tc are the tensor plus
            # and cross, Vx and Vy are the vector "x" and "y", Sl and Sb are
            # the scalar longitudinal and breathing, d is the data)
            self.quadratures[-1][name] = np.zeros(data.num_chunks)
            knames = ['d', 'Tp', 'Tc', 'Vx', 'Vy', 'Sb', 'Sl']
            for i, a in enumerate(knames):
                for b in knames[::-1][:len(knames)-i]:
                    kname = a + 'dot' + b
                    if 'd' in [a, b] and a != b:
                        dtype = np.complex
                    else:
                        dtype = np.float

                        # set "whitened" versions of the antenna pattern product
                        # for SNR calculations, for when using the Students-t
                        # likelihood
                        self.products[-1][kname + 'White'] = np.zeros(data.num_chunks)

                    self.products[-1][kname] = np.zeros(data.num_chunks,
                                                        dtype=dtype)

            # loop over "chunks" into which each data set has been split
            for i, cpidx, cplen in zip(range(data.num_chunks),
                                       data.change_point_indices,
                                       data.chunk_lengths):
                # set the noise standard deviation for a Gaussian likelihood
                if self.likelihood == 'gaussian':
                    stdstrue = data.stds[cpidx:cpidx + cplen]
                else:
                    stdsunity = 1.

                stds = stdstrue if self.likelihood == 'gaussian' else stdsunity

                # get the interpolated response functions
                t0 = float(model.resp.t0)
                
                # interpolation times
                resptimes = np.arange(0., lal.DAYSID_SI,
                                      lal.DAYSID_SI / model.resp.ntimebins)
                inttimes = data.times[cpidx:cpidx + cplen] - t0
                
                # dictionary of chunk data and antenna responses
                rs = OrderedDict()
                rs['d'] = data.data[cpidx:cpidx + cplen]
                rs['Tp'] = np.interp(inttimes, model.resp.fplus.data,
                                     period=lal.DAYSID_SI)
                rs['Tc'] = np.interp(inttimes, model.resp.fcross.data,
                                     period=lal.DAYSID_SI)
                rs['Vx'] = np.interp(inttimes, model.resp.fx.data,
                                     period=lal.DAYSID_SI)
                rs['Vy'] = np.interp(inttimes, model.resp.fy.data,
                                     period=lal.DAYSID_SI)
                rs['Sb'] = np.interp(inttimes, model.resp.fb.data,
                                     period=lal.DAYSID_SI)
                rs['Sl'] = np.interp(inttimes, model.resp.fl.data,
                                     period=lal.DAYSID_SI)

                # get all required combinations of responses and data
                for j, a in enumerate(knames):
                    for b in knames[::-1][:len(knames)-j]:
                        kname = a + 'dot' + b
                        if kname == 'ddotd':
                            # complex conjugate dot product for data
                            self.products[-1][kname][i] = np.vdot(rs[a]/stds,
                                                                  rs[b]/stds).real
                        else:
                            self.products[-1][kname][i] = np.dot(rs[a]/stds, 
                                                                 rs[b]/stds)

                        if 'd' in [a, b] and a != b:
                            # get "whitened" versions for Students-t likelihood
                            self.products[-1][kname + 'White'] = np.dot(rs[a]/stdstrue, 
                                                                        rs[b]/stdstrue)

    def log_likelihood(self):
        """
        The log-likelihood function.
        """

        loglikelihood = 0.

        # loop over the data and models
        for data, model, prods, par in zip(self.data, self.models,
                                           self.products, self.basepars):
            # update parameters in the base par
            for pname, pval in self.parameters.items():
                if self._is_vector_param(pname.upper()):
                    par[pname.upper()] = self._parse_vector_param(par, pname.upper(), pval)
                else:
                    par[pname.upper()] = pval
            
            # calculate the model
            m = model.model(par, usephase=self.include_phase,
                            updateSSB=self.update_ssb,
                            updateBSB=self.include_binary)

            # calculate the likelihood ...

    def _is_vector_param(self, name):
        """
        Check if a parameter is a vector parameter.
        """
    
        # check for integers in name
        intvals = re.findall(r'\d+', name)
        if len(intvals) == 0:
            return False

        # strip out any underscores from name and remove trailing index
        noscores = re.strip('_')[:-len(intvals[-1])]

        if noscores in self.VECTOR_PARAMS:
            return True
        else:
            return False

    def _vector_param_name_index(self, name):
        """
        Get the vector parameter name (stripped of the position)
        """

        intvals = re.findall(r'\d+', name)

        # strip out any underscores from name and remove trailing index
        noscores = re.strip('_')[:-len(intvals[-1])]

        # glitch values start from 1 so subtract 1 from pos
        if name[:2] == 'GL':
            intvals[-1] -= 1

        return (noscores, intvals[-1])

    def _parse_vector_param(self, par, name, value):
        """
        Return a vector parameter.
        """

        vname, vpos = self._vector_param_name_index(name)
        vec = par[vname]

        return vec[pos] = value
