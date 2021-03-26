"""
Classes providing likelihood functions.
"""

import re

import bilby
import lal
import numpy as np
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy
from numba import jit, types
from numba.typed import Dict as numbadict

from .data import HeterodynedData, MultiHeterodynedData
from .signal import HeterodynedCWSimulator
from .utils import logfactorial


class TargetedPulsarLikelihood(bilby.core.likelihood.Likelihood):
    """
    A likelihood, based on the :class:`bilby.core.likelihood.Likelihood`, for
    use in source parameter estimation for continuous wave signals, with
    particular focus on emission from known pulsars. As a default, this class
    will assume a Student's t-likelihood function, as defined in Equation
    12 of [1]_. This likelihood function assumes that the noise process in each
    dataset can be broken down into multiple chunks of zero mean stationary
    Gaussian noise, each with unknown, and independent, values of the standard
    deviation (see :meth:`~cwinpy.data.HeterodynedData.bayesian_blocks` and
    Appendix B of [1]_). The Gaussian likelihood for each chunk is analytically
    marginalised over the standard deviation, using a Jeffreys prior, and the
    product of the likelihoods for each gives the overall likelihood function.
    A Gaussian likelihood function can also be used (Equation 13 of [1]_), if
    estimates of the noise standard deviation for each chunk are available.

    Parameters
    ----------
    data: str, HeterodynedData, MultiHeterodynedData
        A :class:`~cwinpy.data.HeterodynedData` or
        :class:`~cwinpy.data.MultiHeterodynedData` object containing all
        required data.
    priors: :class:`bilby.core.prior.PriorDict`
        A :class:`bilby.core.prior.PriorDict` containing the parameters that
        are to be sampled. This is required to be set before this class is
        initialised.
    likelihood: str
        A string setting which likelihood function to use. This can either be
        'studentst' (the default), or 'gaussian' ('roq' should be added in the
        future).
    numba: bool
        Boolean to set whether to use the `numba` JIT compiled version of the
        likelihood function.
    """

    # a set of parameters that define the "amplitude" of the signal (i.e.,
    # they do not effect the phase evolution of the signal)
    AMPLITUDE_PARAMS = [
        "H0",
        "PHI0",
        "PSI",
        "IOTA",
        "COSIOTA",
        "C21",
        "C22",
        "PHI21",
        "PHI22",
        "I21",
        "I31",
        "LAMBDA",
        "COSTHETA",
        "THETA",
        "Q22",
        "DIST",
    ]

    # amplitude parameters for the "source" model
    SOURCE_AMPLITUDE_PARAMETERS = [
        "H0",
        "I21",
        "I31",
        "LAMBDA",
        "COSTHETA",
        "THETA",
        "Q22",
        "DIST",
    ]

    # the set of potential non-GR "amplitude" parameters
    NONGR_AMPLITUDE_PARAM = [
        "PHI01TENSOR",
        "PHI02TENSOR",
        "PHI01VECTOR",
        "PHI02VECTOR",
        "PHI01SCALAR",
        "PHI02SCALAR",
        "PSI1TENSOR",
        "PSI2TENSOR",
        "PSI1VECTOR",
        "PSI2VECTOR",
        "PSI1SCALAR",
        "PSI2SCALAR",
        "H1PLUS",
        "H2PLUS",
        "H1CROSS",
        "H2CROSS",
        "H1VECTORX",
        "H2VECTORX",
        "H1VECTORY",
        "H2VECTORY",
        "H1SCALARB",
        "H2SCALARB",
        "H1SCALARL",
        "H2SCALARL",
    ]

    # the set of positional parameters
    POSITIONAL_PARAMETERS = ["RAJ", "DECJ", "RA", "DEC", "PMRA", "PMDEC", "POSEPOCH"]

    # the set of potential binary parameters
    BINARY_PARAMS = [
        "PB",
        "ECC",
        "EPS1",
        "EPS2",
        "T0",
        "TASC",
        "A1",
        "OM",
        "PB_2",
        "ECC_2",
        "T0_2",
        "A1_2",
        "OM_2",
        "PB_3",
        "ECC_3",
        "T0_3",
        "A1_3",
        "OM_3",
        "XPBDOT",
        "EPS1DOT",
        "EPS2DOT",
        "OMDOT",
        "GAMMA",
        "PBDOT",
        "XDOT",
        "EDOT",
        "SINI",
        "DR",
        "DTHETA",
        "A0",
        "B0",
        "MTOT",
        "M2",
        "FB",
    ]

    # the parameters that are held as vectors
    VECTOR_PARAMS = ["F", "GLEP", "GLPH", "GLF0", "GLF1", "GLF2", "GLF0D", "GLTD", "FB"]

    def __init__(
        self, data, priors, par=None, det=None, likelihood="studentst", numba=False
    ):

        super().__init__(dict())  # initialise likelihood class

        # set the data
        if isinstance(data, HeterodynedData):
            # convert HeterodynedData to MultiHeterodynedData object
            if data.detector is None:
                raise ValueError("'data' must contain a named detector.")

            if data.par is None:
                raise ValueError("'data' must contain a heterodyne parameter file.")

            self.data = MultiHeterodynedData(data)
        elif isinstance(data, MultiHeterodynedData):
            if None in data.pars:
                raise ValueError(
                    "A data set does not have an associated "
                    "heterodyned parameter set."
                )

            self.data = data
        else:
            raise TypeError(
                "'data' must be a HeterodynedData or MultiHeterodynedData object."
            )

        if not isinstance(priors, bilby.core.prior.PriorDict):
            raise TypeError("Prior must be a bilby PriorDict")
        else:
            self.priors = priors

        # set the likelihood function
        self.likelihood = likelihood
        self._noise_log_likelihood = -np.inf  # initialise noise log likelihood
        self.numba = numba

        # check if prior includes (non-DeltaFunction) non-amplitude parameters,
        # i.e., will the phase evolution have to be included in the search,
        # or binary parameters
        self.include_phase = False
        self.include_binary = False
        self.include_glitch = False
        self.update_ssb = False
        for key in self.priors:
            if (
                key.upper()
                not in self.AMPLITUDE_PARAMS
                + self.BINARY_PARAMS
                + self.POSITIONAL_PARAMETERS
            ) and not self._is_vector_param(key.upper()):
                raise ValueError("Unknown parameter '{}' being used!".format(key))

            if key.upper() not in self.AMPLITUDE_PARAMS:
                if self.priors[key].is_fixed:
                    # check if it's the same value as the par files
                    for het in self.data:
                        if self._is_vector_param(key.upper()):
                            name, idx = self._vector_param_name_index(key.upper())
                            checkval = het.par[name][idx]
                        else:
                            checkval = het.par[key.upper()]

                        if checkval != self.priors[key].peak:
                            self.include_phase = True
                            if key.upper() in self.BINARY_PARAMS:
                                self.include_binary = True
                            elif key.upper() in self.POSITIONAL_PARAMETERS:
                                self.update_ssb = True
                            elif len(key) > 2:
                                if key.upper()[0:2] == "GL":
                                    self.include_glitch = True
                            break
                else:
                    self.include_phase = True
                    if key.upper() in self.BINARY_PARAMS:
                        self.include_binary = True
                    elif key.upper() in self.POSITIONAL_PARAMETERS:
                        self.update_ssb = True
                    elif len(key) > 2:
                        if key.upper()[0:2] == "GL":
                            self.include_glitch = True

        # check if any non-GR "amplitude" parameters are set
        self.nonGR = False
        for key in self.priors:
            if key.upper() in self.NONGR_AMPLITUDE_PARAM:
                self.nonGR = True

        # set up signal model classes
        self.models = []
        self.basepars = []
        for het in self.data:
            self.models.append(
                HeterodynedCWSimulator(
                    het.par,
                    het.detector,
                    het.times,
                    earth_ephem=het.ephemearth,
                    sun_ephem=het.ephemsun,
                )
            )
            # copy of heterodyned parameters
            newpar = PulsarParametersPy()
            for item in het.par.items():
                newpar[item[0]] = item[1]
            self.basepars.append(newpar)

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
        if likelihood.lower() not in [
            "studentst",
            "students-t",
            "studentt",
            "gaussian",
            "normal",
        ]:
            raise ValueError("Likelihood must be 'studentst' or 'gaussian'.")

        if likelihood.lower() in ["studentst", "students-t", "studentt"]:
            self.__likelihood = "studentst"
        else:
            self.__likelihood = "gaussian"

    def dot_products(self):
        """
        Calculate the (noise-weighted) dot products of the data and the
        antenna pattern functions (see Appendix C of [1]_). E.g., for the data this is
        the real value

        .. math::

           (d/\\sigma) \\cdot (d^*/\\sigma) = \\sum_i \\frac{d_id_i^*}{\\sigma_i^2}
           \\equiv \\sum_i \\frac{\\Re{(d)}^2 + \\Im{(d)}^2}{\\sigma_i^2}.

        For the antenna patterns, for example :math:`F_+` and
        :math:`F_{\\times}`, we would have

        .. math::

           (F_+/\\sigma) \\cdot (F_+/\\sigma) = \\sum_i \\frac{{F_+}_i^2}{\\sigma_i^2},

           (F_\\times/\\sigma) \\cdot (F_\\times/\\sigma) = \\sum_i \\frac{{F_\\times}_i^2}{\\sigma_i^2},

           (F_+/\\sigma) \\cdot (F_{\\times}/\\sigma) = \\sum_i \\frac{{F_+}_i{F_\\times}_i}{\\sigma_i^2},

           (d/\\sigma) \\cdot (F_+/\\sigma) = \\sum_i \\frac{d_i{F_+}_i}{\\sigma_i^2},

           (d/\\sigma) \\cdot (F_\\times/\\sigma) = \\sum_i \\frac{d_i{F_\\times}_i}{\\sigma_i^2},

        For non-GR signals, also involving the vector and scalar modes, there
        are similar products.
        """

        self.products = []  # list of products for each data set

        # loop over HeterodynedData and model functions
        for data, model in zip(self.data, self.models):
            if not self.numba:
                self.products.append(dict())
            else:
                dreal = numbadict.empty(
                    key_type=types.unicode_type, value_type=types.float64[:]
                )
                dcomplex = numbadict.empty(
                    key_type=types.unicode_type, value_type=types.complex128[:]
                )
                self.products.append([dreal, dcomplex])

            # initialise arrays in dictionary (Tp and Tc are the tensor plus
            # and cross, Vx and Vy are the vector "x" and "y", Sl and Sb are
            # the scalar longitudinal and breathing, d is the data)
            knames = ["d", "Tp", "Tc", "Vx", "Vy", "Sb", "Sl"]
            for i, a in enumerate(knames):
                for b in knames[::-1][: len(knames) - i]:
                    kname = a + "dot" + b
                    if "d" in [a, b] and a != b:
                        dtype = "complex128"
                    else:
                        dtype = "f8"

                        # set "whitened" versions of the antenna pattern product
                        # for SNR calculations, for when using the Students-t
                        # likelihood
                        if "d" not in [a, b]:
                            if not self.numba:
                                self.products[-1][kname + "White"] = np.zeros(
                                    data.num_chunks, dtype=dtype
                                )
                            else:
                                self.products[-1][0][kname + "White"] = np.zeros(
                                    data.num_chunks, dtype=dtype
                                )

                    if not self.numba:
                        self.products[-1][kname] = np.zeros(
                            data.num_chunks, dtype=dtype
                        )
                    else:
                        if dtype == "f8":
                            self.products[-1][0][kname] = np.zeros(
                                data.num_chunks, dtype=dtype
                            )
                        else:
                            self.products[-1][1][kname] = np.zeros(
                                data.num_chunks, dtype=dtype
                            )

            # loop over "chunks" into which each data set has been split
            for i, cpidx, cplen in zip(
                range(data.num_chunks), data.change_point_indices, data.chunk_lengths
            ):
                # set the noise standard deviation for a Gaussian likelihood
                stdstrue = data.stds[cpidx : cpidx + cplen]
                stds = stdstrue if self.likelihood == "gaussian" else 1.0

                # get the interpolated response functions
                t0 = float(model.resp.t0)

                # interpolation times
                ftimes = np.arange(
                    0.0, lal.DAYSID_SI, lal.DAYSID_SI / model.resp.ntimebins
                )
                inttimes = data.times[cpidx : cpidx + cplen].value - t0

                # dictionary of chunk data and antenna responses
                rs = dict()
                rs["d"] = data.data[cpidx : cpidx + cplen]
                rs["Tp"] = np.interp(
                    inttimes, ftimes, model.resp.fplus.data, period=lal.DAYSID_SI
                )
                rs["Tc"] = np.interp(
                    inttimes, ftimes, model.resp.fcross.data, period=lal.DAYSID_SI
                )
                rs["Vx"] = np.interp(
                    inttimes, ftimes, model.resp.fx.data, period=lal.DAYSID_SI
                )
                rs["Vy"] = np.interp(
                    inttimes, ftimes, model.resp.fy.data, period=lal.DAYSID_SI
                )
                rs["Sb"] = np.interp(
                    inttimes, ftimes, model.resp.fb.data, period=lal.DAYSID_SI
                )
                rs["Sl"] = np.interp(
                    inttimes, ftimes, model.resp.fl.data, period=lal.DAYSID_SI
                )

                # get all required combinations of responses and data
                for j, a in enumerate(knames):
                    for b in knames[::-1][: len(knames) - j]:
                        kname = a + "dot" + b
                        if not self.numba:
                            if kname == "ddotd":
                                # complex conjugate dot product for data
                                self.products[-1][kname][i] = np.vdot(
                                    rs[a] / stds, rs[b] / stds
                                ).real
                            else:
                                self.products[-1][kname][i] = np.dot(
                                    rs[a] / stds, rs[b] / stds
                                )

                            if "d" not in [a, b]:
                                # get "whitened" versions for Students-t likelihood
                                self.products[-1][kname + "White"][i] = np.dot(
                                    rs[a] / stdstrue, rs[b] / stdstrue
                                )
                        else:
                            if kname == "ddotd":
                                # complex conjugate dot product for data
                                self.products[-1][0][kname][i] = np.vdot(
                                    rs[a] / stds, rs[b] / stds
                                ).real
                            else:
                                if "d" in [a, b]:
                                    self.products[-1][1][kname][i] = np.dot(
                                        rs[a] / stds, rs[b] / stds
                                    )
                                else:
                                    self.products[-1][0][kname][i] = np.dot(
                                        rs[a] / stds, rs[b] / stds
                                    )

                            if "d" not in [a, b]:
                                # get "whitened" versions for Students-t likelihood
                                self.products[-1][0][kname + "White"][i] = np.dot(
                                    rs[a] / stdstrue, rs[b] / stdstrue
                                )

    def log_likelihood(self):
        """
        The log-likelihood function.
        """

        loglikelihood = 0.0  # the log likelihood value

        # loop over the data and models
        for data, model, prods, par in zip(
            self.data, self.models, self.products, self.basepars
        ):

            # update parameters in the base par
            for pname, pval in self.parameters.items():
                if self._is_vector_param(pname.upper()):
                    name = self._vector_param_name_index(pname.upper())[0]
                    par[name] = self._parse_vector_param(par, pname.upper(), pval)
                else:
                    # make sure values are floats
                    par[pname.upper()] = float(pval)

                    if pname.upper() in self.SOURCE_AMPLITUDE_PARAMETERS:
                        # reset waveform parameters (otherwise these can
                        # potentially be given by the previous value if the
                        # new value is zero). NOTE: in the future it might be
                        # better to pass a copy of the PulsarParameters
                        # structure to the model function, so that this is not
                        # problem
                        par["C21"] = 0.0
                        par["C22"] = 0.0
                        par["PHI22"] = 0.0
                        par["PHI21"] = 0.0

            # calculate the model
            m = model.model(
                par,
                usephase=self.include_phase,
                updateSSB=self.update_ssb,
                updateBSB=self.include_binary,
                updateglphase=self.include_glitch,
                freqfactor=data.freq_factor,
            )

            # calculate the likelihood
            if self.numba:
                loglikelihood += self._log_likelihood_numba(
                    data.data,
                    data.num_chunks,
                    np.asarray(data.change_point_indices),
                    np.asarray(data.chunk_lengths),
                    m,
                    self.likelihood,
                    prods[0],
                    prods[1],
                    data.stds,
                    self.nonGR,
                    self.include_phase,
                )
            else:
                # loop over stationary data chunks
                for i, cpidx, cplen in zip(
                    range(data.num_chunks),
                    data.change_point_indices,
                    data.chunk_lengths,
                ):
                    # likelihood without pre-summed products
                    if self.likelihood == "gaussian":
                        stds = data.stds[cpidx : cpidx + cplen]
                    else:
                        stds = 1.0

                    if self.include_phase:
                        # data and model for chunk
                        dd = data.data[cpidx : cpidx + cplen] / stds
                        mm = m[cpidx : cpidx + cplen] / stds

                        summodel = np.vdot(mm, mm).real
                        sumdatamodel = np.vdot(dd, mm).real
                    else:
                        # likelihood with pre-summed products
                        mp = m[0]  # tensor plus model component
                        mc = m[1]  # tensor cross model component

                        summodel = (
                            prods["TpdotTp"][i] * (mp.real ** 2 + mp.imag ** 2)
                            + prods["TcdotTc"][i] * (mc.real ** 2 + mc.imag ** 2)
                            + 2.0
                            * prods["TpdotTc"][i]
                            * (mp.real * mc.real + mp.imag * mc.imag)
                        )

                        sumdatamodel = (
                            prods["ddotTp"][i].real * mp.real
                            + prods["ddotTp"][i].imag * mp.imag
                            + prods["ddotTc"][i].real * mc.real
                            + prods["ddotTc"][i].imag * mc.imag
                        )

                        if self.nonGR:
                            # non-GR amplitudes
                            mx = m[2]
                            my = m[3]
                            mb = m[4]
                            ml = m[5]

                            summodel += (
                                prods["VxdotVx"][i] * (mx.real ** 2 + mx.imag ** 2)
                                + prods["VydotVy"][i] * (my.real ** 2 + my.imag ** 2)
                                + prods["SbdotSb"][i] * (mb.real ** 2 + mb.imag ** 2)
                                + prods["SldotSl"][i] * (ml.real ** 2 + ml.imag ** 2)
                                + 2.0
                                * (
                                    prods["TpdotVx"][i]
                                    * (mp.real * mx.real + mp.imag * mx.imag)
                                    + prods["TpdotVy"][i]
                                    * (mp.real * my.real + mp.imag * my.imag)
                                    + prods["TpdotSb"][i]
                                    * (mp.real * mb.real + mp.imag * mb.imag)
                                    + prods["TpdotSl"][i]
                                    * (mp.real * ml.real + mp.imag * ml.imag)
                                    + prods["TcdotVx"][i]
                                    * (mc.real * mx.real + mc.imag * mx.imag)
                                    + prods["TcdotVy"][i]
                                    * (mc.real * my.real + mc.imag * my.imag)
                                    + prods["TcdotSb"][i]
                                    * (mc.real * mb.real + mc.imag * mb.imag)
                                    + prods["TcdotSl"][i]
                                    * (mc.real * ml.real + mc.imag * ml.imag)
                                    + prods["VxdotVy"][i] * (mx.real * my.real)
                                    + (mx.imag * my.imag)
                                    + prods["VxdotSb"][i]
                                    * (mx.real * mb.real + mx.imag * mb.imag)
                                    + prods["VxdotSl"][i]
                                    * (mx.real * ml.real + mx.imag * ml.imag)
                                    + prods["VydotSb"][i]
                                    * (my.real * mb.real + my.imag * mb.imag)
                                    + prods["VydotSl"][i]
                                    * (my.real * ml.real + my.imag * ml.imag)
                                    + prods["SbdotSl"][i]
                                    * (mb.real * ml.real + mb.imag * ml.imag)
                                )
                            )

                            sumdatamodel += (
                                prods["ddotVx"][i].real * mx.real
                                + prods["ddotVx"][i].imag * mx.imag
                                + prods["ddotVy"][i].real * my.real
                                + prods["ddotVy"][i].imag * my.imag
                                + prods["ddotSb"][i].real * mb.real
                                + prods["ddotSb"][i].imag * mb.imag
                                + prods["ddotSl"][i].real * ml.real
                                + prods["ddotSl"][i].imag * ml.imag
                            )

                    # compute "Chi-squared"
                    chisquare = prods["ddotd"][i] - 2.0 * sumdatamodel + summodel

                    if self.likelihood == "gaussian":
                        loglikelihood += 0.5 * chisquare
                        # normalisation
                        loglikelihood -= np.log(lal.TWOPI * stds[0] ** 2)
                    else:
                        loglikelihood += (
                            logfactorial(cplen - 1)
                            - lal.LN2
                            - cplen * lal.LNPI
                            - cplen * np.log(chisquare)
                        )

        return loglikelihood

    @staticmethod
    @jit(nopython=True)
    def _log_likelihood_numba(
        data,
        datanchunks,
        datacps,
        datacls,
        model,
        likelihood="studentst",
        productsreal=None,
        productscomp=None,
        datastds=None,
        nonGR=False,
        includephase=False,
    ):
        """
        This is a version of the standard inner loop of
        :meth:`cwinpy.TargetedPulsarLikelihood.log_likelihood` that used the
        `numba <https://numba.pydata.org/>`_ JIT package to provide some
        speed-up.

        Parameters
        ----------
        data: array_like
            A complex :class:`numpy.ndarray` containing data from a single
            detector.
        datanchunks: int
            The number of chunks into which the data has been split.
        datacps: array_like
            A :class:`numpy.ndarray` containing the change point indices for
            each chunk.
        datacls: array_like
            A :class:`numpy.ndarray` containing lengths of each of the chunks.
        model: array_like
            A complex :class:`numpy.ndarray` containing the signal model.
        likelihood: str
            A string stating whether to use the Student's-t or Gaussian
            likelihood.
        productsreal: dict
            A numba :class:`~numba.typed.Dict` containing
            :class:`numpy.ndarray`'s for model component dot products.
        productscomp: dict
            A numba :class:`~numba.typed.Dict` containing
            :class:`numpy.ndarray`'s for data and model component dot products.
        datastds: array_like
            A :class:`numpy.ndarray` containing data standard deviations.
        nonGR: bool
            A flag to set whether using a non-GR signal.
        includephase: bool
            A flag specifying whether the model has a varying phase evolution.

        Returns
        -------
        loglikelihood: float
            The log-likelihood function calculated for that data set.
        """

        loglikelihood = 0.0

        # calculate the likelihood
        for i, cpidx, cplen in zip(range(datanchunks), datacps, datacls):
            # loop over stationary data chunks

            # likelihood without pre-summed products
            if likelihood == "gaussian":
                stds = datastds[cpidx : cpidx + cplen]
            else:
                stds = np.ones(cplen)

            if includephase:
                # data and model for chunk
                dd = data[cpidx : cpidx + cplen] / stds
                mm = model[cpidx : cpidx + cplen] / stds

                summodel = np.vdot(mm, mm).real
                sumdatamodel = np.vdot(dd, mm).real
            else:
                # likelihood with pre-summed products
                mp = model[0]  # tensor plus model component
                mc = model[1]  # tensor cross model component

                summodel = (
                    productsreal["TpdotTp"][i] * (mp.real ** 2 + mp.imag ** 2)
                    + productsreal["TcdotTc"][i] * (mc.real ** 2 + mc.imag ** 2)
                    + 2.0
                    * productsreal["TpdotTc"][i]
                    * (mp.real * mc.real + mp.imag * mc.imag)
                )

                sumdatamodel = (
                    productscomp["ddotTp"][i].real * mp.real
                    + productscomp["ddotTp"][i].imag * mp.imag
                    + productscomp["ddotTc"][i].real * mc.real
                    + productscomp["ddotTc"][i].imag * mc.imag
                )

                if nonGR:
                    # non-GR amplitudes
                    mx = model[2]
                    my = model[3]
                    mb = model[4]
                    ml = model[5]

                    summodel += (
                        productsreal["VxdotVx"][i] * (mx.real ** 2 + mx.imag ** 2)
                        + productsreal["VydotVy"][i] * (my.real ** 2 + my.imag ** 2)
                        + productsreal["SbdotSb"][i] * (mb.real ** 2 + mb.imag ** 2)
                        + productsreal["SldotSl"][i] * (ml.real ** 2 + ml.imag ** 2)
                        + 2.0
                        * (
                            productsreal["TpdotVx"][i]
                            * (mp.real * mx.real + mp.imag * mx.imag)
                            + productsreal["TpdotVy"][i]
                            * (mp.real * my.real + mp.imag * my.imag)
                            + productsreal["TpdotSb"][i]
                            * (mp.real * mb.real + mp.imag * mb.imag)
                            + productsreal["TpdotSl"][i]
                            * (mp.real * ml.real + mp.imag * ml.imag)
                            + productsreal["TcdotVx"][i]
                            * (mc.real * mx.real + mc.imag * mx.imag)
                            + productsreal["TcdotVy"][i]
                            * (mc.real * my.real + mc.imag * my.imag)
                            + productsreal["TcdotSb"][i]
                            * (mc.real * mb.real + mc.imag * mb.imag)
                            + productsreal["TcdotSl"][i]
                            * (mc.real * ml.real + mc.imag * ml.imag)
                            + productsreal["VxdotVy"][i] * (mx.real * my.real)
                            + (mx.imag * my.imag)
                            + productsreal["VxdotSb"][i]
                            * (mx.real * mb.real + mx.imag * mb.imag)
                            + productsreal["VxdotSl"][i]
                            * (mx.real * ml.real + mx.imag * ml.imag)
                            + productsreal["VydotSb"][i]
                            * (my.real * mb.real + my.imag * mb.imag)
                            + productsreal["VydotSl"][i]
                            * (my.real * ml.real + my.imag * ml.imag)
                            + productsreal["SbdotSl"][i]
                            * (mb.real * ml.real + mb.imag * ml.imag)
                        )
                    )

                    sumdatamodel += (
                        productscomp["ddotVx"][i].real * mx.real
                        + productscomp["ddotVx"][i].imag * mx.imag
                        + productscomp["ddotVy"][i].real * my.real
                        + productscomp["ddotVy"][i].imag * my.imag
                        + productscomp["ddotSb"][i].real * mb.real
                        + productscomp["ddotSb"][i].imag * mb.imag
                        + productscomp["ddotSl"][i].real * ml.real
                        + productscomp["ddotSl"][i].imag * ml.imag
                    )

            # compute "Chi-squared"
            chisquare = productsreal["ddotd"][i] - 2.0 * sumdatamodel + summodel

            if likelihood == "gaussian":
                loglikelihood += 0.5 * chisquare
                # normalisation
                loglikelihood -= np.log(lal.TWOPI * stds[0] ** 2)
            else:
                loglikelihood += (
                    logfactorial(cplen - 1)
                    - lal.LN2
                    - cplen * lal.LNPI
                    - cplen * np.log(chisquare)
                )

        return loglikelihood

    def noise_log_likelihood(self):
        """
        The log-likelihood for the data being consistent with the noise model,
        i.e., when the signal is zero. See Equations 14 and 15 of [1]_.

        Returns
        -------
        float:
            The noise-only log-likelihood
        """

        if np.isfinite(self._noise_log_likelihood):
            # return cached version of the noise log likelihood
            return self._noise_log_likelihood

        self._noise_log_likelihood = 0.0  # the log likelihood value

        # loop over the data and models
        for data, prods in zip(self.data, self.products):
            # calculate the likelihood
            for i, cpidx, cplen in zip(
                range(data.num_chunks), data.change_point_indices, data.chunk_lengths
            ):
                # loop over stationary data chunks
                if self.numba:
                    ddotd = prods[0]["ddotd"][i]
                else:
                    ddotd = prods["ddotd"][i]
                if self.likelihood == "gaussian":
                    self._noise_log_likelihood += 0.5 * ddotd
                    # normalisation
                    self._noise_log_likelihood -= np.log(lal.TWOPI * data.vars[cpidx])
                else:
                    self._noise_log_likelihood += (
                        logfactorial(cplen - 1)
                        - lal.LN2
                        - cplen * lal.LNPI
                        - cplen * np.log(ddotd)
                    )

        return self._noise_log_likelihood

    def _is_vector_param(self, name):
        """
        Check if a parameter is a vector parameter.
        """

        # check for integers in name
        intvals = re.findall(r"\d+", name)
        if len(intvals) == 0:
            return False

        # strip out any underscores from name and remove trailing index
        noscores = re.sub("_", "", name)[: -len(intvals[-1])]

        if noscores in self.VECTOR_PARAMS:
            return True
        else:
            return False

    def _vector_param_name_index(self, name):
        """
        Get the vector parameter name (stripped of the position)
        """

        intvals = re.findall(r"\d+", name)

        # strip out any underscores from name and remove trailing index
        noscores = re.sub("_", "", name)[: -len(intvals[-1])]

        # glitch values start from 1 so subtract 1 from pos
        if name[:2] == "GL":
            intvals[-1] -= 1

        return (noscores, intvals[-1])

    def _parse_vector_param(self, par, name, value):
        """
        Set a vector parameter with the given single value at the place
        specified in the `name`.
        """

        vname, vpos = self._vector_param_name_index(name)
        vec = par[vname]
        vec[vpos] = value

        return vec
