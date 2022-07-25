from copy import deepcopy

import lal
import numpy as np
from arby import reduced_basis
from bilby.core.prior import PriorDict
from numba import jit

from ..signal import HeterodynedCWSimulator
from ..utils import logfactorial


class GenerateROQ:
    def __init__(self, data, x, priors, **kwargs):
        """
        Generate the Reduced Order Quadrature for the parameters in the prior.

        Parameters
        ----------
        data: array_like
            The data set (real or complex) for which the ROQ will be produced.
        x: array_like
            The values at which the data is defined (e.g., a set of times).
        priors: PriorDict
            A :class:`~bilby.core.prior.PriorDict` containing the parameters
            and prior distributions, which will be used to generate the model
            reduced basis set.
        ntraining: int
            The number of training example models to be generated from which to
            produce the reduced basis set. Defaults to 5000.
        sigma: array_like, float
            The noise standard deviation for the data, which is assumed to be
            stationary over the data. If not given, this will be assume to be
            1.0 for all data points.
        store_training_data: bool
            If set to True the class will store the array of models used to
            generate the reduced basis. Note that if using this class multiple
            times, storing the data could become memory intensive. Defaults to
            False.
        greedy_tol: float
            The tolerance to use when generating the reduced basis. Defaults to
            1e-12.
        model: func
            The model function used to generate the training examples. By
            default this assumes the
            :class:`cwinpy.signal.HeterodynedCWSimulator` as the model (see
            Other Parameters below), but it can be an arbitrary Python function
            provided the function has an initial positional argument that takes
            in the ``x`` values, followed by keyword arguments for the other
            function parameters as defined in the priors.

        Other Parameters
        ----------------
        par: PulsarParameters
            A :class:`cwinpy.parfile.PulsarParameters` or parameter file
            containing the parameters used for heterodyning the data.
        det: str
            The name of the detector for which the signal will be modelled.
        usetempo2: bool
            Set to True if the signal model phase evolution should be
            calculated using Tempo2. Defaults to False.
        """

        self.kwargs = kwargs
        self.x = x
        self.data = data
        self.priors = priors

        # the number of training examples to use
        self.ntraining = self.kwargs.get("ntraining", 5000)

        # generate training data
        ts = self.generate_training_set()

        if self.kwargs.get("store_training_data", False):
            # store training data if needed
            self.training_data = ts

        # generate the reduced basis
        self.generate_reduced_basis(ts)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if not isinstance(x, (np.ndarray, list)):
            raise TypeError("x must be an array")

        xarr = np.array(x)

        if len(xarr.shape) != 1:
            raise ValueError("x must be a 1d array")

        if xarr.dtype not in [float, int]:
            raise TypeError("x must be a float or integer")

        self._x = xarr

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, (np.ndarray, list)):
            raise TypeError("Data must be an array")

        dataarr = np.array(data)

        if len(dataarr.shape) != 1:
            raise ValueError("Data must be a 1d array")

        if dataarr.dtype not in [complex, float]:
            raise TypeError("Data must be a complex or float type")

        if len(dataarr) != len(self.x):
            raise ValueError("Data must be the same length as x")

        self._data = dataarr

        # likelihood term for squared data
        self._sigma = self.kwargs.get("sigma", 1.0)

        if not isinstance(self._sigma, (int, float)):
            raise TypeError("sigma must be a number")
        else:
            if self._sigma <= 0.0:
                raise ValueError("sigma must be a positive number")

        self.is_complex = True if self._data.dtype == complex else False

        self._K = np.vdot(self._data, self._data).real

    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, priors):
        if not isinstance(priors, PriorDict):
            raise TypeError("Prior must be a bilby PriorDict")

        self._priors = priors

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model == "HeterodynedCWSimulator":
            if "par" not in self.kwargs:
                raise ValueError("'par', parameter file must be in keyword arguments")

            if "det" not in self.kwargs:
                raise ValueError("'det', detector name must be in keyword arguments")

            par = self.kwargs.get("par")
            det = self.kwargs.get("det")
            model = HeterodynedCWSimulator(
                par, det, times=self.x, usetempo2=self.kwargs.get("usetempo2", False)
            )

            self._initial_par = deepcopy(par)
            self._par = deepcopy(par)

            self._model = model.model
        else:
            if not callable(model):
                raise TypeError("model must be a callable function")

            self._model = model

    def _model_at_nodes(self):
        if hasattr(self, "_initial_par"):
            det = self.kwargs.get("det")
            dt = 60 if len(self._x_nodes) == 1 else None
            model = HeterodynedCWSimulator(
                self._initial_par,
                det,
                times=self._x_nodes,
                usetempo2=self.kwargs.get("usetempo2", False),
                dt=dt,
            )
            self._model_short = model.model
        else:
            self._model_short = self.model

    def generate_training_set(self):
        """
        Generate ``n`` model instances, using values drawn from the prior.
        """

        training_set_model = []

        self.model = self.kwargs.get("model", "HeterodynedCWSimulator")

        # draw values from prior
        samples = self.priors.sample(self.ntraining)

        minrange = np.inf
        maxrange = -np.inf

        for i in range(self.ntraining):
            # update par file
            if hasattr(self, "_par"):
                for prior in samples:
                    self._par[prior] = samples[prior][i]

            if hasattr(self, "_par"):
                m = self.model(
                    deepcopy(self._par),
                    outputampcoeffs=False,
                    updateSSB=True,
                    updateBSB=True,
                    updateglphase=True,
                    freqfactor=self.kwargs.get("freq_factor", 2),
                )
            else:
                m = self.model(
                    self.x, **{prior: samples[prior][i] for prior in samples}
                )

            # get scaling (arby thinks the training set is null if the values
            # are too small)
            if np.min(np.abs(m)) < minrange:
                minrange = np.min(np.abs(m))
            if np.max(np.abs(m)) > maxrange:
                maxrange = np.max(np.abs(m))

            training_set_model.append(m)

        # rescale
        self._scaling = maxrange - minrange
        training_set_model = np.array([m / self._scaling for m in training_set_model])

        return training_set_model

    def generate_reduced_basis(self, training):
        """
        Generate the reduced basis for the training set.
        """

        # reduced basis for the model training set
        if self.is_complex:
            rb_model_real = reduced_basis(
                training_set=np.array(training).real,
                physical_points=self.x,
                greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
                normalize=True,
            )
            rb_model_imag = reduced_basis(
                training_set=np.array(training).imag,
                physical_points=self.x,
                greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
                normalize=True,
            )

            self._Dvec_real = np.einsum(
                "i,ji->j", self.data.real, rb_model_real.basis.data
            )
            self._Dvec_imag = np.einsum(
                "i,ji->j", self.data.imag, rb_model_imag.basis.data
            )

            nodes_real = np.array(sorted(rb_model_real.basis.eim_.nodes))
            nodes_imag = np.array(sorted(rb_model_imag.basis.eim_.nodes))

            self._node_indices = np.union1d(nodes_real, nodes_imag)
            self._Bmat_real = np.array(
                [row[nodes_real] for row in rb_model_real.basis.data]
            )
            self._Bmat_imag = np.array(
                [row[nodes_imag] for row in rb_model_imag.basis.data]
            )
        else:
            rb_model = reduced_basis(
                training_set=np.array(training),
                physical_points=self.x,
                greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
                normalize=True,
            )

            self._Dvec = np.einsum("i,ji->j", self.data, rb_model.basis.data)

            nodes = np.array(sorted(rb_model.basis.eim_.nodes))
            self._node_indices = nodes
            self._Bmat = np.array([row[nodes] for row in rb_model.basis.data])

        # reduced basis for the squared model training set
        rb_model_squared = reduced_basis(
            training_set=np.array((training * np.conj(training)).real),
            physical_points=self.x,
            greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
            normalize=True,
        )

        self._Bvec = np.einsum("ji->j", rb_model_squared.basis.data)
        nodes2 = np.array(sorted(rb_model_squared.basis.eim_.nodes))
        self._B2mat = np.array([row[nodes2] for row in rb_model_squared.basis.data])

        # setup model function at the interpolation nodes
        self._node_indices = np.union1d(self._node_indices, nodes2)

        self._x_nodes = self.x[self._node_indices]

        if self.is_complex:
            self.nbases_real = len(nodes_real)
            self.nbases_imag = len(nodes_imag)
            self._x_node_indices_real = np.searchsorted(self._node_indices, nodes_real)
            self._x_node_indices_imag = np.searchsorted(self._node_indices, nodes_imag)
        else:
            self.nbases = len(nodes)
            self._x_node_indices = np.searchsorted(self._node_indices, nodes)

        self.nbases2 = len(nodes2)
        self._x2_node_indices = np.searchsorted(self._node_indices, nodes2)

        self._model_at_nodes()

    def log_likelihood(self, **kwargs):
        """
        Calculate the log-likelihood using the reduced order basis.

        Parameters
        ----------
        likelihood: str
            When calculating the ROQ likelihood, set whether it is a Gaussian
            likelihood ``"gaussian"`` or Student's t-likelihood ``"studentst"``.
            As with the :class:`cwinpy.pe.likelihood.TargetedPulsarLikelihood`,
            the default is Student's t.
        numba: bool
            This defaults to True in which case the likelihood defined using
            numba will be used.
        """

        likelihood = kwargs.get("likelihood", "studentst").lower()

        if hasattr(self, "_par"):
            if not hasattr(self, "__model_kwargs"):
                self.__model_kwargs = dict(
                    outputampcoeffs=False,
                    updateSSB=True,
                    updateBSB=True,
                    updateglphase=True,
                    freqfactor=self.kwargs.get("freq_factor", 2),
                )

            if "par" not in kwargs:
                raise KeyError("Arguments must contain a PulsarParameter object")
            pos = [deepcopy(kwargs.get("par"))]
        else:
            self.__model_kwargs = {prior: kwargs.get(prior) for prior in self.priors}
            pos = [self._x_nodes]

        # generate the model at the interpolation nodes
        model = self._model_short(*pos, **self.__model_kwargs)

        if kwargs.get("numba", True):
            if self.is_complex:
                return self._complex_log_likelihood_numba(
                    self._K,
                    self._Dvec_real,
                    self._Dvec_imag,
                    self._Bvec,
                    self._Bmat_real,
                    self._Bmat_imag,
                    self._B2mat,
                    model[self._x_node_indices_real].real,
                    model[self._x_node_indices_imag].imag,
                    model[self._x2_node_indices],
                    self._sigma,
                    len(self.data),
                    likelihood=likelihood,
                )
            else:
                return self._log_likelihood_numba(
                    self._K,
                    self._Dvec,
                    self._Bvec,
                    self._Bmat,
                    self._B2mat,
                    model[self._x_node_indices],
                    model[self._x2_node_indices],
                    self._sigma,
                    len(self.data),
                    likelihood=likelihood,
                )
        else:
            # square model
            model2 = (
                model[self._x2_node_indices] * np.conj(model[self._x2_node_indices])
            ).real
            if self.is_complex:
                dm = (
                    np.vdot(
                        np.linalg.solve(
                            self._Bmat_real.T, model[self._x_node_indices_real].real
                        ),
                        self._Dvec_real,
                    ).real
                    + np.vdot(
                        np.linalg.solve(
                            self._Bmat_imag.T, model[self._x_node_indices_imag].imag
                        ),
                        self._Dvec_imag,
                    ).real
                )
            else:
                dm = np.vdot(
                    np.linalg.solve(self._Bmat.T, model[self._x_node_indices]),
                    self._Dvec,
                ).real

            mm = np.vdot(np.linalg.solve(self._B2mat.T, model2), self._Bvec).real
            chisq = self._K + mm - 2 * dm

            if likelihood == "studentst":
                cplen = len(self.data)
                return (
                    logfactorial(cplen - 1)
                    - lal.LN2
                    - cplen * lal.LNPI
                    - cplen * np.log(chisq)
                )
            else:
                return -(0.5 / self._sigma**2) * (chisq)

    @staticmethod
    @jit(nopython=True)
    def _log_likelihood_numba(
        K,
        Dvec,
        Bvec,
        Bmat,
        B2mat,
        model,
        model2,
        sigma,
        cplen,
        likelihood="studentst",
    ):
        # square model
        model2 = (model2 * np.conj(model2)).real

        dm = np.vdot(np.linalg.solve(Bmat.T, model), Dvec).real
        mm = np.vdot(np.linalg.solve(B2mat.T, model2), Bvec).real
        chisq = K + mm - 2 * dm

        # print(dm, mm, K)

        if likelihood == "studentst":
            return (
                logfactorial(cplen - 1)
                - lal.LN2
                - cplen * lal.LNPI
                - cplen * np.log(chisq)
            )
        else:
            return -(0.5 / sigma**2) * (chisq)

    @staticmethod
    @jit(nopython=True)
    def _complex_log_likelihood_numba(
        K,
        Dvec_real,
        Dvec_imag,
        Bvec,
        Bmat_real,
        Bmat_imag,
        B2mat,
        model_real,
        model_imag,
        model2,
        sigma,
        cplen,
        likelihood="studentst",
    ):
        # square model
        model2 = (model2 * np.conj(model2)).real

        dm = (
            np.vdot(np.linalg.solve(Bmat_real.T, model_real), Dvec_real).real
            + np.vdot(np.linalg.solve(Bmat_imag.T, model_imag), Dvec_imag).real
        )
        mm = np.vdot(np.linalg.solve(B2mat.T, model2), Bvec).real
        chisq = K + mm - 2 * dm

        # print(dm, mm, K)

        if likelihood == "studentst":
            return (
                logfactorial(cplen - 1)
                - lal.LN2
                - cplen * lal.LNPI
                - cplen * np.log(chisq)
            )
        else:
            return -(0.5 / sigma**2) * (chisq)
