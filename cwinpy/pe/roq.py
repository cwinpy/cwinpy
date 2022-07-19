from copy import deepcopy

import numpy as np
from arby import reduced_basis
from bilby.core.prior import PriorDict

from ..signal import HeterodynedCWSimulator


class GenerateROQ:
    def __init__(self, data, x, priors, ntraining=1000, **kwargs):
        """
        Generate the Reduced Order Quadrature for the parameters in the prior.

        Parameters
        ----------
        data: array_like
            The data
        """

        self.x = x
        self.data = data
        self.priors = priors
        self.kwargs = kwargs

        # the number of training examples to use
        self.ntraining = ntraining

        # generate training data
        ts = self.generate_training_set()

        if self.kwargs.get("store_training_data", False):
            # store training data if needed
            self.training_data = ts

        # generate the reduced basis
        self.generate_reduced_basis(ts)

        # likelihood term for squared data
        sigma = self.kwargs.get("sigma", 1.0)
        if self.data.dtype == complex:
            self._K = np.sum((self.data.real / sigma) ** 2) + np.sum(
                (self.data.imag / sigma) ** 2
            )
        else:
            self._K = np.sum((self.data / sigma) ** 2)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if not isinstance(x, np.ndarray, list):
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
        if isinstance(model, HeterodynedCWSimulator):
            if "par" not in self.kwargs:
                raise ValueError("'par', parameter file must be in keyword arguments")

            if "det" not in self.kwargs:
                raise ValueError("'det', detector name must be in keyword arguments")

            par = self.kwargs.get("par")
            det = self.kwargs.get("det")
            model = HeterodynedCWSimulator(
                par, det, self.x, usetempo2=self.kwargs.get("usetempo2", False)
            )

            self._initial_par = deepcopy(par)
            self._par = deepcopy(par)

            self._model = model.model
        else:
            self._model = model

    def _model_at_nodes(self, xr, xi, x2):
        if hasattr(self, "_initial_par"):
            det = self.kwargs.get("det")
            modelr = HeterodynedCWSimulator(
                self._initial_par,
                det,
                xr,
                usetempo2=self.kwargs.get("usetempo2", False),
            )
            self._interpolated_model_real = modelr.model
            modeli = HeterodynedCWSimulator(
                self._initial_par,
                det,
                xi,
                usetempo2=self.kwargs.get("usetempo2", False),
            )
            self._interpolated_model_imag = modeli.model
            model2 = HeterodynedCWSimulator(
                self._initial_par,
                det,
                x2,
                usetempo2=self.kwargs.get("usetempo2", False),
            )
            self._interpolated_model2 = model2.model
        else:
            self._interpolated_model = self.model

    def generate_training_set(self):
        """
        Generate ``n`` model instances, using values drawn from the prior.
        """

        training_set_model = []

        self.model = self.kwargs.get("model", HeterodynedCWSimulator)

        # draw values from prior
        samples = self.priors.sample(self.ntraining)
        self.samples = []

        for i in range(self.ntraining):
            # update par file
            self.samples.append([])
            for prior in samples:
                self.samples[-1].append(samples[prior][i])

                if hasattr(self, "_par"):
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

                training_set_model.append(m)

        return training_set_model

    def generate_reduced_basis(self, training):
        """
        Generate the reduced basis for the training set.
        """

        self._rb_model = {}
        self._rb_nodes = {}

        # reduced basis for the model training set
        if self.data.dtype == complex:
            training_m_real = np.array([m.real for m in training])
            training_m_imag = np.array([m.imag for m in training])
        else:
            training_m_real = np.array(training)
            training_m_imag = None

        for t, m in zip(["real", "imag"], [training_m_real, training_m_imag]):
            if m is not None:
                self._rb_model[t] = reduced_basis(
                    training_set=m,
                    physical_point=self.x,
                    greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
                    normalize=True,
                )
                self._rb_nodes[t] = self._rb_model[t].basis.eim_.nodes

        # reduced basis for the squared model training set
        if self.data.dtype == complex:
            training_m = np.array([(m * np.conj(m)).real for m in training])
        else:
            training_m = np.array([m**2 for m in training])

        self._rb_model_squared = reduced_basis(
            training_set=training_m,
            physical_point=self.x,
            greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
            normalize=True,
        )
        self._rb_squared_nodes = self._rb_model_squared.basis.eim_.nodes

        # get matrices
        sigma = self.kwargs.get("sigma", 1.0)
        if self.data.dtype == complex:
            self._D = np.einsum(
                "i,ji->j", self.data.real / sigma, self._rb_model["real"].basis.data
            ) + 1j * np.einsum(
                "i,ji->j", self.data.imag / sigma, self._rb_model["imag"].basis.data
            )

            self._B = (
                self._rb_model["real"].basis.eim_.interpolant
                + 1j * self._rb_model["imag"].basis.eim_.interpolant
            ).T
            self._B2 = (
                self._rb_model_squared["real"].basis.eim_.interpolant
                + 1j * self._rb_model_squared["imag"].basis.eim_.interpolant
            ).T
        else:
            self._D = np.einsum(
                "i,ji->j", self.data / sigma, self._rb_model["real"].basis.data
            )
            self._B = self._rb_model["real"].basis.eim_.interpolant.T
            self._B2 = self._rb_model_squared["real"].basis.eim_.interpolant

        # set model at the interpolation nodes
        self._model_at_nodes()

    def log_likelihood(self, **kwargs):
        """
        Calculate the log-likelihood using the reduced order basis.
        """
