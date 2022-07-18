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
        self.generate_training_set()

        # generate the reduced basis
        self.generate_reduced_basis()

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

    def generate_training_set(self):
        """
        Generate ``n`` model instances, using values drawn from the prior.
        """

        self._training_set_model = []

        model = self.kwargs.get("model", HeterodynedCWSimulator)
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

            modelfunc = model.model

            # copy of heterodyned parameters
            newpar = deepcopy(self.par)
        else:
            modelfunc = model

        # draw values from prior
        samples = self.priors.sample(self.ntraining)
        self.samples = []

        for i in range(self.ntraining):
            # update par file
            self.samples.append([])
            for prior in samples:
                self.samples[-1].append(samples[prior][i])

                if isinstance(model, HeterodynedCWSimulator):
                    newpar[prior] = samples[prior][i]

            if isinstance(model, HeterodynedCWSimulator):
                m = modelfunc(
                    deepcopy(newpar),
                    outputampcoeffs=False,
                    updateSSB=True,
                    updateBSB=True,
                    updateglphase=True,
                    freqfactor=self.kwargs.get("freq_factor", 2),
                )
            else:
                m = modelfunc(self.x, **{prior: samples[prior][i] for prior in samples})

            self._training_set_model.append(m)

    def generate_reduced_basis(self):
        """
        Generate the reduced basis for the training set.
        """

        self._rb_model = {}
        self._rb_nodes = {}
        self._rb_model_squared = {}
        self._rb_squared_nodes = {}

        # reduced basis for the model training set
        if self._training_set_model[0].dtype == complex:
            training_m_real = np.array([m.real for m in self._training_set_model])
            training_m_imag = np.array([m.imag for m in self._training_set_model])
        else:
            training_m_real = np.array(self._training_set_model)
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
            else:
                self._rb_model[t] = None

        # reduced basis for the squared model training set
        if self._training_set_model[0].dtype == complex:
            training_m_real = np.array([m.real**2 for m in self._training_set_model])
            training_m_imag = np.array([m.imag**2 for m in self._training_set_model])
        else:
            training_m_real = np.array([m**2 for m in self._training_set_model])
            training_m_imag = None

        for t, m in zip(["real", "imag"], [training_m_real, training_m_imag]):
            if m is not None:
                self._rb_model_squared[t] = reduced_basis(
                    training_set=m,
                    physical_point=self.x,
                    greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
                    normalize=True,
                )
                self._rb_squared_nodes[t] = self._rb_model_squared[t].basis.eim_.nodes
            else:
                self._rb_model_squared[t] = None
