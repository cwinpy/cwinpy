from copy import deepcopy

import numpy as np
from arby import reduced_basis
from bilby.core.prior import PriorDict

from ..signal import HeterodynedCWSimulator


class GenerateROQ:
    def __init__(self, data, x, priors, det, par, n=1000, **kwargs):
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
        self.det = det
        self.par = par
        self.kwargs = kwargs

        # the number of training examples to use
        self.n = n

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

        if dataarr.dtype not in [complex]:
            raise TypeError("Data must be a complex type")

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

        model = HeterodynedCWSimulator(
            self.par, self.det, self.x, usetempo2=self.kwargs.get("usetempo2", False)
        )

        # copy of heterodyned parameters
        newpar = deepcopy(self.par)

        # draw values from prior
        samples = self.priors.sample(self.n)
        self.samples = []

        for i in range(self.n):
            # update par file
            self.samples.append([])
            for prior in samples:
                newpar[prior] = samples[prior][i]
                self.samples[-1].append(samples[prior][i])

            m = model.model(
                deepcopy(newpar),
                outputampcoeffs=False,
                updateSSB=True,
                updateBSB=True,
                updateglphase=True,
                freqfactor=self.kwargs.get("freq_factor", 2),
            )

            self._training_set_model.append(m)

    def generate_reduced_basis(self):
        """
        Generate the reduced basis for each training set.
        """

        self._rb_model = []
        self._rb_model_squared = []

        self._rb_model_real = reduced_basis(
            training_set=np.array([m.real for m in self._training_set_model]),
            physical_point=self.x,
            greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
            normalize=True,
        )

        self._rb_model_imag = reduced_basis(
            training_set=np.array([m.imag for m in self._training_set_model]),
            physical_point=self.x,
            greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
            normalize=True,
        )

        self._rb_model_squared_real = reduced_basis(
            training_set=np.array([m.real**2 for m in self._training_set_model_real]),
            physical_point=self.x,
            greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
            normalize=True,
        )

        self._rb_model_squared_imag = reduced_basis(
            training_set=np.array([m.imag**2 for m in self._training_set_model_imag]),
            physical_point=self.x,
            greedy_tol=self.kwargs.get("greedy_tol", 1e-12),
            normalize=True,
        )
