#####################
Hierarchical analyses
#####################

For a pulsar to emit gravitational waves it must have a time varying mass quadrupole. The strength
of the observed gravitational-wave signal is proportional to the star's mass quadrupole moment
:math:`Q_{22}`, which is related to the star's fiducial ellipticity [1]_. Using gravitational-wave
observations, an individual pulsar's :math:`Q_{22}` can be estimated, resulting in a posterior
probability distribution marginalised over other parameters of the gravitational-wave signal (this
can include the pulsar's distance if a prior on the distance is known). However, a priori, the
underlying distribution of :math:`Q_{22}` (or equivalently ellipticity) for the pulsar population is
unknown. This is where `hierarchical Bayesian analysis
<https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling>`_ comes in. If you chose a
parameterisable distribution as a prior for the underlying :math:`Q_{22}` distribution, then
posteriors from multiple gravitational-wave pulsar observations can be combined, included those
where no signal is detected, to estimate the parameters (known as `hyperparameters
<https://en.wikipedia.org/wiki/Hyperparameter>`_) of that distribution [2]_. For observations of
:math:`N` pulsars, the combined data from which is represented by :math:`\mathbf{D}`, and assuming a
prior distribution for :math:`Q_{22}` defined by a set of hyperparameters :math:`\vec{\theta}`,
:math:`p(Q_{22}|\vec{\theta},I)`, then the posterior for :math:`\vec{\theta}` is given by:

.. math::

   p(\vec{\theta}|\mathbf{D}, I) = \left(\prod_{i=1}^N \int^{{Q_{22}}_i} p(\mathbf{d}_i|{Q_{22}}_i,I) p({Q_{22}}_i|\vec{\theta},I) {\textrm d}{Q_{22}}_i\right) p(\vec{\theta}|I),

where :math:`p(\mathbf{d}_i|{Q_{22}}_i,I)` is the marginalised likelihood distribution on
:math:`Q_{22}` for the an individual pulsar given it's obervations :math:`\mathbf{d}_i`, and
:math:`p(\vec{\theta}|I)` is the prior on the hyperparameters.

The hierarchical module in CWInPy allows the outputs of pulsar :ref:`parameter estimation<Known
pulsar parameter estimation>` for mutiple sources to be combined to estimate the hyperparameters of
several different potential distributions :math:`Q_{22}` (or ellipticity) distributions. To do this
it is required that the parameter estimation has been performed using the :math:`Q_{22}` when
defining the signal amplitude, rather than :math:`h_0`, and that the distance is either assumed to
be precisely known, or the parameter estimation has included distance within the estimation via the
use of a distance prior.


Evaluating the integral
-----------------------

Within the CWInPy implementation there are two ways provided of evaluating the integrals in the
equation for :math:`p(\vec{\theta}|\mathbf{D}, I)`.

The first method is to evaluate the integral numerically. A Gaussian kernel density estimate of the
*posterior* distribution :math:`p({Q_{22}}_i|\mathbf{d}_i,I)` is generated for each pulsar, which is
converted back into the required *likelihood* distribution via

.. math::

   p(\mathbf{d}_i|{Q_{22}}_i,I) = \frac{p({Q_{22}}_i|\mathbf{d}_i,I)}{p({Q_{22}}_i|I)} p(\mathbf{d}_i|I)

where :math:`p({Q_{22}}_i|I)` is the prior on :math:`Q_{22}` used during the parameter estimation
and :math:`p(\mathbf{d}_i|I)` is the marginal likelihood given the data for the i\ :sup:`th` pulsar.

The second method uses the fact that the integrals are equivalent to calculating the expectation
value of the prior distributuion:

.. math::

   \mathrm{E}[p({Q_{22}}_i|\vec{\theta},I)] = \int^{{Q_{22}}_i} p(\mathbf{d}_i|{Q_{22}}_i,I) p({Q_{22}}_i|\vec{\theta},I) {\textrm d}{Q_{22}}_i \approx \frac{1}{M} \sum_{j=1}^M p({Q_{22}}_{i,j}|\vec{\theta},I),

where the term on the right hand side is calculating the mean of the :math:`Q_{22}` distribution
evaluated at the :math:`M` samples from likelihood. In the case that the prior used for
:math:`Q_{22}` used during the parameter estimation was not uniform then the
:class:`~cwinpy.hierarchical.MassQuadrupoleDistribution` class will attempt to recreate likelihood
samples by reweighting the posterior samples. In this case the scaling factor from the marginal
likelihood is not included, however, Bayes factors for different distributions using the same data
still can still be calculated.

Both these methods should be equivalent.

Available distributions
-----------------------

The currently allowed distributions are:

 * a multi-modal bounded Gaussian distribution: :class:`~cwinpy.hierarchical.BoundedGaussianDistribution`
 * an exponential distribution: :class:`~cwinpy.hierarchical.ExponentialDistribution`
 * a power law distribution: :class:`~cwinpy.hierarchical.PowerLawDistribution`
 * a delta function distribution: :class:`~cwinpy.hierarchical.DeltaFunctionDistribution`

The :class:`~cwinpy.hierarchical.MassQuadrupoleDistribution` can use any of the above distributions
to define the distribution on the mass quadrupole to be inferred from a set of individual pulsar
mass quadrupole posteriors.

Example
=======

In this example we will use the :ref:`simulation<Pulsar simulations>` module to generate a set of
pulsars with mass quadrupoles drawn from an `exponential distribution
<https://en.wikipedia.org/wiki/Exponential_distribution>`_ and then show how the mean of that
exponential distribution can be estimated.

The code below will generate a HTCondor DAG that will simulate 50 pulsars with :math:`Q_{22}` values
drawn from an exponential distribution with a mean of :math:`\mu = 5\!\times\!10^{32}\,{\textrm
kg}\,{\textrm m}^2`. The pulsars' sky locations will be drawn uniformly over the sky and from the
default uniform distance distribution. The orientations of the pulsars will also be drawn from the
default uniform distributions, which are described in the
:class:`~cwinpy.pe.simulation.PEPulsarSimulationDAG` documentation. The simulation will only
generate observations from a single gravitational-wave detector, the LIGO Hanford detector ("H1"),
in this case.

The parameter estimation performed on each of these pulsars will use a uniform prior on
:math:`Q_{22}`, a uniform prior over the orientation parameters (within the non-degenerate ranges),
and will assume the distance to each is precisely know and equal to the simulated value.

.. code-block:: python

   from cwinpy.pe.simulation import PEPulsarSimulationDAG
   from bilby.core.prior import PriorDict, Uniform, Exponential, Sine

   # set the Q22 distribution
   mean = 5e32
   q22dist = Exponential(name="q22", mu=mean)

   # set the prior for each pulsar
   prior = PriorDict({
       "q22": Uniform(0.0, 1e38, name="q22"),
       "iota": Sine(name="iota"),
       "phi0": Uniform(0.0, np.pi, name="phi0"),
       "psi": Uniform(0.0, np.pi / 2, name="psi"),
   })

   # set the detector
   detectors = ["H1"]

   npulsars = 50  # number of pulsars to simulate

   # generate and submit the simulation DAG
   run = PEPulsarSimulationDAG(
       ampdist=ampdist,
       prior=prior,
       npulsars=npulsars, 
       detector=detectors,
       basedir="/home/user/exponential",  # base directory for analysis output
       getenv=True,  # use current host environment variables if working on a cluster 
       submit=True,  # automatically submit the HTCondor DAG
       numba=True,   # use numba for likelihood evaluation
       sampler_kwargs={'Nlive': 1000, 'sample': 'rslice'},
   )

.. note::

   If running on `LSC DataGrid computing resources
   <https://computing.docs.ligo.org/lscdatagridweb/>`_ the ``accountuser`` argument may be required
   to set the HTCondor ``accounting_group_user`` submit variable to your
   ``albert.einstein@ligo.org`` username, and ``accountgroup`` may be required to set the
   ``accounting_group`` submit variable to a valid `accounting tag
   <https://accounting.ligo.org/user>`_. You may also need to run ``ligo-proxy-init
   albert.einstein`` prior to submitting the HTCondor DAG that is generated.

Once the parameter estimation jobs have completed there should be a ``results`` directory within the
directory specified by the ``basedir`` argument to
:class:`~cwinpy.pe.simulation.PEPulsarSimulationDAG`, which contains the results from each pulsar in
the simulation. These can be combined to infer the hyperparameter of the :math:`Q_{22}`
distribution. Here we show how to do this using both methods of evaluating the marginalisation
integrals described :ref:`above<Evaluating the integral>`, and estimating the posterior on the
hyperparameters using both stochastic sampling (in this case using the nested sampling algorithm
implemented in dynesty) and over a grid of points. The former methods is maybe over-the-top for this
single dimensional example, but does have the advantage that the range and resolution of the
required grid does not need to be known a priori.

Sampling the hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To start with, the results for each pulsar need to be read in. This can be done using a bilby
:class:`~bilby.core.result.ResultList` object using (assuming the base directory given above):

.. code-block:: python

   import glob
   from bilby.core.result import ResultList

   # get a list of result files
   resultfiles = glob.glob("/home/user/exponential/results/J*/*.json")
   
   # read in as list of bilby Result objects
   data = ResultList(resultfiles)

Then a :class:`~cwinpy.hierarchical.MassQuadrupoleDistribution` needs to be set up and provided with
the type of distribution for which the hyperparameters will be estimated, the priors on those
hyperparameters, the method of :ref:`evaluating the interals<Evaluating the integral>`, and any
arguments required for the stochastic or grid-based sampling method. In this case the distribution
is an exponential, and we will use a :class:`~bilby.core.prior.HalfNormal` prior (a Gaussian
distribution with a mode a zero, but excluding negative values) for the hyperparameter :math:`\mu`.

.. code-block::

   # set the distribution type
   distribution = "exponential"
   
   # set the hyperparameter prior distribution using a dictionary
   sigma = 1e34
   distkwargs = {"mu": HalfNormal(sigma, name="mu")}

   # create the MassQuadrupoleDistribution object
   mqd = MassQuadrupoleDistribution(
       data=data,
       distribution="exponential",
       distkwargs=distkwargs,
       sampler_kwargs=sampler_kwargs,
       integration_method="expectation",
       nsamples=500,
   )

.. note::

   If you have a very large number of pulsars it may not be memory efficient to read them all in in
   one go to a :class:`~bilby.core.result.ResultList` object. They can instead be pass one at a time
   to the :class:`~cwinpy.hierarchical.MassQuadrupoleDistribution`, using the
   :meth:`~cwinpy.hierarchical.MassQuadrupoleDistribution.add_data` method, e.g.:

   .. code-block:: python

      mqd = MassQuadrupoleDistribution()

API
---

.. automodule:: cwinpy.hierarchical
   :members:

Hierarchical analysis references
--------------------------------

.. [1] `N. K. Johnson-McDaniel & B. J. Owen
   <https://ui.adsabs.harvard.edu/abs/2013PhRvD..88d4004J/abstract>`_, *PRD*,
   **88**, 044004 (2013).
.. [2] `M. Pitkin, C. Messenger, X. Fan
   <https://ui.adsabs.harvard.edu/abs/2018PhRvD..98f3001P/abstract>`_, *PRD*,
   **98**, 063001 (2018).