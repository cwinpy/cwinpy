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

Calculating the likelihood
--------------------------

The above equation requires the marginalised *likelihood* :math:`p(\mathbf{d}_i|{Q_{22}}_i,I)` for
each pulsar. However, the parameter estimation stage instead produces samples drawn from the
*posterior* distribution. It therefore requires undoing the effect of the prior on :math:`Q_{22}`
used for each pulsar. In the case that that prior was uniform over a range, and the edges of the
that range are outside the bulk of the likelihood, the posterior samples can be used as likelihood
samples. If the prior was not uniform then the implementation in CWInPy, via the
:class:`~cwinpy.hierarchical.MassQuadrupoleDistribution` class, will recreate likelihood samples via
reweighting. There will still be a scale factor missing due to the normalisation of the prior and
posterior (the marginal likelihood, aka the Bayesian evidence) not being accounted for. This final
posterior, :math:`p(\vec{\theta}|\mathbf{D}, I)`, that will be produce will be missing a scale
factor. Bayes factors for different distributions using the same data st can still be calculated.

Evaluating the integral
-----------------------

Within the CWInPy implementation there are two ways provided of evaluating the integrals in the
equation for :math:`p(\vec{\theta}|\mathbf{D}, I)`.

The first method is to evaluate the integral numerically. A Gaussian kernel density estimate of the
likelihood distribution :math:`p(\mathbf{d}_i|{Q_{22}}_i,I)` is generated for each pulsar, which is
then used when numerically evaluating the integral on a grid via the trapezium rule.

The second method uses the fact that the integrals are equivalent to calculating the expectation
value of the prior distributuion:

.. math::

   \mathrm{E}[p({Q_{22}}_i|\vec{\theta},I)] = \int^{{Q_{22}}_i} p(\mathbf{d}_i|{Q_{22}}_i,I) p({Q_{22}}_i|\vec{\theta},I) {\textrm d}{Q_{22}}_i \approx \frac{1}{M} \sum_{j=1}^M p({Q_{22}}_{i,j}|\vec{\theta},I),

where the term on the right hand side is calculating the mean of the :math:`Q_{22}` distribution
evaluated at the :math:`M` samples from likelihood.

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
pulsars with mass quadrupoles drawn from an expontial distribution and then show how the mean of
that exponential distribution can be estimated.

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