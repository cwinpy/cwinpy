########################
Reduced Order Quadrature
########################

When performing Bayesian parameter estimation, it is often the case that calculating the likelihood
function is the main computational expenses. This could be due to the likelihood being calculated
for a large data set, or the source model being slow to evaluate, or a combination of both.
Stochastic sampling methods often required hundreds of thousands or millions of likelihood
evaluations to properly characterise the posterior probability distribution, so a slow likelihood
means slow inference.

A method of speeding up the likelihood evaluation is to use *Reduced Order Quadrature* (ROQ) [1]_
[2]_ [3]_. This works well if the source model has strong correlations within the required parameter
space. In such a case, a set of orthonormal basis vectors can be found that when linearly combined
with appropriate weights gives an accurate representation of the original model anywhere in
parameter space. The basis vectors can be precomputed before any inference and used to generate an
interpolant for the model. If there are :math:`N` basis vectors, the original model only needs to be
calculated at :math:`N` points to work out the weights for combining the basis vectors to reproduce
the full model. If the original length of the data is :math:`M` and :math:`N \ll M` then there can
be a significant speed up in calculating the model.

ROQ itself refers to a further neat trick that reduces the need to sum terms in the likelihood
function over the full :math:`M` values and instead only sum over :math:`N` values. The basis
vectors allow the construction of pre-calculated interpolants for the :math:`(d \cdot m)` and
:math:`(m \cdot m)` terms in the likelihood (where :math:`d` and :math:`m` are the data and the
model, respectively) that again only rely on the model being calculated to :math:`N` values to
accurately reconstruct the full terms.

There is, of course, some initial overhead to calculating the basis vectors, but this is often
relatively small compared to the overall savings over the course of the full parameter estimation.
There may also be cases where the parameter space is too uncorrelated (maybe just due to being too
large) to mean that the number of basis vectors is less than the overall length of the data. In such
cases, it would just revert back to calculating the full likelihood.

ROQ in CWInPy
-------------

Within CWInPy, if performing parameter estimation in the case where the heterodyne has been assumed
to perfectly remove the signal's phase evolution (i.e., no parameters that cause the phase to
evolve, such as frequency or sky location parameters, are to be estimated) then the likelihood
function is already efficient and uses pre-summed terms (see
:meth:`~cwinpy.pe.likelihood.TargetedPulsarLikelihood.dot_products` and Appendix C of [4]_. However,
if the phase is potentially evolving (i.e., the observed electromagnetic pulse timing solution does
not exactly match the gravitational-wave evolution and a residual phase evolution is left in the
heterodyned data), and as such parameter estimation needs to be performed over some phase parameters
(e.g., frequency) then these pre-summations cannot be done and each likelihood evaluation for
different phase parameter values must re-calculate the model at all time stamps in the data. Along
with the generally larger parameter space to explore, this will considerably slow down the
inference.

While CWInPy's likelihood evaluations will operate in this mode by default, in these cases it can be
useful to instead use *Reduced Order Quadrature* to help speed things up. There are several caveats
to this, in particular, different frequency bins are highly uncorrelated, so it is unlikely that an
small reduced basis set will be found if the frequency offset of many tens of frequency bins. Also,
for the length of dataset often found in practice (or order a million data points over a year of
observations), computer memory constraints may make it necessary to build the reduced basis on
shorter blocks of data. Unlike the use of ROQs for parameter estimation with Compact Binary
Coalescences [2]_, the same set of basis vectors cannot generally be used for different sources, so
need to be calculated individually for each source and dataset.

ROQ comparison
==============

In a first example of using the ROQ likelihood we will generate a simulate continuous signal and
then purposely heterodyne it using frequency and frequency derivative parameters that are offset
from the true values. We will then attempt to estimate the source parameters including the frequency
and frequency derivative using both the standard likelihood and the ROQ likelihood. The script for
this is shown below.

To view the code that generates the simulated signal click the hidden dropdown below. 

.. dropdown:: Signal generation

    .. literalinclude:: example1.py
       :lines: 4-8,11,14-126

.. literalinclude:: example1.py
   :lines: 3,8-13,127-

ROQ API
-------

The API for the ROQ generation class is given below.

.. autoclass:: cwinpy.pe.roq.GenerateROQ
    :members:

ROQ References
==============

.. [1] `H. Antil et al. <https://ui.adsabs.harvard.edu/abs/2012arXiv1210.0577A/abstract>`_,
   *Journal of Scientific Computing*, **57**, 604-637 (2013).

.. [2] `P. Canizares, S. E. Field, J. R. Gair & M. Tiglio <https://ui.adsabs.harvard.edu/abs/2013PhRvD..87l4005C/abstract>`,
   *PRD*, **87**, 124005 (2013).

.. [3] `M. Tiglio & A. Villanueva <https://ui.adsabs.harvard.edu/abs/2022LRR....25....2T/abstract>`,
   *LRR*, **25**, 2 (2022).

.. [4] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
       <https://arxiv.org/abs/1705.08978v1>`_, 2017.