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

Usage
=====

The usage of the ROQ likelihood can be specified by using keyword arguments to the
:func:`cwinpy.pe.pe` functions, e.g., 

.. code-block:: python

    pe(
        ...
        roq=True,
        roq_kwargs={"ntraining": 5000, "greedy_tol": 1e-12},
    )

where the ``roq_kwargs`` dictionary contains keywords for the :class:`~cwinpy.pe.roq.GenerateROQ`
class.

These keywords can also be used in the ``cwinpy_pe``, ``cwinpy_pe_pipeline``, ``cwinpy_knope`` and
``cwinpy_knope_pipeline`` configuration files or APIs. For example, to use the ROQ likelihood a
``cwinpy_pe`` configuration file would contain:

.. code-block:: ini

   roq = True
   roq_kwargs = {'ntraining': 5000, 'greedy_tol': 1e-12}

.. attention::

    There are several things to be aware of when using the ROQ likelihood:

    1. When generating the reduced order model, a training set of ``ntraining`` models is produced. If
    this number is large and the data set is long, this can potentially run into computer memory
    constraints. To avoid this, the ``bbmaxlength`` keyword of the
    :class:`~cwinpy.data.HeterodynedData` can be passed to ensure that the data is broken up into
    smaller chunks on which the model can be individually trained. This would be achieved through
    the ``data_kwargs`` keyword of :func:`cwinpy.pe.pe`, e.g., using

    .. code-block:: python

        pe(
            ...
            roq=True,
            roq_kwargs={"ntraining": 5000, "greedy_tol": 1e-12},
            data_kwargs={"bbmaxlength": 10000},
        )

    if using :func:`cwinpy.pe.pe` directly, or

    .. code-block:: ini

       roq = True
       roq_kwargs = {'ntraining': 5000, 'greedy_tol': 1e-12}
       data_kwargs = {'bbmaxlength': 10000}

    in a configuration file.

    2. To give the reduced order model the best chance of producing a reduced basis that is not too
    large, it is best (if possible) to try and update the pulsar parameter file being used to have
    its epoch, i.e., the ``PEPOCH`` to being at the start or the centre of the dataset that is being
    analysed. This may require updating frequencies, frequency derivatives and sky locations to that
    epoch, as applicable.

    3. If wanting to use the ROQ as part of a pipeline running on multiple pulsars, you will most
    likely have to specify different prior files for each pulsar. This can be done by making sure
    that the ``priors`` option in the configuration file or passed directly to :func:`cwinpy.pe.pe`,
    is either:
    
    * the path to a directory that contains individual prior file for each pulsar, where the files
      are identified by containing the pulsar's ``PSRJ`` name in the filename;
    * a list of paths to individual prior files for each pulsar, where the files
      are identified by containing the pulsar's ``PSRJ`` name in the filename;
    * or, a dictionary with keys being the pulsar ``PSRJ`` names and values being paths to the prior
      file for that pulsar.

    For example, the configuration file could contain:

    .. code-block:: ini

       priors = ['/home/user/parfiles/J0534+2200.prior', '/home/user/parfiles/J0835-4510.par']

       # or alternatively (if the prior files are in the same directory)
       priors = /home/user/parfiles

       # or using a dictionary (if, say, the prior files do not contain the
       # pulsar name in the file name)
       priors = {'J0534+2200': '/home/user/parfiles/crab.prior', 'J0835-4510': '/home/user/parfiles/vela.par'}

ROQ comparison example
======================

In a first example of using the ROQ likelihood, we will generate a simulated continuous signal and
then purposely heterodyne it using frequency and frequency derivative parameters that are offset
from the "true" values. We will then attempt to estimate the source parameters including the
frequency and frequency derivative using both the standard likelihood and the ROQ likelihood. The
:download:`script <example1_roq.py>` for this is shown below.

To view the code that generates the simulated signal click the hidden dropdown below. 

.. dropdown:: Signal generation
    :icon: code
    :animate: fade-in
    :color: secondary
    :margin: 0

    .. literalinclude:: example1_roq.py
       :lines: 4-8,11,14-126

.. literalinclude:: example1_roq.py
   :lines: 3,8-13,127-

In this case, when running without the ROQ likelihood a single likelihood evaluation takes about 3.6
ms and the inference takes nearly 21 minutes in total. For the ROQ likelihood, a single likelihood
evaluation takes about 0.6 ms and the inference takes just over three minutes. So, we see a
significant speed advantage with the ROQ.

The posteriors from both cases can be see below and show very good agreement.

.. thumbnail:: example1_roq_posteriors.png
   :width: 450px
   :align: center

ROQ API
-------

The API for the ROQ generation class is given below. In general, direct use of this class is not
required and generation of ROQ for likelihoods will be performed via the
:class:`~cwinpy.pe.likelihood.TargetedPulsarLikelihood` class and the
:meth:`~cwinpy.data.HeterodynedData.generate_roq` method of the
:class:`~cwinpy.data.HeterodynedData` class.

.. autoclass:: cwinpy.pe.roq.GenerateROQ
    :members:

ROQ References
==============

.. [1] `H. Antil et al. <https://ui.adsabs.harvard.edu/abs/2012arXiv1210.0577A/abstract>`_,
   *Journal of Scientific Computing*, **57**, 604-637 (2013).

.. [2] `P. Canizares, S. E. Field, J. R. Gair & M. Tiglio <https://ui.adsabs.harvard.edu/abs/2013PhRvD..87l4005C/abstract>`_,
   *PRD*, **87**, 124005 (2013).

.. [3] `M. Tiglio & A. Villanueva <https://ui.adsabs.harvard.edu/abs/2022LRR....25....2T/abstract>`_,
   *LRR*, **25**, 2 (2022).

.. [4] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
       <https://arxiv.org/abs/1705.08978v1>`_, 2017.