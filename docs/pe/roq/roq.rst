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
model, respectively) that again only rely on the model being calculated at :math:`N` values to
accurately reconstruct the full terms.

There is, of course, some initial overhead to calculating the basis vectors, but this is often
relatively small compared to the overall savings over the course of the full parameter estimation.
There may also be cases where the parameter space is too uncorrelated (maybe just due to being too
large) to mean that the number of basis vectors is less than the overall length of the data. In such
cases, it would just revert back to calculating the full likelihood.

CWInPy uses the `arby <https://arby.readthedocs.io/>`_ package for the generation of the reduced
order model, reduced basis vectors and interpolants for the likelihood function components.

ROQ in CWInPy
-------------

Within CWInPy, if performing parameter estimation in the case where the heterodyne has been assumed
to perfectly remove the signal's phase evolution (i.e., no parameters that cause the phase to
evolve, such as frequency or sky location parameters, are to be estimated) then the likelihood
function is already efficient and uses pre-summed terms (see
:meth:`~cwinpy.pe.likelihood.TargetedPulsarLikelihood.dot_products` and Appendix C of [4]_).
However, if the phase is potentially evolving (i.e., the observed electromagnetic pulse timing
solution does not exactly match the gravitational-wave evolution and a residual phase evolution is
left in the heterodyned data), and as such parameter estimation needs to be performed over some
phase parameters (e.g., frequency) then these pre-summations cannot be done and each likelihood
evaluation for different phase parameter values must re-calculate the model at all time stamps in
the data. Along with the generally larger parameter space to explore, this will considerably slow
down the inference.

While CWInPy's likelihood evaluations will operate in this mode by default, in these cases it can be
useful to instead use *Reduced Order Quadrature* to help speed things up. There are several caveats
to this, in particular, different frequency bins are highly uncorrelated, so it is unlikely that a
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
    constraints, due to the full matrix of training models needing to be stored in memory. To avoid
    this, the ``bbmaxlength`` keyword of the :class:`~cwinpy.data.HeterodynedData` can be passed to
    ensure that the data is broken up into smaller chunks on which the model can be individually
    trained. This would be achieved through the ``data_kwargs`` keyword of :func:`cwinpy.pe.pe`,
    e.g., using

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
    its epoch, i.e., the ``PEPOCH`` value, being at the start or the centre of the dataset that is
    being analysed. This may require updating frequencies, frequency derivatives and sky locations
    to that epoch, as applicable. This may be more necessary if using the ROQ likelihood for known
    pulsars with an older observation epoch, but should not be problematic if following up CW
    candidates from other wider parameter CW signal searches.

    3. If wanting to use the ROQ as part of a pipeline running on multiple pulsars, you will most
    likely have to specify different prior files for each pulsar. This can be done by making sure
    that the ``priors`` option in the configuration file, or passed directly to
    :func:`cwinpy.pe.pe`, is either:
    
    * the path to a directory that contains individual prior files for each pulsar, where the files
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

In this case, when running *without* the ROQ likelihood a single likelihood evaluation takes about
3.6 ms and the inference takes nearly 21 minutes in total. For the ROQ likelihood, a single
likelihood evaluation takes about 0.6 ms and the inference takes just over three minutes. So, we see
a significant speed advantage with the ROQ.

The posteriors from both cases can be see below and show very good agreement.

.. thumbnail:: example1_roq_posteriors.png
   :width: 450px
   :align: center

ROQ on a hardware injection
===========================

As another example, we will use data containing a continuous hardware injection signal to show the
ROQ likelihood in action. In particular, we will follow the :ref:`example<Example: single detector
data>`_ given in the :ref:`parameter estimation documentation<Known pulsar parameter estimation>`.
The heterodyned data file containing the signal can be downloaded here
:download:`fine-H1-PULSAR08.txt.gz <../data/fine-H1-PULSAR08.txt.gz>` and the Tempo(2)-style pulsar
parameter (``.par``) file containing the parameters for this simulated signal can be downloaded
:download:`here <../data/PULSAR08.par>`. It is worth noting that in this case the frequency epoch
(``PEPOCH`` in the ``.par`` file) of MJD 52944, i.e., a GPS time of 751680013, is about 12 years
before the start of the heterodyned data (GPS 1132477888).

We will attempt to perform parameter estimation over the unknown amplitude parameters and over a
small range of rotational frequency and frequency derivative. In the prior file shown below (which
can be downloaded :download:`here <../data/roq_example_prior.txt>`) the a prior spans 20 nHz in
rotation frequency and 0.002 nHz/s in rotation frequency derivative:

.. literalinclude:: ../data/roq_example_prior.txt
   :language: python

To show that having the frequency epoch and data epoch being considerably different can be
problematic for the ROQ, we will build a reduced order model for this case:

.. code-block:: python

   from cwinpy import HeterodynedData

   # read in the data
   het = HeterodynedData("fine-H1-PULSAR08.txt.gz", par="PULSAR08.par", detector="H1")

   # generate the reduced order model using the prior
   roq = het.generate_roq("roq_example_prior.txt")

   # show number of model basis vectors
   print(f"Data length: {len(het)}\nNumber of training data: {roq[0].ntraining}\nNumber of model bases: {len(roq[0]._x_nodes)}")

.. code-block:: text

   Data length: 7979
   Number of training data: 5000
   Number of model bases: 2133

We can see that the number of model bases is only a factor of about 4 less than the length of the
data, meaning that the speed advantage may not be very significant. Also, the number is only a
factor of about two less than the number of training models input, which runs that risk that parts
of the parameter space are not well covered and that that basis is therefore `overfitted
<https://en.wikipedia.org/wiki/Overfitting>`_` to the training set.

We will therefore instead update the ``.par`` file parameters and priors to the middle of the
data epoch and see the effect (will will keep the overall prior ranges the same).

.. warning::

   In this case, we know the true parameters of the signal so there's no risk in updating both the
   ``.par`` file while keeping the prior ranges that same, but in a real situation it may not
   be possible to update the ``.par`` file without expanding the prior ranges to reflect the
   uncertainty at the new epoch.

.. code-block:: python

   from astropy.time import Time
   from bilby.core.prior import PriorDict
   from cwinpy import HeterodynedData
   from cwinpy.parfile import PulsarParameters

   het = HeterodynedData("fine-H1-PULSAR08.txt.gz")
   par = PulsarParameters("PULSAR08.par")

   # time to new epoch
   dt = het.times.value[int(len(het) // 2)] - par["PEPOCH"]

   # update rotation frequency
   f0new = par["F0"] + par["F1"] * dt
   par["F"] = [f0new, par["F1"]]

   # update epoch
   par["PEPOCH"] = het.times.value[int(len(het) // 2)]

   # output new par (for later use!)
   par.pp_to_par("PULSAR08_updated.par")

   # read in prior
   prior = PriorDict(filename="roq_example_prior.txt")

   # update frequency prior
   prior["f0"].minimum = f0new - 1e-7
   prior["f0"].maximum = f0new + 1e-7

   # output new prior (for later use!)
   prior.to_file(outdir=".", label="roq_example_prior_update")

   # read in data with updated parameters
   hetnew = HeterodynedData("fine-H1-PULSAR08.txt.gz", par=par, detector="H1")

   # generate ROQ
   roq = hetnew.generate_roq(prior)

   print(f"Data length: {len(hetnew)}\nNumber of training data: {roq[0].ntraining}\nNumber of model bases: {len(roq[0]._x_nodes)}")

.. code-block:: text

   Data length: 7979
   Number of training data: 5000
   Number of model bases: 110

We see that the number of model bases is considerably reduced, so a significant speed up should
occur when using the ROQ likelihood.

We will now run parameter estimation using the ROQ likelihood using the update ``.par`` file and
prior:

.. code-block:: bash

   cwinpy_pe --par-file PULSAR08_updated.par --data-file fine-H1-PULSAR08.txt.gz --detector H1 -o roq -l PULSAR08 --prior roq_example_prior_update.prior --roq

In this case, a single likelihood evaluation takes just over 0.3 ms and the analysis finishes in
just over 4 minutes.

Plotting the results (after moving into the ``roq`` directory given as the output location) gives:

.. code-block:: python

   from cwinpy.plot import Plot

   plot = Plot("PULSAR08_result.hdf5", pulsar="../PULSAR08_updated.par", untrig="cosiota")
   plot.plot()
   plot.save("example2_roq_posteriors.png", dpi=200)

.. thumbnail:: ../data/roq/example2_roq_posteriors.png
   :width: 450px
   :align: center

which shows that the frequency and frequency derivatives are found at the correct values well within
the prior ranges.

Using ROQ with a search pipeline
--------------------------------

The ROQ likelihood can be used within a search pipeline, i.e., when running
``cwinpy_knope_pipeline`` or ``cwinpy_pe_pipeline``. This will required individual prior files be
specified for each source being searched for as the default priors will not require an ROQ (it will
work will an ROQ, but is not necessary). We will perform a search for two pulsars in LIGO O1 data:

* the hardware injection, ``PULSAR01``, for which we will "accidentally" heterodyne using a slightly
  offset sky location and then perform parameter estimation over a small sky patch including the
  true location;
* the pulsar PSR J1932+17, which was an :ref:`outlier<Example: analysis outlier>`_ in the O1
  analysis, for which we will search over a small frequency and frequency derivative range based on
  expanding the uncertainty given by the electromagnetic timing solution by a factor of 5.

For the ``PULSAR01`` hardware injection, we will use a ``.par`` file containing

.. literalinclude:: ../data/PULSAR01_offset.par
   :language: text

where the right ascension and declination have been shifted by 0.01 mrad (0.138 seconds) and -0.01 mrad
(-2.063 arcsec), respectively, compared to their actual values. We will use a prior file that
searches over right ascension and declination over a uniform patch spanning ±0.02 mrad in each
coordinate:

.. literalinclude:: ../data/PULSAR01_prior.txt
   :language: python

For PSR J1932+17, the ``.par`` file we will use (as can be found :ref:`here<Example: analysis outlier>`_)
contains:

.. literalinclude:: ../../skyshifting/J1932+17.par
   :language: text

and the prior file is:

.. literalinclude:: ../data/ROQ_J1932+17_prior.txt
   :language: python

The configuration file used is:

.. literalinclude:: ../data/roq_pipeline_example.ini
   :language: ini

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