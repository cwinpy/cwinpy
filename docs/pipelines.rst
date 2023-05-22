##################
Analysis pipelines
##################

Background information about the analysis pipelines provided by CWInPy, and instructions/examples
for running them, are described in the following sections:

.. toctree::
   :maxdepth: 2

   Processing raw data aka "Heterodyning" <heterodyne/heterodyne>
   Bayesian parameter estimation <pe/pe>
   Known pulsar pipeline aka "knope" <knope/knope>
   Sky-shifting <skyshifting>

Quick start
~~~~~~~~~~~

The data processing stage and the full analysis pipeline can be set up and run using simple
one-liner commands. For example, running an analysis on `PSR J0737-3039A
<https://en.wikipedia.org/wiki/PSR_J0737%E2%88%923039>`_ using LIGO data from the first observing
run in the advanced detector era, `O1 <https://gwosc.org/O1/>`__, can be performed
with:

.. code-block:: bash

   $ cwinpy_knope_pipeline --run O1 --pulsar J0737-3039A

.. note::

   This will use default settings for all parts of the pipeline including outputting results in the
   current working directory. It also assumes you are running on a submit node of a machine running
   the `HTCondor <https://htcondor.readthedocs.io/>`__ job scheduler and have access to `open
   gravitational-wave data <https://gwosc.org/data/>`_ via `CVMFS
   <https://gwosc.org/cvmfs/>`__. See the :ref:`Known pulsar analysis pipeline` and
   :ref:`Heterodyning data` sections for more details.

Source parameter specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CWInPy relies on having information about a source specified using a Tempo(2)-style [1]_ pulsar
parameter file. The specifications of such a file can be found in the `Tempo2 manual
<https://www.jb.man.ac.uk/~pulsar/Resources/tempo2_manual.pdf>`__ [2]_. For CWInPy usage the file
should contain at least the following parameter keys:

* ``PSRJ``: the source's name in the `J2000 <https://en.wikipedia.org/wiki/Epoch_(astronomy)#Julian_years_and_J2000>`__ epoch coordinate system
* ``RAJ``: the source's `right ascension <https://en.wikipedia.org/wiki/Right_ascension>`__ in the format ``HH:MM:SS.S``
* ``DECJ``: the source's `declination <https://en.wikipedia.org/wiki/Declination>`__ in the format ``DD:MM:SS.S``
* ``F0``: the source's rotation frequency in Hz

in a set of key-value pairs. An example would be:

.. code:: text

   PSRJ J0534+2200
   RAJ  05:34:23.85482
   DECJ 22:00:12.95856
   F0   29.6643629353

The amount of whitespace between keys and values is not important.

IGWN Cluster usage
~~~~~~~~~~~~~~~~~~

If working on an `IGWN Computing Grid <https://computing.docs.ligo.org/guide/grid/>`__ cluster there
are certain HTCondor job values that `must be set
<https://computing.docs.ligo.org/guide/condor/tutorial/#how-to-describe-a-job>`__. These are:

* ``accounting_group``: a tag indicating the analysis that is being performed (for accounting
  purposes). Information on valid `accounting tags
  <https://computing.docs.ligo.org/guide/condor/accounting/>`__ can be found at
  `accounting.ligo.org/user <https://accounting.ligo.org/user>`__. In general, if using CWInPy, the
  "Search group" should be *Continuous Wave Group* (``cw``) and the "Search Pipeline" should be
  *Targeted Searches: Bayesian pipeline* (``targeted.bayesian``). An example tag for a production search
  on O3 data would be: ``ligo.prod.o3.cw.targeted.bayesian``.
* ``request_memory``: an upper limit on the amount of RAM required for a job to successfully complete.
* ``request_disk``: an upper limit on the amount of disk space required for a job to successfully
  complete. If transferring input files, this must include the space required for those input files
  as well as any products produced by the job.

For the ``request_memory`` and ``request_disk`` values, defaults values are used within CWInPy if
these are not explicitly set in configuration files or if using "Quick setup" options. In most
cases, these default values should work well, so user supplied values are not necessary.

Running on the Open Science Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pipelines within CWInPy can be run on the `Open Science Grid <https://opensciencegrid.org/>`__
(OSG) provided you have access to a HTCondor submit machine that has access to the OSG (see `here
<https://computing.docs.ligo.org/guide/condor/submission/#submithosts>`__ for OSG submit host for
use within the LIGO-Virgo-KAGRA Collaboration). To run CWInPy pipelines on the OSG you will need to
setup the jobs (or point at CWInPy executables) within a `CVMFS-hosted IGWN conda environment
<https://computing.docs.ligo.org/conda/>`__, or use the Singularity image containing the latest
development version of CWInPy (this should only be used for testing purposes). To do the former, you
should make sure to activate an IGWN conda distribution, e.g.,:

>>> conda activate igwn-py38

and within your pipeline configuration file the following flag should be set

.. code-block::

   osg = True

in the ``[*_dag]`` section (where the ``*`` might be ``knope``, ``heterodyne`` or ``pe`` depending
on the pipeline). If running a pipeline using "Quick setup" arguments, then the ``--osg`` flag can
be used instead of providing a configuration file.

To use the Singularity container, in addition to setting ``osg = True``, the configuration file must
have:

.. code-block::

   singularity = True

in the same ``[*_dag]`` section.

Local use of HTCondor
~~~~~~~~~~~~~~~~~~~~~

The pipelines described in the links above can be used to generate `HTCondor
<https://htcondor.readthedocs.io/>`__ DAGs to run analyses on a pool of machines across a computer
cluster. However, if you do not have access to a computer cluster and still want to run the
HTCondor-DAG-producing pipelines on a single Linux machine (preferably one with multiple cores and a
reasonable amount of memory), you can install HTCondor on your machine. Instructions on installing
HTCondor, which by default configures HTCondor to run on a single node, are documented
`here <https://htcondor.readthedocs.io/en/latest/getting-htcondor/index.html>`__. It is also
available to install in a conda environment through
`conda-forge <https://anaconda.org/conda-forge/htcondor>`__.
    
To allow the Condor jobs to start running immediately rather than waiting for your computer to be idle
you can create a ``/etc/condor/condor_config`` file containing (see the "Test-job
Policy Example" section below `here
<https://htcondor.readthedocs.io/en/latest/admin-manual/policy-configuration.html#examples-of-policy-configuration>`__):

.. code:: bash

   START      = ($(START)) || Owner == "username"
   SUSPEND    = ($(SUSPEND)) && Owner != "username"
   CONTINUE   = $(CONTINUE)
   PREEMPT    = ($(PREEMPT)) && Owner != "username"
   KILL       = $(KILL)

where ``username`` is your username, or that of the user who's jobs will be run. Then run:

>>> sudo condor_restart

Pipeline references
~~~~~~~~~~~~~~~~~~~

.. [1] `G. B. Hobbs, R. T. Edwards & R. N. Manchester <https://ui.adsabs.harvard.edu/abs/2006MNRAS.369..655H/abstract>`_,
   *MNRAS*, **369**, p. 655-672 (2006)
.. [2] G. Hobbs & R. Edwards, `TEMPO2 user manual <https://www.jb.man.ac.uk/~pulsar/Resources/tempo2_manual.pdf>`_,
   v2.0.
