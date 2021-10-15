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

Quick start
~~~~~~~~~~~

The data processing stage and full analysis pipeline can be set up and run using simple one-liner
commands. For example, running an analysis on `PSR J0737-3039A
<https://en.wikipedia.org/wiki/PSR_J0737%E2%88%923039>`_ using LIGO data from the first observing
run in the advanced detector era, O1, can be performed with:

>>> cwinpy_knope_dag --run O1 --pulsar J0737-3039A

.. note::

   This will use default settings for all parts of the pipeline including outputting results in the
   current working directory. It also assumes you are running on a submit node of a machine running
   the `HTCondor <https://htcondor.readthedocs.io/>`_ job scheduler and have access to `open
   gravitational-wave data <https://www.gw-openscience.org/data/>`_ via `CVMFS
   <https://www.gw-openscience.org/cvmfs/>`_. See the :ref:`Known pulsar analysis pipeline` and
   :ref:`Heterodyning data` sections for more details.

Local use of HTCondor
~~~~~~~~~~~~~~~~~~~~~

The pipelines described in the links above can be used to generate `HTCondor
<https://htcondor.readthedocs.io/>`_ DAGs to run analyses on a pool of machines across a computer
cluster. However, if you do not have access to a computer cluster and still want to run the
HTCondor-DAG-producing pipelines on a single Linux machine (preferably one with multiple cores and a
reasonable amount of memory), you can install HTCondor on your machine. It is recommended to install
``mini[ht]condor``, which configures HTCondor to run on a single node as documented
`here <https://research.cs.wisc.edu/htcondor/instructions/>`_.
    
To allow the Condor jobs to start running immediately rather than waiting for your computer to be idle
you can create a ``/etc/condor/condor_config.local`` file containing (see the "Test-job
Policy Example" section below `here
<https://htcondor.readthedocs.io/en/latest/admin-manual/policy-configuration.html#examples-of-policy-configuration>`_):

.. code:: bash

   START      = ($(START)) || Owner == "username"
   SUSPEND    = ($(SUSPEND)) && Owner != "username"
   CONTINUE   = $(CONTINUE)
   PREEMPT    = ($(PREEMPT)) && Owner != "username"
   KILL       = $(KILL)

where ``username`` is your username, or that of the user who's jobs will be run. Then run:

>>> sudo condor_restart
