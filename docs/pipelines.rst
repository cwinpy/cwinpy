##################
Analysis pipelines
##################

Background information about the analysis pipelines and instructions/examples for running them are
described in the following sections:

.. toctree::
   :maxdepth: 2

   Processing raw data aka "Heterodyning" <heterodyne/heterodyne>
   Bayesian parameter estimation <pe/pe>
   Known pulsar pipeline aka "knope" <knope/knope>

Quick start
~~~~~~~~~~~

The data processing stage and full analysis pipeline can be setup and run using simple one line
commands. For example, running an analysis on `PSR J0737-3039A
<https://en.wikipedia.org/wiki/PSR_J0737%E2%88%923039>`_ using LIGO data from the first observing
run in the advanced detector era, O1, can be performed with:

>>> cwinpy_knope_dag --run O1 --pulsar J0737-3039A

.. note::

   This will use default settings for all parts of the pipeline including outputting results in the
   current working directory. It also assumes you are running on a submit node of a machine running
   the `HTCondor <https://htcondor.readthedocs.io/>`_ job scheduler and have access to `open
   gravitational-wave data <https://www.gw-openscience.org/data/>`_ via `CVMFS
   <https://www.gw-openscience.org/cvmfs/>`_. See the <knope/knope> and <heterodyne/heterodyne>
   sections for more details.
