##############################
Known pulsar analysis pipeline
##############################

CWInPy can be used to run an end-to-end pipeline to search for gravitational waves from known
pulsars in data from ground-based gravitational-wave detectors. This "known pulsar pipeline" is
known here as "*knope*".

The pipeline has two main stages:

#. data preprocessing involving heterodyning, filtering and down-sampling raw gravitational-wave
   detector strain time-series;
#. using the processed data to estimate the posterior probability
   distributions of the unknown gravitational-wave signal parameter via Bayesian inference.

These stages can be run seperately via command line executables, or via their Python APIs, but can
be run together using the interface documented here. Comprehensive documentation for the
:ref:`heterodyne<Heterodyning data>` and :ref:`parameter estimation<Known pulsar parameter
estimation>` is provided elsewhere on this site and will be required for running the full pipeline.

Running the *knope* analysis
----------------------------

CWInPy has two command line executables for running *knope*: ``cwinpy_knope`` and 
``cwinpy_knope_dag``. The former will run the pipeline on the machine on which it is called, while
the latter will generate a HTCondor DAG to run the analysis across multiple machines. For example,
users of CWInPy within the LVK can run the DAG on collaboration computer clusters. It is also
possible to run the analysis via the `Open Science Grid <https://opensciencegrid.org/>`_ if you
have access to an appropriate submit machine. In addition to the command line executables, the same
results can be using the Python API with the :func:`cwinpy.knope.knope` and
:func:`cwinpy.knope.knope_dag` functions, respectively.

It is highly recommended to run this analysis using ``cwinpy_knope_dag`` (instructions for setting
up HTCondor on your own Linux machine are provided :ref:`here<Local use of HTCondor>`) and the
instructions here will focus on that method. A brief description of using ``cwinpy_knope`` will be
provided, although this should primarily be used, if required, for quick testing.

For LVK users, if running on proprietory data you may need to generate a proxy certificate to allow
the analysis script to access frame files, e.g.,:

.. code:: bash

   ligo-proxy-init -p albert.einstein

In many of the examples below we will assume that you are able to access the open LIGO and Virgo
data available from the `GWOSC <https://www.gw-openscience.org/>`_ via `CVMFS
<https://cvmfs.readthedocs.io/>`_. To find out more about accessing this data see the instructions
`here <https://www.gw-openscience.org/cvmfs/>`_.

Configuration file

Knope API
---------

.. automodule:: cwinpy.knope
   :members: knope, knope_dag