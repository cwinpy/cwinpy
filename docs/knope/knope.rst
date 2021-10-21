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

These stages can be run separately via command line executables, or via their Python APIs, but can
be run together using the interface documented here. Comprehensive documentation for the
:ref:`heterodyne<Heterodyning data>` and :ref:`parameter estimation<Known pulsar parameter
estimation>` is provided elsewhere on this site and will be required for running the full pipeline.

Running the *knope* analysis
----------------------------

CWInPy has two command line executables for running *knope*: ``cwinpy_knope`` and 
``cwinpy_knope_dag``. The former will run the pipeline on the machine on which it is called, while
the latter will generate an `HTCondor <https://htcondor.readthedocs.io/en/latest/>`_ DAG to run the analysis across multiple machines. For example,
users of CWInPy within the LVK can run the DAG on collaboration computer clusters. It is also
possible to run the analysis via the `Open Science Grid <https://opensciencegrid.org/>`_ if you
have access to an appropriate submit machine. In addition to the command line executables, the same
results can be using the Python API with the :func:`cwinpy.knope.knope` and
:func:`cwinpy.knope.knope_dag` functions, respectively.

It is highly recommended to run this analysis using ``cwinpy_knope_dag`` (instructions for setting
up `HTCondor <https://htcondor.readthedocs.io/en/latest/>`_ on your own *Linux* machine are provided :ref:`here<Local use of HTCondor>`) and the
instructions here will focus on that method. A brief description of using ``cwinpy_knope`` will be
provided, although this should primarily be used, if required, for quick testing.

For LVK users, if running on proprietary data you may need to generate a proxy certificate to allow
the analysis scripts to access frame files, e.g.,:

.. code:: bash

   ligo-proxy-init -p albert.einstein

In many of the examples below we will assume that you are able to access the open LIGO and Virgo
data available from the `GWOSC <https://www.gw-openscience.org/>`_ via `CVMFS
<https://cvmfs.readthedocs.io/>`_. To find out more about accessing this data see the instructions
`here <https://www.gw-openscience.org/cvmfs/>`_.

Configuration file
------------------

The primary way to run the pipeline is by supplying the ``cwinpy_knope_dag`` executable with a
configuration file. This configuration file will be a concatenation of the configurations that are
separately required for the :ref:`data processing<Heterodyning data>` using
``cwinpy_heterodyne_dag`` and the :ref:`parameter estimation<Known pulsar parameter
estimation>` using ``cwinpy_pe_dag``. You should consult those parts of the documentation for more
detail on the input parameters in each case.

.. note::

   In the case of the ``cwinpy_knope_dag`` configuration, the inputs to the parameter estimation
   stage (pulsar parameter files and heterodyned data files) will be set automatically from the
   outputs of the data processing stage.

   Any parameters used for simulated data in the parameter estimation ``[pe]`` part will also be ignored.

An example configuration file, with inline comments describing the inputs, is given below:

.. literalinclude:: cwinpy_knope_dag.ini



Quick setup
===========

The ``cwinpy_knope_dag`` script has some quick setup options that allow an analysis to be
launched in one line without the need to define a configuration file. These options **require** that
the machine/cluster that you are running HTCondor on has access to open data from GWOSC available
via CVMFS. It is also recommended that you run CWInPy from within an `IGWN conda environment
<https://computing.docs.ligo.org/conda/>`_ 

For example, if you have a TEMPO(2)-style pulsar parameter file, e.g., ``J0740+6620.par``, and you
want to analyse the open `O1 data <https://www.gw-openscience.org/O1/>`_ for the two LIGO detectors
you can simply run:

.. code-block:: bash

   cwinpy_knope_dag --run O1 --pulsar J0740+6620.par --output /home/usr/O1

where ``/home/usr/O1`` is the name of the directory where the run information and
results will be stored (if you don't specify an ``--output`` then the current working directory will
be used). This command will automatically submit the HTCondor DAG for the job. To specify multiple
pulsars you can use the ``--pulsar`` option multiple times. If you do not have a parameter file
for a pulsar you can instead use the ephemeris given by the `ATNF pulsar catalogue
<https://www.atnf.csiro.au/research/pulsar/psrcat/>`_. To do this you need to instead supply a
pulsar name (as recognised by the catalogue), for example, to run the analysis using O2 data for the
pulsar `J0737-3039A <https://en.wikipedia.org/wiki/PSR_J0737%E2%88%923039>`_ you could do:

.. code-block:: bash

   cwinpy_knope_dag --run O2 --pulsar J0737-3039A --output /home/usr/O2

Internally the ephemeris information is obtained using the :class:`~psrqpy.search.QueryATNF` class
from `psrqpy <https://psrqpy.readthedocs.io/en/latest/>`_.

CWInPy also contains information on the continuous :ref:`hardware injections<Hardware Injections>`
performed in each run, so if you wanted the analyse the these in, say, the LIGO `sixth science run
<https://www.gw-openscience.org/archive/S6/>`_, you could do:

.. code-block:: bash

   cwinpy_knope_dag --run S6 --hwinj --output /home/usr/hwinjections

Other command line arguments for ``cwinpy_knope_dag``, e.g., for setting specific detectors,
can be found :ref:`below<knope Command line arguments>`. If running on a LIGO Scientific
Collaboration cluster the ``--accounting-group-tag`` flag must be set to a valid `accounting tag
<https://accounting.ligo.org/user>`_, e.g.,:

.. code-block:: bash

   cwinpy_knope_dag --run O1 --hwinj --output /home/user/O1injections --accounting-group-tag ligo.prod.o1.cw.targeted.bayesian

.. note::

   The quick setup will only be able to use default parameter values for the heterodyne and parameter
   estimation. For "production" analyses, or if you want more control over the parameters, it is
   recommended that you use a configuration file to set up the run.

   The frame data used by the quick setup defaults to that with a 4096 Hz sample rate. However, if
   analysing sources with frequencies above about 1.6 kHz this should be switched, using the
   ``--samplerate`` flag to using the 16 kHz sampled data. By default, if analysing hardware
   injections for any of the advanced detector runs the 16 kHz data will be used due to frequency of
   the fastest pulsar being above 1.6 kHz.

.. _knope Command line arguments:

*knope* Command line arguments
------------------------------

The command line arguments for ``cwinpy_knope`` (as extracted using ``cwinpy_knope --help``) are
given below:

.. literalinclude:: knope_help.txt
   :language: none

The command line arguments for ``cwinpy_knope_dag`` (as extracted using
``cwinpy_knope_dag --help``) are:

.. literalinclude:: knope_dag_help.txt
   :language: none

.. _knope API:

Knope API
---------

.. automodule:: cwinpy.knope
   :members: knope, knope_dag