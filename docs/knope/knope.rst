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
``cwinpy_knope_pipeline``. The former will run the pipeline on the machine on which it is called,
while the latter will generate an `HTCondor <https://htcondor.readthedocs.io/en/latest/>`__ DAG to
run the analysis across multiple machines. For example, users of CWInPy within the LVK can run the
DAG on collaboration computer clusters. It is also possible to run the analysis via the `Open
Science Grid <https://opensciencegrid.org/>`_ if you have access to an appropriate submit machine.
In addition to the command line executables, the same results can be using the Python API with the
:func:`cwinpy.knope.knope` and :func:`cwinpy.knope.knope_pipeline` functions, respectively.

It is highly recommended to run this analysis using ``cwinpy_knope_pipeline`` (instructions for
setting up `HTCondor <https://htcondor.readthedocs.io/en/latest/>`__ on your own *Linux* machine are
provided :ref:`here<Local use of HTCondor>`) and the instructions here will focus on that method. A
brief description of using ``cwinpy_knope`` will be provided, although this should primarily be
used, if required, for quick testing.

.. note::

   For LVK users, if requiring access to `proprietary IGWN frame data
   <https://computing.docs.ligo.org/guide/auth/scitokens/#data>`__ (i.e., non-public frames visible
   to those within the LVK collaboration) via `CVMFS
   <https://computing.docs.ligo.org/guide/cvmfs/>`__ rather than frames locally stored on a cluster,
   you will need to `generate a SciToken
   <https://computing.docs.ligo.org/guide/auth/scitokens/#get>`__ to allow the analysis script to
   access them. To enable the pipeline script to find science segments and frame URLs you must
   `generate a token <https://computing.docs.ligo.org/guide/auth/scitokens/#get-default>`__ with:

   .. code:: bash

      htgettoken -a vault.ligo.org -i igwn

   .. warning::

      On some systems still using the `X.509 authentication
      <https://computing.docs.ligo.org/guide/auth/x509/>`__ you may instead need to run

      .. code:: bash

         ligo-proxy-init -p albert.einstein

      to find the science segments and frame URLs.

   and then run:

   .. code:: bash

      condor_vault_storer -v igwn

   to allow the HTCondor jobs to access the credentials.

In many of the examples below we will assume that you are able to access the open LIGO and Virgo
data available from the `GWOSC <https://gwosc.org/>`__ via `CVMFS
<https://cvmfs.readthedocs.io/>`__. To find out more about accessing this data see the instructions
`here <https://gwosc.org/cvmfs/>`__.

Configuration file
------------------

The primary way to run the pipeline is by supplying the ``cwinpy_knope_pipeline`` executable with a
configuration file. This configuration file will be a concatenation of the configurations that are
separately required for the :ref:`data processing<Heterodyning data>` using
``cwinpy_heterodyne_pipeline`` and the :ref:`parameter estimation<Known pulsar parameter
estimation>` using ``cwinpy_pe_pipeline``. You should consult those parts of the documentation for
more detail on the input parameters in each case.

.. note::

   In the case of the ``cwinpy_knope_pipeline`` configuration, the inputs to the parameter
   estimation stage (pulsar parameter files and heterodyned data files) will be set automatically
   from the outputs of the data processing stage.

   Any parameters used for simulated data in the parameter estimation ``[pe]`` part will also be ignored.

An example configuration file, with inline comments describing the inputs, is given below:

.. literalinclude:: cwinpy_knope_pipeline.ini

Quick setup
===========

The ``cwinpy_knope_pipeline`` script has some quick setup options that allow an analysis to be
launched in one line without the need to define a configuration file. These options **require** that
the machine/cluster that you are running HTCondor on has access to open data from GWOSC available
via CVMFS. It is also recommended that you run CWInPy from within an `IGWN conda environment
<https://computing.docs.ligo.org/conda/>`_ 

For example, if you have a TEMPO(2)-style pulsar parameter file, e.g., ``J0740+6620.par``, and you
want to analyse the open `O1 data <https://gwosc.org/O1/>`_ for the two LIGO detectors
you can simply run:

.. code-block:: bash

   cwinpy_knope_pipeline --run O1 --pulsar J0740+6620.par --output /home/usr/O1

where ``/home/usr/O1`` is the name of the directory where the run information and
results will be stored (if you don't specify an ``--output`` then the current working directory will
be used). This command will automatically submit the HTCondor DAG for the job. To specify multiple
pulsars you can use the ``--pulsar`` option multiple times. If you do not have a parameter file
for a pulsar you can instead use the ephemeris given by the `ATNF pulsar catalogue
<https://www.atnf.csiro.au/research/pulsar/psrcat/>`_. To do this you need to instead supply a
pulsar name (as recognised by the catalogue), for example, to run the analysis using O2 data for the
pulsar `J0737-3039A <https://en.wikipedia.org/wiki/PSR_J0737%E2%88%923039>`_ you could do:

.. code-block:: bash

   cwinpy_knope_pipeline --run O2 --pulsar J0737-3039A --output /home/usr/O2

Internally the ephemeris information is obtained using the :class:`~psrqpy.search.QueryATNF` class
from `psrqpy <https://psrqpy.readthedocs.io/en/latest/>`_.

.. note::

   Any pulsar parameter files extracted from the ATNF pulsar catalogue will be held in a
   subdirectory of the directory from which you invoked the ``cwinpy_knope_pipeline`` script, called
   ``atnf_pulsars``. If such a directory already exists and contains the parameter file for a
   requested pulsar, then that file will be used rather than being reacquired from the catalogue. To
   ensure files are up-to-date, clear out the ``atnf_pulsars`` directory as required.

CWInPy also contains information on the continuous :ref:`hardware injections<Hardware Injections>`
performed in each run, so if you wanted the analyse the these in, say, the LIGO `sixth science run
<https://gwosc.org/archive/S6/>`_, you could do:

.. code-block:: bash

   cwinpy_knope_pipeline --run S6 --hwinj --output /home/usr/hwinjections

Other command line arguments for ``cwinpy_knope_pipeline``, e.g., for setting specific detectors,
can be found :ref:`below<knope Command line arguments>`. If running on a LIGO Scientific
Collaboration cluster the ``--accounting-group`` flag must be set to a valid `accounting tag
<https://accounting.ligo.org/user>`_, e.g.,:

.. code-block:: bash

   cwinpy_knope_pipeline --run O1 --hwinj --output /home/user/O1injections --accounting-group ligo.prod.o1.cw.targeted.bayesian

.. note::

   The quick setup will only be able to use default parameter values for the heterodyne and parameter
   estimation. For "production" analyses, or if you want more control over the parameters, it is
   recommended that you use a configuration file to set up the run.

   The frame data used by the quick setup defaults to that with a 4096 Hz sample rate. However, if
   analysing sources with frequencies above about 1.6 kHz this should be switched, using the
   ``--samplerate`` flag to using the 16 kHz sampled data. By default, if analysing hardware
   injections for any of the advanced detector runs the 16 kHz data will be used due to frequency of
   the fastest pulsar being above 1.6 kHz.

*knope* examples
----------------

A selection of example configuration files for using with ``cwinpy_knope_pipeline`` are shown below.
These examples are generally fairly minimal and make use of many default settings. If basing your
own configuration files on these, the various input and output directory paths should be changed.
These all assume a user ``matthew.pitkin`` running on the `ARCCA Hawk Computing Centre
<https://computing.docs.ligo.org/guide/computing-centres/hawk/>`__ or the `LDAS@Caltech
<https://computing.docs.ligo.org/guide/computing-centres/cit/>`__ cluster.

.. note::

   When running an analysis it is always worth regularly checking on the Condor jobs using `condor_q
   <https://htcondor.readthedocs.io/en/latest/man-pages/condor_q.html>`__. There is no guarantee
   that all jobs will successfully complete or that some jobs might not get `held
   <https://htcondor.readthedocs.io/en/latest/users-manual/managing-a-job.html?highlight=hold#job-in-the-hold-state>`__
   for some reason. These may require some manual intervention such as resubmitting a generated
   `rescue DAG
   <https://htcondor.readthedocs.io/en/latest/users-manual/dagman-workflows.html#rescue-dags>`__ or
   `releasing <https://htcondor.readthedocs.io/en/latest/man-pages/condor_release.html>`__ held jobs
   after diagnosing and fixing (maybe via `condor_qedit
   <https://htcondor.readthedocs.io/en/latest/man-pages/condor_qedit.html?highlight=condor_qedit>`__)
   the hold reason.

O1 LIGO (proprietary) data, single pulsar
=========================================

An example configuration file for performing a search for a single pulsar, the Crab pulsar
(J0534+2200), for a signal emitted from the :math:`l=m=2` harmonic, using proprietary O1 LIGO data
can be downloaded :download:`here <examples/knope_example_O1_ligo.ini>` and is reproduced below:

.. literalinclude:: examples/knope_example_O1_ligo.ini

This requires a pulsar parameter file for the Crab pulsar within the
``/home/matthew.pitkin/projects/O1pulsars`` directory and a prior file, called e.g.,
``J0534+2200.prior`` in the ``/home/matthew.pitkin/projects/O1priors`` directory. An example prior
file might contain:

.. code-block::

   h0 = FermiDirac(1.0e-24, mu=1.0e-22, name='h0')
   phi0 = Uniform(minimum=0.0, maximum=pi, name='phi0')
   iota = Sine(minimum=0.0, maximum=pi, name='iota')
   psi = Uniform(minimum=0.0, maximum=pi/2, name='psi')

This would be run with:

.. code-block:: bash

   $ cwinpy_knope_pipeline knope_example_O1_ligo.ini

.. note:: 

   Before running this (provided you have
   `installed the SciTokens tools <https://computing.docs.ligo.org/guide/auth/scitokens/#install>`__)
   you should run:

   .. code-block:: bash

      $ htgettoken --audience https://datafind.ligo.org --scope gwdatafind.read -a vault.ligo.org --issuer igwn

   to generate `SciTokens authentication <https://computing.docs.ligo.org/guide/auth/scitokens/>`__
   for accessing the proprietary data via CVMFS.

Once complete, this example should have generated the heterodyned data files for H1 and L1 (within
the ``/home/matthew.pitkin/projects/cwinpyO1/heterodyne`` directory):

.. code-block:: python

   from cwinpy import MultiHeterodynedData

   het = MultiHeterodynedData(
      {
          "H1": "H1/heterodyne_J0534+2200_H1_2_1126073529-1137253524.hdf5",
          "L1": "L1/heterodyne_J0534+2200_L1_2_1126072156-1137250767.hdf5",
      }
   )

   fig = het.plot(together=True, remove_outliers=True)
   fig.show()

.. thumbnail:: examples/knope_example1_bks.png
   :width: 600px
   :align: center

and the posterior samples for an analysis using both LIGO detectors (within the
``/home/matthew.pitkin/projects/cwinpyO1/results`` directory):

.. code-block:: python

   from cwinpy.plot import Plot

   plot = Plot(
      "J0534+2200/cwinpy_pe_H1L1_J0534+2200_result.hdf5",
      parameters=["h0", "iota", "psi", "phi0"].
   )
   plot.plot()

   # add 95% upper limit line
   plot.fig.get_axes()[0].axvline(
      plot.upper_limit(parameter="h0"),
      color="k",
      ls="--",
   )

   plot.fig.show()

.. thumbnail:: examples/knope_example1_pe.png
   :width: 600px
   :align: center

.. note::

   To run the example and also perform parameter estimation for each individual detector (rather
   than just the coherent multi-detector analysis), the additional option ``incoherent = True``
   should be added in the ``[pe]`` section of the configuration file.

O1 & O2 LIGO/Virgo (open) data, multiple pulsars
================================================

An example configuration file for performing a search for multiple pulsars, for a signal emitted
from the :math:`l=m=2` harmonic, using open O1 and O2 LIGO and Virgo data can be downloaded :download:`here
<examples/knope_example_O2_open.ini>` and is reproduced below:

.. literalinclude:: examples/knope_example_O2_open.ini

This requires the parameter files for the pulsars to be within the
``/home/matthew.pitkin/projects/O2pulsars`` directory (in the example results shown below this
directory contained files for the two pulsars `J0737-3039A
<https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.67&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=J0737-3039A&ephemeris=long&submit_ephemeris=Get+Ephemeris&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query>`__
and `J1843-1113
<https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.67&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=J1843-1113&ephemeris=long&submit_ephemeris=Get+Ephemeris&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query>`__).
In this example, a single prior file for all pulsars is used at the location
``/home/matthew.pitkin/projects/O2priors/priors.txt`` directory. This might contain, e.g.,

.. code-block::

   h0 = FermiDirac(1.0e-24, mu=1.0e-22, name='h0')
   phi0 = Uniform(minimum=0.0, maximum=pi, name='phi0')
   iota = Sine(minimum=0.0, maximum=pi, name='iota')
   psi = Uniform(minimum=0.0, maximum=pi/2, name='psi')

Due to using data from two runs, the start and end times of each run need to be specified (the lists
within the associated dictionaries), and appropriate frame types, channels and segment types need to
be provided for each run.

In this example, it uses TEMPO2 (via libstempo) for calculating the phase evolution of the signal.
Due to the ``incoherent = True`` value in the ``[pe]`` section, this analysis will perform parameter
estimation for each pulsar for each individual detector and also for the full multi-detector data
set.

This would be run with:

.. code-block:: bash

   $ cwinpy_knope_pipeline knope_example_O2_open.ini

Displaying the results
~~~~~~~~~~~~~~~~~~~~~~

Once the analysis is complete, a table of upper limits can be produced using the
:class:`~cwinpy.pe.peutils.UpperLimitTable` class (this requires the `cweqgen
<https://cweqgen.readthedocs.io/>`__ package to be installed as it makes used of the
:func:`cweqgen.equations.equations` function). The :class:`~cwinpy.pe.peutils.UpperLimitTable` class
is a child of an :class:`astropy.table.QTable` and retains all its attributes. We can, for example,
create a table showing the 95% upper limits on amplitude, ellipticity, mass quadrupole and the ratio
to the spin-down limits:

.. code-block:: python

   from cwinpy.pe import UpperLimitTable

   # directory containing the parameter estimation results
   resdir = "/home/matthew.pitkin/projects/cwinpyO2/results"

   tab = UpperLimitTable(
       resdir=resdir,
       ampparam="h0",  # get the limits on h0
       includeell=True,  # calculate the ellipticity limit
       includeq22=True,  # calculate the mass quadrupole limit
       includesdlim=True,  # calculate the spin-down limit and ratio
       upperlimit=0.95,  # calculate the 95% credible upper limit (the default)
   )

   # output the table in rst format, so it renders nicely in these docs!
   print(tab.table_string(format="rst"))

=========== ====== ==================== ==== =================== ================== ================== ============== =================== =================== ================== ================== ================== =================== ================== ================== ============== =================== ================== ================== ==============
       PSRJ  F0ROT                F1ROT DIST         H0_L1_95%UL       ELL_L1_95%UL       Q22_L1_95%UL SDRAT_L1_95%UL               SDLIM     H0_H1L1V1_95%UL   ELL_H1L1V1_95%UL   Q22_H1L1V1_95%UL SDRAT_H1L1V1_95%UL         H0_H1_95%UL       ELL_H1_95%UL       Q22_H1_95%UL SDRAT_H1_95%UL         H0_V1_95%UL       ELL_V1_95%UL       Q22_V1_95%UL SDRAT_V1_95%UL
=========== ====== ==================== ==== =================== ================== ================== ============== =================== =================== ================== ================== ================== =================== ================== ================== ============== =================== ================== ================== ==============
J0737-3039A  44.05 -3.41×10\ :sup:`-15` 1.10 2.52×10\ :sup:`-26` 3.37×10\ :sup:`-6` 2.61×10\ :sup:`32`            3.9 6.45×10\ :sup:`-27` 1.59×10\ :sup:`-26` 2.13×10\ :sup:`-6` 1.64×10\ :sup:`32`               2.46 1.84×10\ :sup:`-26` 2.47×10\ :sup:`-6` 1.91×10\ :sup:`32`           2.85 1.65×10\ :sup:`-25` 2.21×10\ :sup:`-5` 1.71×10\ :sup:`33`             25
 J1843-1113 541.81 -2.78×10\ :sup:`-15` 1.26 7.53×10\ :sup:`-26` 7.64×10\ :sup:`-8`  5.9×10\ :sup:`30`             51 1.45×10\ :sup:`-27` 4.52×10\ :sup:`-26` 4.59×10\ :sup:`-8` 3.54×10\ :sup:`30`                 31 1.03×10\ :sup:`-25` 1.05×10\ :sup:`-7`  8.1×10\ :sup:`30`             71 9.91×10\ :sup:`-25` 1.01×10\ :sup:`-6` 7.77×10\ :sup:`31`            683
=========== ====== ==================== ==== =================== ================== ================== ============== =================== =================== ================== ================== ================== =================== ================== ================== ============== =================== ================== ================== ==============

The :meth:`~cwinpy.pe.peutils.UpperLimitTable.plot` method of the
:class:`~cwinpy.pe.peutils.UpperLimitTable` can also be used to produce summary plots of the
results, e.g., upper limits on :math:`h_0` or ellipticity against signal frequency:

.. code-block:: python

   # get O2 amplitude spectral density estimates so we can overplot sensitivity estimates
   import requests
   import numpy as np
   O2H1asd = "O2_H1_asd.txt"
   O2H1obs = 158 * 86400  # 158 days
   np.savetxt(
       O2H1asd,
       np.array([[float(val.strip()) for val in line.split()] for line in requests.get(
           "https://dcc.ligo.org/public/0156/G1801950/001/2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt"
       ).text.strip().split("\n")]),
   )
   O2L1asd = "O2_L1_asd.txt"
   O2L1obs = 154 * 86400  # 154 days
   np.savetxt(
       O2L1asd,
       np.array([[float(val.strip()) for val in line.split()] for line in requests.get(
           "https://dcc.ligo.org/public/0156/G1801952/001/2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt"
       ).text.strip().split("\n")]),
   )

   # plot h0 (by default this will use the joint H1L1V1 upper limits)
   fig = tab.plot(
       column="h0",
       histogram=True,  # include a histogram of h0 values on rhs of plot
       showsdlim=True,  # show spin-down limits
       asds=[O2H1asd, O2L1asd],  # calculate and show sensitivity estimate
       tobs=[O2H1obs, O2L1obs],  # observing times for sensitivity estimate
   )
   fig.show()

.. thumbnail:: examples/knope_example2_h0uls.png
   :width: 600px
   :align: center

.. code-block:: python

   # plot ellipticity upper limits
   fig = tab.plot(
      column="ell",
      histogram=True,
      showsdlim=True,
      showq22=True,  # show axis including Q22 values
      showtau=True,  # add isocontour of constant characteristic age (for n=5)
   )
   fig.show()

.. thumbnail:: examples/knope_example2_elluls.png
   :width: 600px
   :align: center

.. note::

   By default, if calculating limits on the ellipticity/spin-down ratio, the
   :class:`~cwinpy.pe.peutils.UpperLimitTable` will attempt to get values for the pulsar's distance
   and (intrinsic) frequency derivative using the values found in the `ATNF Pulsar Catalogue
   <https://www.atnf.csiro.au/research/pulsar/psrcat/>`__. These values can be manually provided if
   necessary.

Searching for non-GR polarisation modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The heterodyned output of the above pipeline can be re-used to perform a search for a signal that
contains some combination of generic tensor, vector and scalar polarisation modes, i.e., a non-GR
signal. This will assume that the signals are still emitted at twice the rotation frequency of the
the pulsars. This will just use the :ref:`cwinpy_pe_pipeline<Known pulsar parameter estimation>` and
a new prior file. An example configuration file for this can be downloaded :download:`here
<examples/cwinpy_pe_example_O2_nongr.ini>` and is reproduced below:

.. literalinclude:: examples/cwinpy_pe_example_O2_nongr.ini

while a possible prior file might contain, e.g.,:

.. code-block::

   hplus = FermiDirac(1.0e-24, mu=1.0e-22, name='hplus')
   hcross = FermiDirac(1.0e-24, mu=1.0e-22, name='h0')
   phi0tensor = Uniform(minimum=0.0, maximum=2*pi, name='phi0tensor')
   psitensor = Uniform(minimum=0.0, maximum=2*pi, name='psitensor')
   hvectorx = FermiDirac(1.0e-24, mu=1.0e-22, name='hvectorx')
   hvectory = FermiDirac(1.0e-24, mu=1.0e-22, name='hvectory')
   phi0vector = Uniform(minimum=0.0, maximum=2*pi, name='phi0vector')
   psivector = Uniform(minimum=0.0, maximum=2*pi, name='psivector')
   hscalarb = FermiDirac(1.0e-24, mu=1.0e-22, name='hscalarb')
   phi0scalar = Uniform(minimum=0.0, maximum=2*pi, name='phi0scalar')
   psiscalar = Uniform(minimum=0.0, maximum=2*pi, name='psiscalar')

In this case, the prior assumes a signal containing tensor (``hplus`` and ``hcross``), vector
(``hvectorx`` and ``hvectory``) and scalar (``hscalarb``, but not ``hscalarl`` as they are
completely degenerate) amplitudes and associated initial phases and polarisations. The phases and
polarisation all cover a :math:`2\pi` radian range.

This would be run with:

.. code-block:: bash

   $ cwinpy_pe_pipeline cwinpy_pe_example_O2_nongr.ini

The analysis will take significantly longer (10s of hours) than the default analysis, which assumes
just four unknown parameters.

This could have been achieved using the original ``cwinpy_knope_pipeline`` by passing it the non-GR
prior file.

We can look at a plot for one of the pulsars showing the posteriors for all parameters for each
detector with: 

.. code-block:: python

   from cwinpy.plot import Plot

   plot = Plot(
       {
           "Joint": "J0737-3039A/cwinpy_pe_H1L1_J0737-3039A_result.hdf5",
           "H1": "J0737-3039A/cwinpy_pe_H1_J0737-3039A_result.hdf5",
           "L1": "J0737-3039A/cwinpy_pe_L1_J0737-3039A_result.hdf5",
       },
       parameters=[
           "hplus",
           "hcross",
           "hvectorx",
           "hvectory",
           "hscalarb",
           "phi0tensor",
           "phi0vector",
           "phi0scalar",
           "psitensor",
           "psivector",
           "psiscalar"
       ]
   )

   plot.plot()
   plot.fig.show()

.. thumbnail:: examples/nongr_example.png
   :width: 600px
   :align: center

O3 LIGO/Virgo (proprietary) data, dual harmonic
===============================================

An example configuration file for performing a search for multiple pulsars, for a signal with
components a both the rotation frequency and twice the rotation frequency (the :math:`l=2,m=1` and
the :math:`l=m=2` harmonics), using proprietary O3 LIGO and Virgo data can be downloaded
:download:`here <examples/knope_example_O3_dual.ini>` and is reproduced below:

.. literalinclude:: examples/knope_example_O3_dual.ini

This requires the parameters for the pulsar to be within the
``/home/matthew.pitkin/projects/O3pulsars`` directory (in the example results shown below this
directory contained files for the two pulsars `J0437-4715
<https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.67&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=J0437-4715&ephemeris=long&submit_ephemeris=Get+Ephemeris&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query>`__
and `J0771-6830
<https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.67&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=J0711-6830&ephemeris=long&submit_ephemeris=Get+Ephemeris&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query>`__).
In this example the default priors and used, so no prior file is explicitly given.

The pulsar parameter files used in the example were generated using the DE436 solar system
ephemeris, which is not one of the default files available within LALSuite. Therefore, these have
had to be generated using the commands:

.. code-block:: bash

   $ lalpulsar_create_solar_system_ephemeris_python --target SUN --year-start 2000 --interval 20 --num-years 40 --ephemeris DE436 --output-file sun00-40-DE436.dat
   $ lalpulsar_create_solar_system_ephemeris_python --target EARTH --year-start 2000 --interval 2 --num-years 40 --ephemeris DE436 --output-file earth00-40-DE436.dat

and then gzipped. Alternatively, the ``usetempo2`` option could be used, where the latest TEMPO2 version
will contain these ephemerides.

This would be run with:

.. code-block:: bash

   $ cwinpy_knope_pipeline knope_example_O3_dual.ini

.. note::

   If running on open GWOSC data, the O3a and O3b periods need to be treated as separate runs as in
   the :ref:`above example<O1 & O2 LIGO/Virgo (open) data, multiple pulsars>`. This is automatically
   dealt with if setting up a ``cwinpy_knope_pipeline`` run using the :ref:`"Quick setup"<Quick setup>`.

Once complete, this example should have generated the heterodyned data files for H1, L1 and V1
(within the ``/home/matthew.pitkin/projects/cwinpyO3/heterodyne`` directory) for the signals at both
once and twice the pulsar rotation frequency.

We could look at the heterodyned data at the rotation frequency for the pulsar J0437-4715 using:

.. code-block:: python

   from cwinpy import MultiHeterodynedData

   het = MultiHeterodynedData(
      {
          "H1": "H1/heterodyne_J0437-4715_H1_1_1238166018-1269363618.hdf5",
          "L1": "L1/heterodyne_J0437-4715_L1_1_1238166018-1269363615.hdf5",
          "V1": "V1/heterodyne_J0437-4715_V1_1_1238166018-1269363618.hdf5",
      }
   )

   fig = het.plot(together=True, remove_outliers=True, markersize=2)
   fig.show()

.. thumbnail:: examples/knope_example3_bks.png
   :width: 600px
   :align: center

The results of the parameter estimation stage would be found in the
``/home/matthew.pitkin/projects/cwinpyO3/results`` directory. Posteriors parameter distributions for
all the parameters for J0437-4715, for each detector and the joint detector analysis, could be plotted
using:

.. code-block:: python

   from cwinpy.plot import Plot

   plot = Plot(
       {
           "Joint": "J0437-4715/cwinpy_pe_H1L1V1_J0437-4715_result.hdf5",
           "H1": "J0437-4715/cwinpy_pe_H1_J0437-4715_result.hdf5",
           "L1": "J0437-4715/cwinpy_pe_L1_J0437-4715_result.hdf5",
           "V1": "J0437-4715/cwinpy_pe_V1_J0437-4715_result.hdf5",
       },
       parameters=[
           "c21",
           "c22",
           "phi21",
           "phi22",
           "iota",
           "psi",
       ]
   )

   plot.plot()
   plot.fig.show()

.. thumbnail:: examples/knope_example3_pe.png
   :width: 600px
   :align: center

O2 LIGO (open) data, "transient-continuous" signal following a pulsar glitch
============================================================================

As described in :ref:`this example<Example: a simulated transient-continuous signal>`, the signal
model used by CWInPy can be one that is described as "transient-continuous" (see., e.g., [1]_ and
[2]_). Unlike the standard signal that is assumed to be "always on", the *transient-continuous*
signal is modulated by a window function; currently this can consist of a rectangular window, i.e.,
the signal abruptly turns of and then off, or an exponential window, i.e., the signal turns on and
then the amplitude decays away exponentially. Such a signal might occur after an event such as a
`pulsar glitch <https://en.wikipedia.org/wiki/Glitch_(astronomy)>`__, which could "raise a mountain"
on the star that subsequently decays away on a timescale of days-to-weeks-to-months [3]_.

An example configuration file for performing a search for such a signal is given below. In this case
it uses open LIGO data from the `O2 observing run <https://gwosc.org/O2/>`__ to search
for an exponentially decaying signal following a glitch observed in the Vela pulsar on 12 December
2016 [4]_.

.. literalinclude:: examples/knope_example_O2_vela_transient.ini

Unlike in the :ref:`O2 example above<O1 & O2 LIGO/Virgo (open) data, multiple pulsars>`, this does
not use data from the Virgo detector as that only covers the end of the O2 run, which will be well
past the assumed transient timescale. This example assumes that you have a file (``vela.par``)
containing the ephemeris for the Vela pulsar from radio observations overlapping the O2 run. To use
the *transient-continuous* model, this file needs to contain a line defining the transient window
type, in this case an exponential:

.. code-block:: none

   TRANSIENTWINDOWTYPE EXP

The example also needs a prior file (``vela_transient_prior.txt``) which contains the priors on the
parameters of the window: the start time (taken to be close to the observed glitch time) and the
decay time constant. If following the analysis of [2]_, the prior on the signal start time can be
±0.5 days either side of the observed glitch time (this is actually much larger than the uncertainty
on the glitch time), which is at an MJD of 57734.4849906 (or GPS(TT) time of 1165577852). The signal
decay time here is set from to be from 1.5 days to 150. An example of such a prior to use is:

.. code-block::

   h0 = FermiDirac(1.0e-24, mu=1.0e-22, name='h0', latex_label='$h_0$')
   phi0 = Uniform(minimum=0.0, maximum=pi, name='phi0', latex_label='$\phi_0$', unit='rad')
   iota = Sine(minimum=0.0, maximum=pi, name='iota', latex_label='$\iota$', unit='rad')
   psi = Uniform(name='psi', minimum=0, maximum=np.pi / 2, latex_label='$\psi$', unit='rad')
   transientstarttime = Uniform(name='transientstarttime', minimum=57733.9849906, maximum=57734.9849906, latex_label='$t_0$', unit='d')
   transienttau = Uniform(name='transienttau', minimum=1.5, maximum=150, latex_label='$\\tau$', unit='d')

.. note::

   If setting the ``transientstarttime`` and ``transienttau`` prior values using MJD and days,
   respectively, the ``unit`` keyword of the prior must be set to ``'d'``, otherwise the start time
   will be assumed to be a GPS time and the decay timescale will be assumed to be in seconds.

This would be run with:

.. code-block:: bash

   $ cwinpy_knope_pipeline knope_example_O2_vela_transient.ini

The results of the parameter estimation stage would be found in the
``/home/matthew.pitkin/projects/vela/results`` directory. Posteriors parameter distributions for all
the parameters, including the transient parameters, for each detector and the joint detector
analysis, could be plotted using:

.. code-block:: python

   from cwinpy.plot import Plot

   plot = Plot(
       {
           "Joint": "J0835-4510/cwinpy_pe_H1L1_J0835-4510_result.hdf5",
           "H1": "J0835-4510/cwinpy_pe_H1_J0835-4510_result.hdf5",
           "L1": "J0835-4510/cwinpy_pe_L1_J0835-4510_result.hdf5",
       },
       parameters=[
           "h0",
           "phi0",
           "iota",
           "psi",
           "transientstarttime",
           "transienttau",
       ]
   )

   plot.plot()
   plot.fig.show()

   # output 95% upper limits on h0
   plot.upper_limit("h0", 0.95)

.. thumbnail:: examples/knope_example4_pe.png
   :width: 600px
   :align: center

The 95% credible upper limits on :math:`h_0` in this case are:

* H1: :math:`6.8\!\times\!10^{-24}`
* L1: :math:`4.1\!\times\!10^{-24}`
* Joint (H1 & L1): :math:`5.3\!\times\!10^{-24}`

It is interesting to see that there is a strong correlation between :math:`h_0` and the transient
decay time constant :math:`\tau_{\rm{trans}}`, with a broader distribution of :math:`h_0` associated
with smaller values of :math:`\tau_{\rm{trans}}`. This is expected, as you would expect to be more
sensitive to longer signals.

Generating results webpages
---------------------------

If you have run the ``cwinpy_knope_pipeline``, it will generate a version of the input
:ref:`configuration file<Configuration file>` within the analysis directory called
``knope_pipeline_config.ini``. After the analysis is complete, i.e., the HTCondor DAG has
successfully run, this configuration file can be used as an input for generating a set of results
pages. These pages (based on `PESummary <https://docs.ligo.org/lscsoft/pesummary/>`_ [5]_) will
display summary tables and plots of the results for all sources, e.g., upper limits on the
gravitational-wave amplitude, as well as pages with information and posterior plots for each
individual pulsar.

These pages can be generated using the ``cwinpy_generate_summary_pages`` command line interface
script or the :func:`~cwinpy.pe.summary.generate_summary_pages` API, which give control over what
information is generated and output.

For example, to generate summary webpages that include a table of upper limits on gravitational-wave
amplitude, plots of these limits (also after converting to equivalent limits on the each pulsar's
ellipticity and mass quadrupole moment), estimates of the signal-to-noise ratio and Bayesian odds,
and individual pages for each pulsar, one could just run:

.. code-block:: bash

   $ cwinpy_generate_summary_pages --outpath /home/albert.einstein/public_html/analysis \
   --url https://ldas-jobs.ligo.caltech.edu/~albert.einstein/analysis \
   knope_pipeline_config.ini

where ``--outpath`` points to the path of a web-accessible directory and ``--url`` gives the
associated URL of that directory (for URLs for the user webpage for each LDG cluster, see the
appropriate cluster listed `here <https://computing.docs.ligo.org/guide/computing-centres/ldg/>`__
and look for the "User webspace"; currently the URL is not actually used, so a dummy string can be
passed). If you have many sources, this could takes tens of minutes to run.

.. warning::

   The webpage generation scripts are currently not included in the CWInPy test suite and as such
   the interface may be buggy.

.. _knope Command line arguments:

*knope* Command line arguments
------------------------------

The command line arguments for ``cwinpy_knope`` can be found using:

.. command-output:: cwinpy_knope --help

The command line arguments for ``cwinpy_knope_pipeline`` can be found using:

.. command-output:: cwinpy_knope_pipeline --help

Knope API
---------

.. automodule:: cwinpy.knope
   :members: knope, knope_pipeline
.. automodule:: cwinpy.pe.summary
   :members: generate_summary_pages

Knope references
----------------

.. [1] `R. Prix, S. Giampanis, S. & C. Messenger, C. <https://ui.adsabs.harvard.edu/abs/2011PhRvD..84b3007P/abstract>`_,
   *PRD*, **84**, 023007 (2011)
.. [2] `D. Keitel et al. <https://ui.adsabs.harvard.edu/abs/2019PhRvD.100f4058K/abstract>`_,
   *PRD*, **100**, 064058 (2019)
.. [3] `G. Yim & D. I. Jones <https://ui.adsabs.harvard.edu/abs/2020MNRAS.498.3138Y/abstract>`_,
   *MNRAS*, **498**, 3138-3152 (2020)
.. [4] `J. Palfreyman <https://ui.adsabs.harvard.edu/abs/2016ATel.9847....1P/abstract>`_,
   *Astron. Telegram*, **9847** (2016)
.. [5] `C. Hoy & V. Raymond <https://ui.adsabs.harvard.edu/abs/2021SoftX..1500765H/abstract>`_,
   *SoftwareX*, **15**, 100765 (2021)
