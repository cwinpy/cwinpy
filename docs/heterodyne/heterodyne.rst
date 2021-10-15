#################
Heterodyning data
#################

Gravitational-wave strain data is often sampled at rates of 4096 Hz or 16384 Hz. It is only feasible
to perform :ref:`parameter estimation<Known pulsar parameter estimation>` for a continuous
gravitational-wave signal from a particular pulsar using data that is heavily downsampled compared
to this rate. As described in :ref:`Heterodyned data`, this is done using the method from [1]_. The
pulsar's phase evolution is assumed to be well described by a Taylor expansion

.. math::

   \phi(t') = 2 \pi C\left(f_0(t'-T_0) + \frac{1}{2}\dot{f}_0(t'-T_0)^2 + \dots\right)

where :math:`f_0` is the source rotation frequency, :math:`\dot{f}_0` is the first frequency
derivative, :math:`t'` is the time in a frame at rest with respect to the pulsar (the pulsar proper
time, which for isolated pulsars is assumed to be consistent with the solar system barycentre),
:math:`T_0` is the time reference epoch in the pulsar proper time, and :math:`C` is an emission
mechanism-dependent scaling factor (often set to 2, for emission from the :math:`l=m=2` quadrupole
mode). 

.. note::
   The analysis code described below can use a Taylor expansion in the phase up to arbitrarily high
   orders. The model can also include terms in addition to this expansion, such as those describing
   the evolution of the phase following a glitch (using the model described in Eqn. 1 of [2]_) and
   sinusoidal terms to whiten timing noise (using the model described in Eqn. 10 of [3]_).
  
The raw real time series of :math:`h(t)` from a detector is "heterodyned",

.. math::

   h'(t) = h(t) e^{-2\pi i \phi(t + \Delta t(t))},

to remove the high-frequency variation of the potential source signal, where :math:`\Delta t(t)` is
a time-, source- and detector-position-dependent delay term that accounts for the Doppler and
relativistic delays between the time at the detector and the pulsar proper time (see, e.g., Equation
1 of [3]_). The resulting complex series :math:`h'(t)` is low-pass filtered (effectively a band-pass
filter on the two-sided complex data) using a high order Butterworth filter with a given knee
frequency, and down-sampled, via averaging, to a far lower sample rate than the original raw data.
In general, gravitational-wave detector data is sampled at 16384 Hz, and often the heterodyned data
is downsampled to 1/60 Hz (one sample per minute).

CWInPy can be used to perform this heterodyning, taking files containing raw gravitational-wave
strain and returning the complex, filtered, down-sampled time series in a form that can be read in
as a :class:`~cwinpy.data.HeterodynedData` object. To generate the phase evolution used for the
heterodyne CWInPy can either use functions within `LALSuite
<https://lscsoft.docs.ligo.org/lalsuite/>`_ (the default) or, if installed, use the `Tempo2
<https://bitbucket.org/psrsoft/tempo2/src/master/>`_ pulsar timing package  (via the `libstempo
<https://vallis.github.io/libstempo/>`_ wrapper package).

CWInPy comes with an executable, ``cwinpy_heterodyne``, for implementing this, which closely (but
not identically) emulates the functionality from the `LALSuite
<https://lscsoft.docs.ligo.org/lalsuite/>`_ code ``lalapps_heterodyne_pulsar``.

There is also an API for running this analysis from within a Python shell or script as described
:ref:`below<heterodyne API>`.

Running the analysis
--------------------

The ``cwinpy_heterodyne`` executable and :ref:`API<heterodyne API>` can be used to process
gravitational-wave data for individual pulsars or multiple pulsars. We will cover some examples of
running analyses via use of command line arguments, a configuration file supplied to
``cwinpy_heterodyne``, or through the :ref:`API<heterodyne API>`. The current command line arguments
for ``cwinpy_heterodyne`` are given :ref:`below<heterodyne Command line arguments>`.

If running an analysis for multiple pulsars on a large stretch of data it is recommended that you
split the analysis up to run as many separate jobs. If you have access to a computer cluster (such
as those available to the LVK, or via the `Open Science Grid <https://opensciencegrid.org/>`_), or
an individual machine (see below), running the `HTCondor <https://htcondor.readthedocs.io/en/latest/>`_
job scheduler system then the analysis can be split up using the ``cwinpy_heterodyne_dag`` pipeline
script (see :ref:`Running using HTCondor`). In some cases you may need to generate a proxy
certificate to allow the analysis script to access frame files, e.g.,:

.. code:: bash

   ligo-proxy-init -p albert.einstein

In many of the examples below we will assume that you are able to access the open LIGO and Virgo
data available from the `GWOSC <https://www.gw-openscience.org/>`_ via `CVMFS
<https://cvmfs.readthedocs.io/>`_. To find out more about accessing this data see the instructions
`here <https://www.gw-openscience.org/cvmfs/>`_. If using GWOSC data sampled at 4 kHz it should be
noted that that this has a low-pass filter applied that causes a sharp drop-off above about 1.6 kHz,
which is below the Nyquist rate. Therefore, if analysing sources with gravitational-wave signal
frequencies greater than about 1.6 kHz the 16 kHz sample rate data should be used.

.. note::

   To run a heterodyne analysis as multiple jobs on your own machine with HTCondor see :ref:`Local
   use of HTCondor`.

Example: two simulated pulsar signals
=====================================

For the first example we will generate some simulated data containing signals from two (fake)
pulsars. To make the simulation manageable in terms of the amount of data and to have a quick run
time we will generate only one day of data at a sample rate of 16 Hz (the standard LIGO/Virgo sample
rate is 16384 Hz).

Generating the data
###################

To generate the data we will use the LALSuite `programme
<https://lscsoft.docs.ligo.org/lalsuite/lalapps/makefakedata__v5_8c.html>`_
``lalapps_Makefakedata_v5`` (you can skip straight to the heterodyning description
:ref:`here<Heterodyning the data>`). The two fake pulsars have parameters defined in Tempo(2)-style
parameter files (where frequencies, frequency derivatives and phases are the rotational values
rather than the gravitational-wave values), as follows:

* an isolated pulsar in a file called ``J0123+0123.par``

.. literalinclude:: examples/J0123+0123.par

* a pulsar in a binary system in a file called ``J0404-0404.par``

.. literalinclude:: examples/J0404-0404.par

One way to create the simulated data is as follows (where in this case we generate data with a very
low level of simulated noise, so that the signals can be seen prominently):

.. literalinclude:: examples/example_mfd_1.py
   :language: python

This should create the file ``H-H1_FAKEDATA-1000000000-86400.gwf`` in the gwf format.

Heterodyning the data
#####################

We will show how to heterodyne the data in ``H-H1_FAKEDATA-1000000000-86400.gwf`` for the two
different pulsars by i) using a configuration file for the ``cwinpy_heterodyne`` executable, and ii)
using the Python API.

Using the executable
^^^^^^^^^^^^^^^^^^^^

For most inputs we will use the default values as described in the API for
:class:`~cwinpy.heterodyne.Heterodyne`, but otherwise we can set the heterodyne parameters via a
configuration file, in this case called ``example1_config.ini``, containing:

.. literalinclude:: examples/example1_config.ini 

Running the ``cwinpy_heterodyne`` executable is done with:

.. code-block:: bash

   cwinpy_heterodyne --config example1_config.ini

The outputs (HDF5 files containing :class:`~cwinpy.data.HeterodynedData` objects) will be placed in
the ``heterodyneddata`` directory as specified by the ``output`` option in the configuration file.
The default output file name format follows the convention
``heterodyne_{pulsarname}_{detector}_{frequencyfactor}_{starttime}_{endtime}.hdf5``. Therefore, the
above command creates the two files:

* ``heterodyne_J0123+0123_H1_2_1000000000-1000086400.hdf5``
* ``heterodyne_J0404-0404_H1_2_1000000000-1000086400.hdf5``

We can now show the signals in the data, and compare them to purely simulated heterodyned data (the
original frame files were created with code that is largely independent of CWInPy), with, e.g.:

.. code-block:: python

   from cwinpy import HeterodynedData

   h1 = HeterodynedData.read("heterodyne_J0123+0123_H1_2_1000000000-1000086400.hdf5")
   fig = h1.plot(which="both")  # "both" specifies plotting real and imaginary data
   
   # create simulated heterodyned signal purely with CWInPy
   fakeh1 = HeterodynedData(
       times=h1.times,
       fakeasd=1e-48,  # add very small amount of noise
       detector="H1",
       inject=True,
       par=h1.par
   )

   ax = fig.gca()
   ax.plot(fakeh1.times, fakeh1.data.real, "k--")
   ax.plot(fakeh1.times, fakeh1.data.imag, "k--")
   ax.set_title(h1.par["PSRJ"])
   fig.tight_layout()

   fig.show()

.. thumbnail:: examples/example1_plot.png
   :width: 600px
   :align: center

The coloured lines show the data as heterodyned from the frame data for J0123+0123, while the
overplotted dashed lines show fake heterodyned signals produced by CWInPy.

Using the Python API
^^^^^^^^^^^^^^^^^^^^

There are two ways that the same thing can be achieved within Python via the API. Both use the
:func:`~cwinpy.heterodyne.heterodyne` function, which is just a wrapper to the
:class:`~cwinpy.heterodyne.Heterodyne` class, but also runs the heterodyne via
:meth:`~cwinpy.heterodyne.Heterodyne.heterodyne`.

The first option is to use a configuration file as above with:

.. code-block:: python

   from cwinpy.heterodyne import heterodyne

   het = heterodyne(config="example1_config.ini")

The second way is to explicitly pass all the options as arguments, e.g.,

.. code-block:: python

   from cwinpy.heterodyne import heterodyne

   het = heterodyne(
       starttime=1000000000,
       endtime=1000086400,
       detector="H1",
       channel="H1:FAKE_DATA",
       framecache="H-H1_FAKEDATA-1000000000-86400.gwf",
       pulsarfiles=["J0123+0123.par", "J0404-0404.par"],
       output="heterodyneddata",
       resamplerate=1.0 / 60.0,
       includessb=True,  # correct to solar system barycentre
       includebsb=True,  # correct to binary system barycentre
   )

Using Tempo2
^^^^^^^^^^^^

If you have the `Tempo2 <https://bitbucket.org/psrsoft/tempo2/src/master/>`_ pulsar timing package,
and the `libstempo <https://vallis.github.io/libstempo/>`_ Python wrapper for it, installed you can
use it to generate the phase evolution used to heterodyne the data. This is achieved by including:

.. code-block::

   usetempo2 = True

in your configuration file, or as another keyword argument to the
:func:`~cwinpy.heterodyne.heterodyne` function. The ``includeX`` keywords are not required in this
case and all delay corrections will be included.

.. note::

   For signals from sources in binary systems there will be a phase offset between signals
   heterodyned using the default `LALSuite <https://lscsoft.docs.ligo.org/lalsuite/>`_  functions
   and those heterodyned using Tempo2. This offset is not present for non-binary sources. In general
   this is not problematic, but will mean that the if heterodyning data containing a simulated
   binary signal created using, e.g., ``lalapps_Makefakedata_v5``, the recovered initial phase will
   not be consistent with the expected value.

Comparisons between heterodyning a described in the previous section and that using Tempo2 are shown
below (the heterodyne using Tempo2 is shown as the black dashed lines):

.. thumbnail:: examples/example1_plot_tempo1.png
   :width: 600px
   :align: center

.. thumbnail:: examples/example1_plot_tempo2.png
   :width: 600px
   :align: center

As noted there is a constant phase offset for the binary source J0404-0404, but the absolute value
of the time series' can be seen to be the same:

.. thumbnail:: examples/example1_plot_tempo3.png
   :width: 600px
   :align: center

Example: hardware injections in LIGO O1 data
============================================

In this example we will heterodyne the data for several `hardware injection
<https://www.gw-openscience.org/o1_inj/>`_ signals in LIGO Handford (H1) data during a day of the
first observing run `O1 <https://www.gw-openscience.org/O1/>`_. This will require access to the data
via `CVMFS <https://www.gw-openscience.org/cvmfs/>`_. The data time span will be from 1132478127 to
1132564527.

The example will look for the hardware injection signals ``5`` and ``6`` from the table `here
<https://www.gw-openscience.org/static/injections/o1/cw_injections.html>`_ (note that the table
contains the gravitational-wave signal frequency and frequency derivative, which must be halved to
give equivalent "rotational" values in the parameter files). Files containing the parameters for all
these injections for each observing run are packaged with CWInPy with locations given in the
:obj:`~cwinpy.info.HW_INJ` dictionary.

For this we can use the following configuration file:

.. literalinclude:: examples/example2_config.ini

In the above file the base CVMFS directory containing the strain data files has been specified,
which will be recursively searched for corresponding data. The ``includeflags`` and ``excludeflags``
values have been used to set the valid `time segments
<https://www.gw-openscience.org/archive/dataset/O1/>`_ of data to use, with ``H1_DATA`` specifying
to use all available valid science quality data for the H1 detector, and ``H1_NO_CW_HW_INJ``
specifying the exclusion of times when no continuous-wave hardware injections were being carried
out.

Running this analysis can then be achieved with the following code, where the first two lines just
substitute the location of the parameter files into the configuration file:

.. code-block:: bash

   basepath=`python -c "from cwinpy.info import HW_INJ_BASE_PATH; print(HW_INJ_BASE_PATH)"`
   sed -i "s|{hwinjpath}|$basepath|g" example2_config.ini
   cwinpy_heterodyne --config example2_config.ini

If the CVMFS data is being downloaded on-the-fly then (depending on your internet connection speed)
this may take on the order of ten minutes to run.

The outputs (HDF5 files containing :class:`~cwinpy.data.HeterodynedData` objects) will be placed in
the ``heterodyneddata`` directory as specified by the ``output`` option in the configuration file.
The default output file name format follows the convention
``heterodyne_{pulsarname}_{detector}_{frequencyfactor}_{starttime}_{endtime}.hdf5``. Therefore, the
above command creates the two files:

* ``heterodyne_JPULSAR05_H1_2_1132478127-1132564527.hdf5``
* ``heterodyne_JPULSAR06_H1_2_1132478127-1132564527.hdf5``

We can take a look at the heterodyned data for the hardware injection number ``5`` with:

.. code-block:: python

   from cwinpy import HeterodynedData

   hwinj = HeterodynedData.read("heterodyne_JPULSAR05_H1_2_1132478127-1132564527.hdf5")
   # "both" specifies plotting real and imaginary data, "remove_outliers" removes outliers!
   fig = hwinj.plot(which="both", remove_outliers=True)
   fig.show()

.. thumbnail:: examples/example2_plot.png
   :width: 600px
   :align: center

We can see the signal in the data by taking a spectrum:

.. code-block:: python

   figspec = hwinj.periodogram(remove_outliers=True)
   figspec[-1].show()

.. thumbnail:: examples/example2_spectrum_plot.png
   :width: 600px
   :align: center

If running on data spanning a whole observing run it makes sense to split up the analysis into many
individual jobs and run them in parallel. This can be achieved by creating a HTCondor DAG, which can
be run on a computer cluster (or on multiple cores on a single machine), as described
:ref:`below<Running using HTCondor>`.

Example: two stage heterodyne
=============================

As described in, e.g., [1]_, the heterodyne can be performed in two stages. For example, the first
stage could account for the signal's phase evolution, but neglect Doppler and relativistic
solar/binary system effects while still low-pass filtering and heavily downsampling the data. The
second stage would then apply the solar/binary system effects at the lower sample rate. In the past,
with ``lalapps_heterodyne_pulsar``, this two stage approach provided speed advantages, although with
CWInPy that advantage is negligible. However, the two stage approach can be useful if you want to
analyse data with a preliminary source ephemeris, and then re-heterodyne the same data with an
updated source ephemeris. In most cases it is recommended to heterodyne in a single stage, which
also allows slightly more aggressive filtering to be applied.

To perform the run from :ref:`the above example<Example: hardware injections in LIGO O1 data>` in
two stages one could use the following configuration (called, e.g., ``example3_stage1_config.ini``)
file for stage 1:

.. literalinclude:: examples/example3_stage1_config.ini

and then run the commands:

.. code-block:: bash

   basepath=`python -c "from cwinpy.info import HW_INJ_BASE_PATH; print(HW_INJ_BASE_PATH)"`
   sed -i "s|{hwinjpath}|$basepath|g" example3_stage1_config.ini
   cwinpy_heterodyne --config example3_stage1_config.ini

The stage 2 configuration file (called, e.g, ``example3_stage2_config.ini``) would then be:

.. literalinclude:: examples/example3_stage2_config.ini

which could be run with:

.. code-block:: bash

   sed -i "s|{hwinjpath}|$basepath|g" example3_stage2_config.ini
   cwinpy_heterodyne --config example3_stage2_config.ini

In this case the intermediate heterodyned data will be store in the ``heterodyneddata`` directory
and the final heterodyned data will be in the ``heterodyneddata_stage2`` directory. We can plot
spectra of these outputs for comparison with, e.g.:

.. code-block:: python

   from matplotlib import pyplot as plt
   from cwinpy import HeterodynedData

   # read in stage 1 and stage 2 data
   stage1 = HeterodynedData.read(
      "heterodyneddata/heterodyne_JPULSAR05_H1_2_1132478127-1132564527.hdf5"
   )
   stage2 = HeterodynedData.read(
      "heterodyneddata_stage2/heterodyne_JPULSAR05_H1_2_1132478127-1132564527.hdf5"
   )

   # create figure
   fig, ax = plt.subplots(1, 2, figsize=(8,5))

   # plot periodogram of the stage 1 data on the left
   stage1.periodogram(remove_outliers=True, ax=ax[0])
   ax[0].set_title("Stage 1 (full band)")

   # plot zoom of stage 1 on the right
   stage1.periodogram(remove_outliers=True, ax=ax[1], label="Stage 1 (zoom)")

   # plot stage 2 over the zoom of stage 1
   stage2.periodogram(remove_outliers=True, ax=ax[1], linestyle="--", color="g", label="Stage 2")

   fig.show()

.. thumbnail:: examples/example3_spectrum_plot.png
   :width: 600px
   :align: center

From the right panel it can be seen that the second stage of the heterodyne shifts the signal peak
to approximately zero Hz, as expected, and increases the power in the peak, which will be slightly
spread out over several frequency bins after only the first heterodyne stage.

Running using HTCondor
----------------------

When heterodyning long stretches of data it is preferable to split the observations up into more
manageable chunks of time. The can be achieved by splitting up the analysis and running it as
multiple independent jobs on a machine/cluster, or over the `Open Science Grid
<https://opensciencegrid.org/>`_, using the `HTCondor <https://htcondor.readthedocs.io/en/latest/>`_
job scheduler system. This can be done using the ``cwinpy_heterodyne_dag`` executable (or the
:func:`~cwinpy.heterodyne.heterodyne_dag` API).

This can be run using a configuration script containing the information as described in the example
below:

.. literalinclude:: cwinpy_heterodyne_dag.ini

where this contains information for heterodyning data from the O1 *and* O2 observing runs for the
two LIGO detectors, H1 and L1. Comments about each input parameter, and different potential input
options are given inline; some input parameters are also commented out using a ``;`` in which case
the default values would be used. For more information on the various HTCondor options see the `user
manual <https://htcondor.readthedocs.io/en/v8_8_4/users-manual/index.html>`_.

This configuration file could then be run to generate the HTCondor DAG using:

.. code-block:: bash

   cwinpy_heterodyne_dag cwinpy_heterodyne_dag.ini

and the generated DAG then submitted (if the ``submitdag`` option is set to ``False`` in the
configuration file) using:

.. code-block:: bash

   condor_submit_dag /home/username/heterodyne/submit/dag_cwinpy_heterodyne.submit

.. note::

   When running ``condor_submit_dag`` you need to make sure you call it from the same directory that
   you ran ``cwinpy_heterodyne_dag`` from and make sure the path to the DAG file is relative to the
   current directory.

This example will generate the following directory tree structure:

.. code-block:: bash

   /home/username/heterodyne
                   ├── configs  # directory containing configuration files for individual cwinpy_heterodyne runs
                   ├── submit   # directory containing the Condor submit and DAG files
                   ├── log      # directory containing the Condor log files
                   ├── H1       # directory containing the heterodyned data files for the H1 detector
                   └── L1       # directory containing the heterodyned data files for the L1 detector

By default the multiple heterodyned data files for each pulsar created due to the splitting will be
merged using the ``cwinpy_heterodyne_merge`` executable (see the
:func:`~cwinpy.heterodyne.heterodyne_merge` API). If the ``remove`` option is set in the
configuration file then the individual unmerged files will be removed, but by default they will be
kept (although not for the :ref:`Quick setup`).

The default naming format of the output heterodyned data files in their respective detector
directories will be:
``heterodyne_{pulsarname}_{detector}_{frequencyfactor}_{gpsstart}-{gpsend}.hdf5`` although this can
be altered using the ``label`` option.

.. note::

   If running on LIGO Scientific Collaboration computing clusters the ``acounting_group`` value must
   be specified and provide a valid tag. Valid tag names can be found `here
   <https://accounting.ligo.org/user>`_ unless custom values for a specific cluster are allowed.

   As stated earlier, if accessing proprietary LIGO/Virgo data on a cluster you will need to make
   sure to run:

   .. code-block:: bash

      ligo-proxy-init -p albert.einstein

   where ``albert.einstein`` is substituted for your username, before running the executable.

Open Science Grid
^^^^^^^^^^^^^^^^^

If you have access to resources on the Open Science Grid (OSG) then the analysis can also be run on
them. This can be achieved by setting the ``osg`` value in the configuration file to be ``True``.
Before launching the script you should make sure that you are using CWInPy from within an `IGWN
conda environment <https://computing.docs.ligo.org/conda/>`_ as distributed over CVMFS.

Two stage approach
^^^^^^^^^^^^^^^^^^

As described in :ref:`Example: two stage heterodyne`, the heterodyne can be run in two stages with,
for example, the first stage taking account of the phase evolution while ignoring
Doppler/relativistic effects, and the second stage subsequently including these effects. While it is
recommended to perform the heterodyne in a single stage, it may sometimes be useful to have
intermediate products.

To create a DAG for this "two stage" approach the following option needs to be set in the
``[heterodyne]`` section of the configuration file:

.. code-block::

   [heterodyne]
   stages = 2

The `resamplerate` option can then be given as a list containing two values: the resample rates for
each stage. If not given, the default is to resample to 1 Hz for the first stage and 1/60 Hz for the
second stage. The values in the dictionary given for the ``outputdir`` option should also be lists
of two directories where the outputs of each stage will be located. The options ``includessb``,
``includebsb``, ``includeglitch`` and ``includefitwaves`` should also be two-valued lists of
booleans stating which phase model components to include in each stage. By default, the first stage
will have these all as ``False`` and the second stage will have them all as ``True``.

.. note::

   By default, if running the the two stage approach, the knee frequency of the low-pass filter will
   be 0.5 Hz compared to 0.1 Hz if running a single stage. This differs from the default used by
   ``lalapps_heterodyne_pulsar``, which uses 0.25 Hz.

Re-heterodyning data
^^^^^^^^^^^^^^^^^^^^

If you have previously heterodyned data files that you want to re-heterodyne (these might be the
outputs of the first stage of a two stage analysis) then you can use the configuration file with the
``stages`` option set to ``1``, but instead supply the ``heterodyneddata`` option in the
``[heterodyne]`` section rather than frame information. This can be a path to a directory containing
previously heterodyned data (:class:`~cwinpy.data.HeterodynedData` objects saved in HDF5 format), an
inidividual file path (if analysing a single pulsar), or a dictionary keyed to the pulsar name and
pointing to the heterodyned data path for that source.

Quick setup
===========

The ``cwinpy_heterodyne_dag`` script has some quick setup options that allow an analysis to be
launched in one line without the need to define a configuration file. These options **require** that
the machine/cluster that you are running HTCondor on has access to open data from GWOSC available
via CVMFS. It is also recommended that you run CWInPy from within an `IGWN conda environment
<https://computing.docs.ligo.org/conda/>`_ 

For example, if you have a Tempo(2)-style pulsar parameter file, e.g., ``J0740+6620.par``, and you
want to analyse the open `O1 data <https://www.gw-openscience.org/O1/>`_ for the two LIGO detectors
you can simply run:

.. code-block:: bash

   cwinpy_heterodyne_dag --run O1 --pulsar J0740+6620.par --output /home/usr/heterodyneddata

where ``/home/usr/heterodyneddata`` is the name of the directory where the run information and
results will be stored (if you don't specify an ``--output`` then the current working directory will
be used). This command will automatically submit the HTCondor DAG for the job. To specify multiple
pulsars you can use the ``--pulsar`` option multiple times. If you do not have a parameter file for
a pulsar you can instead use the ephemeris given by the `ATNF pulsar catalogue
<https://www.atnf.csiro.au/research/pulsar/psrcat/>`_. To do this you need to instead supply a
pulsar name (as recognised by the catalogue), for example, to run the analysis using O2 data for the
pulsar `J0737-3039A <https://en.wikipedia.org/wiki/PSR_J0737%E2%88%923039>`_ you could do:

.. code-block:: bash

   cwinpy_heterodyne_dag --run O2 --pulsar J0737-3039A --output /home/usr/heterodyneddata

Internally the ephemeris information is obtained using the :class:`~psrqpy.search.QueryATNF` class
from `psrqpy <https://psrqpy.readthedocs.io/en/latest/>`_.

CWInPy also contains information on the continuous :ref:`hardware injections<Hardware Injections>`
performed in each run, so if you wanted the analyse the these in, say, the LIGO `sixth science run
<https://www.gw-openscience.org/archive/S6/>`_, you could do:

.. code-block:: bash

   cwinpy_heterodyne_dag --run S6 --hwinj --output /home/usr/hwinjections

Other command line arguments for ``cwinpy_heterodyne_dag``, e.g., for setting specific detectors,
can be found :ref:`below<heterodyne Command line arguments>`. If running on a LIGO Scientific
Collaboration cluster the ``--accounting-group-tag`` flag must be set to a valid `accounting tag
<https://accounting.ligo.org/user>`_, e.g.,:

.. code-block:: bash

   cwinpy_heterodyne_dag --run O1 --hwinj --output /home/user/O1injections --accounting-group-tag ligo.prod.o1.cw.targeted.bayesian

.. note::

   The quick setup will only be able to use default parameter values for the heterodyne. For
   "production" analyses, or if you want more control over the parameters, it is recommended that
   you use a configuration file to set up the run.

   The frame data used by the quick setup defaults to that with a 4096 Hz sample rate. However, if
   analysing sources with frequencies above about 1.6 kHz this should be switched, using the
   ``--samplerate`` flag to using the 16 kHz sampled data. By default, if analysing hardware
   injections for any of the advanced detector runs the 16 kHz data will be used due to frequency of
   the fastest pulsar being above 1.6 kHz.

.. _heterodyne Command line arguments:

Command line arguments
----------------------

The command line arguments for ``cwinpy_heterodyne`` (as extracted using ``cwinpy_heterodyne --help``) are
given below:

.. literalinclude:: heterodyne_help.txt
   :language: none

The command line arguments for ``cwinpy_heterodyne_dag`` (as extracted using
``cwinpy_heterodyne_dag --help``) are:

.. literalinclude:: heterodyne_dag_help.txt
   :language: none

.. _heterodyne API:

Heterodyne API
--------------

.. automodule:: cwinpy.heterodyne
   :members: Heterodyne, heterodyne, heterodyne_dag, heterodyne_merge

References
==========

.. [1] `R. Dupius & G. Woan
   <https://ui.adsabs.harvard.edu/#abs/2005PhRvD..72j2002D/abstract>`_,
   *Phys. Rev. D*, **72**, 102002 (2005)

.. [2] `M. Yu et al,
   <https://ui.adsabs.harvard.edu/abs/2013MNRAS.429..688Y/abstract>`_,
   *Mon. Not. R. Astron. Soc.*, **429**, 688 (2013)

.. [3] `G. B. Hobbs, R. T. Edwards & R. N. Manchester
   <https://ui.adsabs.harvard.edu/abs/2006MNRAS.369..655H/abstract>`_,
   *Mon. Not. R. Astron. Soc.*, **369**, 655 (2006)