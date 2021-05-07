#################
Heterodyning data
#################

Gravitational-wave strain data is often sampled at 16384 Hz. It is only feasible to perform
:ref:`parameter estimation<Known pulsar parameter estimation>` for a continuous gravitational-wave
signal from a particular pulsar using data that is heavily downsampled. As described in
:ref:`Heterodyned data`, this is done using the method from [1]_. The pulsar's phase evolution is
assumed to be well described by a Taylor expansion

.. math::

   \phi(t') = 2 \pi C\left(f_0(t'-T_0) + \frac{1}{2}\dot{f}_0(t'-T_0)^2 + \dots\right)

where :math:`f_0` is the source rotation frequency, :math:`\dot{f}_0` is the first frequency
derivative, :math:`t'` is the time in a frame at rest with respect to the pulsar (the pulsar proper
time, which for isolated pulsars is assumed to be consistent with the solar system barycentre),
:math:`T_0` is the time reference epoch in the pulsar proper time, and :math:`C` is an emission
mechanism-dependent scaling factor (often set to 2, for emission from the :math:`l=m=2` quadrupole
mode.) 

.. note::
   The analysis code described below can use a Taylor expansion in the phase up to arbitrarily high
   orders. The model can also include terms in addition to this expansion, such as those describing
   the evolution of the phase following a glitch (using the model described in Eqn. 1 of [2]_) and
   sinusoidal terms to whiten timing noise (using the model described in Eqn. 10 of [3]_).
  
The raw real time series of :math:`h(t)` from a detector is "heterodyned",

.. math::

   h'(t) = h(t) e^{-2\pi i \phi(t + \Delta t(t))},

to remove the high-frequency variation of the potential source signal, where :math:`\Delta t(t)` is
a time- and source- and detection-position-dependent delay term that accounts for the Doppler and
relativistic delays between the time at the detector and the pulsar proper time (see, e.g., Equation
1 of [3]_). The resulting complex series :math:`h'(t)` is low-pass filtered (effectively a band-pass
filter on the two-sided complex data) using a high order Butterworth filter, with a given knee
frequency, and down-sampled, via averaging, to a far lower sample rate than the original raw data.
In general, gravitational-wave detector data is sampled at 16384 Hz, and often the heterodyned data
is downsampled to 1/60 Hz (one sample per minute).

CWInPy can be used to perform this heterodyning, taking files containing raw gravitational-wave
strain and returning the complex, filtered, down-sampled time series in a form that can be read in
as a :class:`~cwinpy.data.HeterodynedData` object.

CWInPy comes with an executable, ``cwinpy_heterodyne``, for implementing this, which closely (but
not identically) emulates the functionality from the `LALSuite
<https://lscsoft.docs.ligo.org/lalsuite/>`_ code ``lalapps_heterodyne_pulsar``.

There is also an API for running this analysis from within a Python shell or script as described
:ref:`below<heterodyne API>`.

Running the analysis
--------------------

The ``cwinpy_heterodyne`` script, and :ref:`API<heterodyne API>`, can be used to process
gravitational-wave data for individual pulsars or multiple pulsars. We will cover some examples of
running analyses via use of command line arguments or a configuration file supplied to
``cwinpy_heterodyne``, or through the :ref:`API<heterodyne API>`. The current command line arguments
for ``cwinpy_heterodyne`` are given :ref:`below<heterodyne Command line arguments>`.

If running an analysis for multiple pulsars on a large stretch of data it is recommended that you
split the analysis up to run as many separate jobs. If you have access to a computer cluster (such
as those available to the LVK, or via the OSG), or an individual machine (see below), running the
`HTCondor <https://research.cs.wisc.edu/htcondor/>`_ job scheduler system, the analysis can be split
up using the ``cwinpy_heterodyne_dag`` pipeline script. We will also describe examples of using
this.

In many of the examples below we will assume that you are able to access the open LIGO and Virgo
data available from the `GWOSC <https://www.gw-openscience.org/>`_ via `CVMFS
<https://cvmfs.readthedocs.io/>`_. To find out more about accessing this data see the instructions
`here <https://www.gw-openscience.org/cvmfs/>`_.

.. note::

   To run a heterodyne analysis as multiple jobs on your own machine you can install HTCondor for a
   single machine. Installation instructions for various Linux distributions are found `here
   <https://research.cs.wisc.edu/htcondor/instructions/>`_ and it is recommended to install
   ``mini[ht]condor``, which configures HTCondor to run on a single node. If you have multiple cores
   and a large amount of memory this can be a reasonable option.

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
``lalapps_Makefakedata_v5`` (skip straight to the heterodyning description :ref:`here<Heterodyning
the data>`). The two fake pulsars have parameters defined in TEMPO(2)-style parameter files (where
frequencies, frequency derivatives and phases are the rotational values rather than the
gravitational-wave values), as follows:

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

We will show how to heterodyning the data in ``H-H1_FAKEDATA-1000000000-86400.gwf`` for the two
different pulsars by using i) a configuration file for the ``cwinpy_heterodyne`` script, ii) the
Python API.

Using the script
^^^^^^^^^^^^^^^^

For most inputs we will use the default values as described in the API for
:class:`~cwinpy.heterodyne.Heterodyne`, but otherwise we can set the heterodyne parameters via a
configuration file, in this case called ``example1_config.ini``, containing:

.. literalinclude:: examples/example1_config.ini 

Running the ``cwinpy_heterodyne`` script is done with:

.. code-block:: bash

   cwinpy_heterodyne --config example1_config.ini

In this case the outputs (HDF5 files containing :class:`~cwinpy.data.HeterodynedData` objects) will
be placed in the ``heterodyneddata`` directory as specified by the ``output`` option in the
configration file. The default output file name format follows the convention
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

There are two ways that the same thing can be acheived within Python via the API. Both use the
:func:`~cwinpy.heterodyne.heterodyne` function, which is just a wrapper to the
:class:`~cwinpy.heterodyne.Heterodyne`, but also runs the heterodyne via
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

Example: hardware injections in LIGO O1 data
============================================

In this example we will heterodyne the data for several `hardware injection
<https://www.gw-openscience.org/o1_inj/>`_ signals in LIGO Handford (H1) data during a day of the
first observing run `O1 <https://www.gw-openscience.org/O1/>`_. This will require access to the data
via `CVMFS <https://www.gw-openscience.org/cvmfs/>`_. The data time span will be from 1132478127 to
1132564527.

The example assumes that you have a directory called ``pulsars`` containing TEMPO-style parameter
files for the injections labelled ``0``, ``3``, ``5``, ``6`` and ``8`` in the table `here
<https://www.gw-openscience.org/static/injections/o1/cw_injections.html>`_ (note that the table
contains the signal frequency and frequency derivative, which must be halved to give equivalent
"rotational" values in the parameter files). These files can be found in :download:`this tarball
<examples/hardware_injections.tar.gz>`.

For this we can use the following configuration file:

.. literalinclude:: examples/example2_config.ini

In the above file the base CVMFS directory containing the strain data files has been specified,
which will be recursively searched for corresponding data. The ``includeflags`` and ``excludeflags``
values have been used to set the valid `time segments
<https://www.gw-openscience.org/archive/dataset/O1/>`_ of data to use, with ``H1_CBC_CAT2``
specifying to use all available valid science quality data for the H1 detector (using ``H1_DATA``
can still have gaps), and ``H1_NO_CW_HW_INJ`` specifying the exclusion of times when no
continuous-wave hardware injections were being carried out.

Running this analysis can then be achieved with:

.. code-block:: bash

   cwinpy_heterodyne --config example1_config.ini


.. _heterodyne Command line arguments:

Command line arguments
----------------------

The command line arguments for ``cwinpy_heterodyne`` (as extracted using ``cwinpy_heterodyne --help``) are
given below:

.. literalinclude:: heterodyne_help.txt
   :language: none

.. _heterodyne API:

Heterodyne API
--------------

.. automodule:: cwinpy.heterodyne
   :members: Heterodyne, heterodyne, heterodyne_dag

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