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
a time- and source-and-position-dependent delay term that accounts for the Doppler and relativistic
delays between the time at the detector and the pulsar proper time (see, e.g., Equation 1 of [3]_).
The resulting complex series :math:`h'(t)` is low-pass filtered (effectively a band-pass filter on
the two-sided complex data) using a high order Butterworth filter, with a given knee frequency, and
down-sampled, via averaging, to a far lower sample rate than the original raw data. In general
gravitational-wave detector data is sampled at 16384 Hz, and often the heterodyned data is
downsampled to 1/60 Hz (one sample per minute).

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
gravitational wave data for individual pulsars or multiple pulsars. We will cover some examples of
running analyses via use of command line arguments or a configuration file supplied to
``cwinpy_heterodyne``, or through the :ref:`API<heterodyne API>`. The current command line arguments
for ``cwinpy_heterodyne`` are given :ref:`below<heterodyne Command line arguments>`.

If running an analysis for multiple pulsars on a large stretch of data it is recommended
that you split the analysis up to run as many separate jobs. If you have access to a computer
cluster (such as those available to the LVK, or via the OSG), or individual machine (see below),
running the `HTCondor <https://research.cs.wisc.edu/htcondor/>`_ job scheduler system, the analysis
can be split up using the ``cwinpy_heterodyne_dag`` pipeline script. We will also describe examples
of using this.

In many of the example below we will assume that you are able to access the open LIGO and Virgo data
available from the `GWOSC <https://www.gw-openscience.org/>`_ via `CVMFS
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
the data>`). The fake pulsars have parameters defined in TEMPO(2)-style parameter files (where
frequencies, frequency derivatives and phases are the rotational values rather than the
gravitational-wave values), as follows:

* an isolated pulsar in a file called ``J0123+0123.par``

.. code-block:: bash

   PSRJ    J0123+0123
   RAJ     01:23:00.0
   DECJ    01:23:00.0
   F0      3.4728
   F1      -4.93827e-11
   F2      1.17067e-18
   PEPOCH  55818.074666481479653
   EPHEM   DE405
   UNITS   TDB
   H0      3.0e-24
   PHI0    0.5
   COSIOTA 0.1
   PSI     0.5

* a pulsar in a binary system in a file called ``J0404-0404.par``

.. code-block:: bash

   PSRJ    J0404-0404
   RAJ     04:04:00.0
   DECJ    -04:04:00.0
   F0      1.93271605
   F1      4.93827e-13
   F2      -6.7067e-21
   PEPOCH  55818.074666481479653
   A1      1.4
   ECC     0.09
   OM      0.0
   T0      55817.9830345370355644263
   PB      0.1
   BINARY  BT
   EPHEM   DE405
   UNITS   TDB
   H0      7.5e-25
   PHI0    0.35
   COSIOTA 0.6
   PSI     1.1

One way to create the simulated data is as follows:

.. code-block:: python

   import shutil
   import subprocess as sp
   from lalpulsar.PulsarParametersWrapper import PulsarParametersPy
   from astropy.utils.data import download_file

   # set data start, duration and bandwidth
   fakedatastart = 1000000000
   fakedataduration = 86400  # 1 day in seconds
   fakedatabandwidth = 8  # 8 Hz

   parfiles = ["J0123+0123.par", "J0404-0404.par"]

   # create injection files for lalapps_Makefakedata_v5
   # requirements for Makefakedata pulsar input files
   isolatedstr = """\
   Alpha = {alpha}
   Delta = {delta}
   Freq = {f0}
   f1dot = {f1}
   f2dot = {f2}
   refTime = {pepoch}
   h0 = {h0}
   cosi = {cosi}
   psi = {psi}
   phi0 = {phi0}
   """

   binarystr = """\
   orbitasini = {asini}
   orbitPeriod = {period}
   orbitTp = {Tp}
   orbitArgp = {argp}
   orbitEcc = {ecc}
   """

   injfile = "inj.dat"
   fp = open(injfile, "w")

   for i, enumerate(parfile) in parfiles:
       p = PulsarParametersPy(parfile)
       fp.write("[Pulsar {}]\n".format(i+1))

       # set parameters (multiply freqs/phase by 2)
       mfddic = {
           "alpha": p["RAJ"],
           "delta": p["DECJ"],
           "f0": 2 * p["F0"],
           "f1": 2 * p["F1"],
           "f2": 2 * p["F2"],
           "pepoch": p["PEPOCH"],
           "h0": p["H0"],
           "cosi": p["COSIOTA"],
           "psi": p["PSI"],
           "phi0": 2 * p["PHI0"],
       }
       fp.write(isolatedstr.format(**mfddic))

       if p["BINARY"] is not None:
           mfdbindic = {
               "asini": p["A1"],
               "Tp": p["T0"],
               "period": p["PB"],
               "argp": p["OM"],
               "ecc": p["ECC"],
           }
           fp.write(binarystr.format(**mfdbindic))

       fp.write("\n")
   fp.close()

   # set ephemeris files
   efile = download_file(
       DOWNLOAD_URL.format("earth00-40-DE405.dat.gz"), cache=True
   )
   sfile = download_file(DOWNLOAD_URL.format("sun00-40-DE405.dat.gz"), cache=True)

   # set detector
   detector = "H1"
   channel = "{}:FAKE_DATA".format(detector)

   # set noise amplitude spectral density (use a small value to see the signal clearly)
   sqrtSn = 1e-29

   # set Makefakedata commands
   cmds = [
       "-F",
       ".",
       "--outFrChannels={}".format(channel),
       "-I",
       detector,
       "--sqrtSX={0:.1e}".format(sqrtSn),
       "-G",
       str(fakedatastart),
       "--duration={}".format(fakedataduration),
       "--Band={}".format(fakedatabandwidth),
       "--fmin",
       "0",
       '--injectionSources="{}"'.format(injfile),
       "--outLabel={FAKEDATA}",
       '--ephemEarth="{}"'.format(efile),
       '--ephemSun="{}"'.format(sfile),
   ]

   # run makefakedata
   sp.run([shutil.which("lalapps_Makefakedata_v5")] + cmds)
       
This should create the following files XXX in the gwf format.

Heterodyning the data
#####################

.. _heterodyne Command line arguments:

Command line arguments
----------------------

The command line arguments for ``cwinpy_heterodyne`` (as extracted using ``cwinpy_heterodyne --help``) are
given below:

.. literalinclude:: heterodyne_help.txt
   :language: none

.. _heterodyne API:

API
---

.. automodule

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