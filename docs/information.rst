###############
Run information
###############

In CWInPy you can extract information on the observing run start and end times and the parameters of
`continuous-wave hardware injections <https://www.gw-openscience.org/O3/o3a_inj/>`_.

Run times
---------

The start and end GPS times of the LIGO and Virgo observation runs available as open data via `GWOSC
<https://www.gw-openscience.org/data/>`_ can be found by in the :obj:`~cwinpy.info.RUNTIMES`
:class:`~cwinpy.info.Runtimes` dictionary. This dictionary has methods to convert the start and end
times to ISO format (or Modified Julian Data format), extract times for a single observing run, or a
single detector. The available run names currently are: ``S5``, ``S6``, ``O1``, ``O2`` and ``O3a``.

.. code-block:: python

   from cwinpy import RUNTIMES # print run times for H1 in ISO format
   print(RUNTIMES.detector("H1").iso) {'S5': {'H1': ['2005-11-04 16:00:00.000', '2007-10-01
   00:00:00.000']}, 'S6': {'H1': ['2009-07-07 21:00:00.000', '2010-10-20 15:00:00.000']}, 'O1':
   {'H1': ['2015-09-12 00:00:00.000', '2016-01-19 16:00:00.000']}, 'O2': {'H1': ['2016-11-30
   16:00:00.000', '2017-08-25 22:00:00.000']}, 'O3a': {'H1': ['2019-04-01 15:00:00.000', '2019-10-01
   15:00:00.000']}}

Hardware injections
-------------------

In each of the observing runs simulated signals from a variety of sources have been directly
"injected" into the detector (see [1]_). This includes continuous signals at a variety of
frequencies and amplitudes. TEMPO-style pulsar parameter files containing the parameters of these
continuous signals are packaged with CWInPy and can be found in the :obj:`~cwinpy.info.HW_INJ`
dictionary. This dictionary is keyed on the observing run names and for each run contains both the
file paths and the contents of the files in :class:`~cwinpy.parfile.PulsarParameters` objects that
were injected.

.. note::

   The hardware injections were only performed in the LIGO detectors and not the Virgo detector.

The files have names of the form ``PULSARXX.par`` with example contents for ``PULSAR00.par`` from
the S5 run being::

   PSRJ    JPULSAR00
   F0      132.7885526
   F1      -2.075e-12
   RA      04:46:12.4627784428
   DEC     -56:13:02.9490031074
   PEPOCH  52944.0007428703684126958
   UNITS   TDB
   EPHEM   DE200
   H0      2.46648883177e-25
   IOTA    0.651944874871
   PSI     0.770087086
   PHI0    1.33

In these files the ``F0`` and ``F1`` values are equivalent to a pulsar's rotation frequency (Hz) and
frequency derivative (Hz/s), and therefore are half the signal frequency/frequency derivative values
(as given in the `GWOSC tables
<https://www.gw-openscience.org/static/injections/cw/S5_injections/S5_injection_params_s5try3.html>`_)
in the data. The initial phase ``PHI0`` value is also the equivalent rotational phase (rads) and is
therefore half the signal phase. The right ascension ``RA`` is given in "HH:MM:SS" format and the
declination ``DEC`` is given in "DD:MM:SS" format. The ``PEPOCH`` value is the epoch at which the
initial phase and frequency are defined and is given as a Modified Julian Day in the solar system
barycentre frame. The ``UNITS`` value defines whether the `Barcentric Dynamical Time
<https://en.wikipedia.org/wiki/Barycentric_Dynamical_Time>`_ (TDB) or `Barycentric Coordinate Time
<https://en.wikipedia.org/wiki/Barycentric_Coordinate_Time>`_ (TCB) was used to generate the
signals. The ``EPHEM`` value defines the JPL solar system ephemeris used to create the signals; this
is either ``DE200`` or ``DE405`` for the injections.

In some cases the gravitational-wave polarisation angle ``PSI`` has been rotated to be positive
valued compared to the values from the GWOSC tables: :math:`\psi \rightarrow \psi + \pi/2`, which
also requires an equivalent rotation of the initial phase: :math:`\phi_0 \rightarrow (\phi_0 +
\pi/2) \text{ mod } \pi`.

The names of the GWOSC analysis segments used for each run are given in
:obj:`~cwinpy.info.HW_INJ_SEGMENTS`, which include the time segments to include and those to exclude
if required.

S5
~~

In S5 there were 10 continuous injections labeled ``PULSAR00``-``PULSAR09``. There were several
epochs of injections with different amplitudes, but the values stored here represent those from
`"s5try3"
<https://www.gw-openscience.org/static/injections/cw/S5_injections/S5_injection_params_s5try3.html>`_
that were present between GPS times of 829412600 and 875301345 as given in
:obj:`~cwinpy.info.HW_INJ_RUNTIMES`. These injections used the DE200 solar system ephemeris.

S6
~~

In S6 the same `10 injectons <https://www.gw-openscience.org/s6hwcw/>`_ as :ref:`S5` were used.

O1
~~

In O1 there were `15 continuous injections
<https://www.gw-openscience.org/static/injections/o1/cw_injections.html>`_ labeled
``PULSAR00``-``PULSAR14``. The first 10 are the same as those in :ref:`S5` and :ref:`S6`, but with
adjusted amplitudes. All injections now use the DE405 solar system ephemeris.

O2
~~

In O2 there were the same `15 continuous injections
<https://www.gw-openscience.org/O2_injection_params/>`_ as in :ref:`O1`, although a subset have
different amplitudes.

O3
~~

In O3 there were `18 continuous injections
<https://www.gw-openscience.org/O3/O3April1_injection_parameters/>`_ labeled
``PULSAR00``-``PULSAR17``. The first 15 are the same as those in :ref:`O1` and :ref:`O2`, but with
adjusted amplitudes. The injection labeled ``PULSAR15`` is a very high frequency source, with a
signal frequency of 2991 Hz. The injections labeled ``PULSAR16`` and ``PULSAR17`` are simulated to
be in binary systems.

CVMFS data locations
--------------------

The directory locations of GWOSC open data available via CVMFS can be found in the
:obj:`~cwinpy.info.CVMFS_FRAME_DATA_LOCATIONS` dictionary (for ``.gwf`` frame files) and the
:obj:`~cwinpy.info.CVMFS_HDF5_DATA_LOCATIONS` dictionary (for HDF5 files). These are given for each
observing run and each detector, with strain data sampled at 4096 Hz in the ``"4k"`` key (available
for all observing runs) and data sampled at 16384 Hz in the ``"16k"`` key (available for all
*advanced detector* runs, starting with the ``O`` prefix).

Run information references
--------------------------

.. [1] `C. Biwer et al <https://ui.adsabs.harvard.edu/abs/2017PhRvD..95f2002B/abstract>`_, *PRD*,
   **95**, 062002 (2017)

.. automodule:: cwinpy.info
   :members: