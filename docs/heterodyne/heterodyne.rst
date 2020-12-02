#################
Heterodyning data
#################

Gravitational-wave strain data is often sampled at 16384 Hz. It is only feasible to perform
:ref:`parameter estimation <pe>` for a continuous gravitational-wave signal from a particular pulsar
using data that is heavily downsampled. As described in :ref:`data:heterodyned-data`, this is done
using the method from [1]_. The pulsar's phase evolution is assumed to be well described by a Taylor
expansion

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
:ref:`below<API>`.

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