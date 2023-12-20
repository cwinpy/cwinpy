###################
Simulating a signal
###################

Here we describe the :class:`~cwinpy.signal.HeterodynedCWSimulator` class for simulating a
signal from a continuous wave source after application of a heterodyne as described in Equations 7
and 8 of [1]_.

When generating the signal using the :meth:`~cwinpy.signal.HeterodynedCWSimulator.model` method, the
default is to assume that the phase evolution used in the heterodyne exactly matches the signal's
phase evolution and that emission is from the :math:`l=m=2` quadrupole mode (a triaxial source
emitting at twice the rotation frequency), i.e., the :math:`\Delta\phi` term in Equations 8 of [1]_
is zero. If the file/object that provides the signal parameters contains parameters for
non-tensorial polarisation modes (i.e., if assuming emission is not purely as given by general
relativity; see Table 6 of [1]_ for the names of the required parameters) then those polarisations
will be included in the signal (e.g., Equation 11 of [1]_), otherwise general relativity will be
assumed and only the "+" and "×" polarisations included.

The :meth:`~cwinpy.signal.HeterodynedCWSimulator.model` method can also generate model amplitude
coefficients for use in a "pre-summed" likelihood function, as described in Appendix C and Equations
C8 and C12 of [1]_, by passing the ``outputampcoeffs`` argument as ``True``. For a GR signal this
means two complex amplitude values get returned, while for non-GR signals six complex amplitudes are
returned.

The signal model at the rotation frequency, i.e., from the :math:`l=2,m=1` mode, can be set by
passing ``freqfactor=1.0`` as an argument to :meth:`~cwinpy.signal.HeterodynedCWSimulator.model`.
This requires that the file/object containing the source parameters has the :math:`C_{21}` and
:math:`\Phi_{21}` parameters defined as given in Equation 7 of [1]_.

.. note::
   
   If generating a simulated signal in IPython or a Jupyter notebook,
   creating an instance of this class can be very slow (taking minutes
   compared to a fraction of a second), due to redirections of
   ``stdout``/``stderr`` in the SWIG-wrapped ``LIGOTimeGPS`` class. To
   avoid this the call to :class:`~cwinpy.signal.HeterodynedCWSimulator`
   should be either done within a context manager, e.g.,:

   .. code-block:: python

      import lal

      with lal.no_swig_redirect_standard_output_error():
            sim = HeterodyneCWSimulator(...)

   or by globally disabling redirection with:

   .. code-block:: python

      import lal

      lal.swig_redirect_standard_output_error(True)

      sim = HeterodyneCWSimulator(...)

Phase offset signal
===================

The :meth:`~cwinpy.signal.HeterodynedCWSimulator.model` method can also be used to generate a signal
that is offset, in terms of its phase evolution, from the heterodyne phase. This can be used to
simulate cases where the heterodyned was not performed with an identical phase evolution to the
signal (due to differences between the source rotational/binary parameters). The pulsar parameter
file/object passed when constructing a :class:`~cwinpy.signal.HeterodynedCWSimulator` object is
assumed to contain the parameters for the heterodyne phase model, while
:meth:`~cwinpy.signal.HeterodynedCWSimulator.model` can be passed an "updated" file/object
containing potentially different (phase) parameters that are the "true" source parameters using the
``newpar`` argument (and ``updateX`` arguments for defining which components of the phase evolution
need updating compared to the heterodyne values). The difference in phase, as calculated using
Equation 10 of [1]_, will then be used when generating the signal.

Using Tempo2
------------

By default, the phase difference evolution will be calculated using functions within LALSuite, which
has its own routines for calculating the solar system and binary system delay terms. However, if the
`Tempo2 <https://bitbucket.org/psrsoft/tempo2/src/master/>`_ pulsar timing software [2]_, and the
`libstempo <https://vallis.github.io/libstempo/>`_ Python wrapper for this, are installed, then
Tempo2 can instead be used for the generation of the phase evolution. This is done by setting the
``usetempo2`` argument to :meth:`~cwinpy.signal.HeterodynedCWSimulator.model` to ``True``.

.. note::

   If using Tempo2 you cannot use the ``updateX`` keyword arguments to specify whether or not to
   recalculate a component of the phase model. All components will be regenerated if present.

Examples
========

An example usage to generate the complex heterodyned signal time series is:

.. code-block:: python

   from cwinpy.signal import HeterodynedCWSimulator
   from cwinpy import PulsarParameters
   from astropy.time import Time
   from astropy.coordinates import SkyCoord
   import numpy as np

   # set the pulsar parameters
   par = PulsarParameters()
   pos = SkyCoord("01:23:34.5 -45:01:23.4", unit=("hourangle", "deg"))
   par["PSRJ"] = "J0123-4501"
   par["RAJ"] = pos.ra.rad
   par["DECJ"] = pos.dec.rad
   par["F"] = [123.456789, -9.87654321e-12]  # frequency and first derivative
   par["PEPOCH"] = Time(58000, format="mjd", scale="tt").gps  # frequency epoch
   par["H0"] = 5.6e-26     # GW amplitude
   par["COSIOTA"] = -0.2   # cosine of inclination angle
   par["PSI"] = 0.4        # polarization angle (rads)
   par["PHI0"] = 2.3       # initial phase (rads)

   # set the GPS times of the data
   times = np.arange(1000000000.0, 1000086400.0, 3600, dtype=np.float128)

   # set the detector
   det = "H1"  # the LIGO Hanford Observatory

   # create the HeterodynedCWSimulator object
   het = HeterodynedCWSimulator(par, det, times=times)

   # get the model complex strain time series
   model = het.model()

.. note::

   If you have a pulsar parameter file you do not need to create a
   :class:`~cwinpy.parfile.PulsarParameters` object, you can just pass the path of that file, e.g.,

   .. code-block:: python

      par = "J0537-6910.par"
      het = HeterodynedCWSimulator(par, det, times=times)

An example (showing both use of the default phase evolution calculation and that using Tempo2) of
getting the time series for a signal that has phase parameters that are not identical to the
heterodyned parameters would be:

.. code-block:: python

   from cwinpy.signal import HeterodynedCWSimulator
   from cwinpy import PulsarParameters
   from astropy.time import Time
   from astropy.coordinates import SkyCoord
   from copy import deepcopy
   import numpy as np
   
   # set the "heterodyne" pulsar parameters
   par = PulsarParameters()
   pos = SkyCoord("01:23:34.5 -45:01:23.4", unit=("hourangle", "deg"))
   par["PSRJ"] = "J0123-4501"
   par["RAJ"] = pos.ra.rad
   par["DECJ"] = pos.dec.rad
   par["F"] = [123.4567, -9.876e-12]  # frequency and first derivative
   par["PEPOCH"] = Time(58000, format="mjd", scale="tt").gps  # frequency epoch
   par["H0"] = 5.6e-26     # GW amplitude
   par["COSIOTA"] = -0.2   # cosine of inclination angle
   par["PSI"] = 0.4        # polarization angle (rads)
   par["PHI0"] = 2.3       # initial phase (rads)

   # set the times
   times = np.arange(1000000000.0, 1000086400.0, 600, dtype=np.float128)

   # set the detector
   det = "H1"  # the LIGO Hanford Observatory

   # create the HeterodynedCWSimulator object
   het = HeterodynedCWSimulator(par, det, times=times)

   # set the updated parameters
   parupdate = deepcopy(par)
   parupdate["RAJ"] = pos.ra.rad + 0.001
   parupdate["DECJ"] = pos.dec.rad - 0.001
   parupdate["F"] = [123.456789, -9.87654321e-12]  # different frequency and first derivative

   # get the model complex strain time series
   model = het.model(parupdate, updateSSB=True)

   # do the same but using Tempo2
   hettempo2 = HeterodynedCWSimulator(par, det, times=times, usetempo2=True)
   modeltempo2 = hettempo2.model(parupdate)

Overplotting these (the Tempo2 version is shown as black dashed lines) we see:

.. code-block:: python

   from cwinpy import HeterodynedData

   hetdata = HeterodynedData(model, times=times, detector=det, par=parupdate)
   fig = hetdata.plot(which="both")
   ax = fig.gca()
   
   # over plot Tempo2 version
   ax.plot(times, modeltempo2.real, "k--")
   ax.plot(times, modeltempo2.imag, "k--")
   fig.tight_layout()

.. thumbnail:: images/signal_comparison_example.png
   :width: 600px
   :align: center

.. note::

   For signals from sources in binary systems there will be a phase offset between signals
   heterodyned using the default `LALSuite <https://lscsoft.docs.ligo.org/lalsuite/>`_  functions
   and those using Tempo2. This offset is not present for non-binary sources. In general this is not
   problematic, but if using Tempo2 it will mean that the recovered initial phase will not be
   consistent with the expected value.

In this example using Tempo2 is several hundred times slower, but still runs in about a hundred ms
rather than a couple of hundred μs.

Signal API
==========

.. automodule:: cwinpy.signal
   :members:

.. autoclass:: cwinpy.signal.HeterodynedCWSimulator
   :members:

Signal references
=================

.. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
    <https://arxiv.org/abs/1705.08978v1>`_ (2017).

.. [2] `G. B. Hobbs, R. T. Edwards, R. N. Manchester
    <https://ui.adsabs.harvard.edu/abs/2006MNRAS.369..655H/abstract>`_, *MNRAS*, **369**, 2 (2006).
