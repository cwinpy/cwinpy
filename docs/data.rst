#############
Data handling
#############

Classes to deal with data products from searches for continuous sources of gravitational waves that
can be used for inference.

Heterodyned Data
================

The data product used for source parameter inference in the Bayesian targeted pulsar searches
perform by the `LIGO Scientific Collaboration <https://www.ligo.org/>`_ and `Virgo Collaboration
<http://www.virgo-gw.eu/>`_ (e.g., [1]_, [2]_) is a filtered and band-passed complex time series
[3]_.

In forming the data it assumes that the source phase evolution is described by a Taylor expansion of
the form:

.. math::

   \phi(t') = 2 \pi C\left(f_0(t'-T_0) + \frac{1}{2}\dot{f}_0(t'-T_0)^2 + \dots\right)

where :math:`f_0` is the source rotation frequency, :math:`\dot{f}_0` is the first frequency
derivative (higher order derivatives can also be used), :math:`t'` is the time in a frame at rest
with respect to the pulsar (the pulsar proper time, which for isolated pulsars is assumed to be
consistent with the solar system barycentre), :math:`T_0` is the time reference epoch in the pulsar
proper time, and :math:`C` is an emission mechanism-dependent scaling factor (often set to 2, for
emission from the :math:`l=m=2` quadrupole mode.) In searches for known pulsars, the frequency, its
derivatives, and the source position are assumed to be known from the observed properties of the
pulsar.

The raw real time series of :math:`h(t)` from a detector is "heterodyned",

.. math::

   h'(t) = h(t) e^{-2\pi i \phi(t + \Delta t(t))},

to remove the high-frequency variation of the potential source signal, where :math:`\Delta t(t)` is
a time- and source-and-position-dependent delay term that accounts for the Doppler and relativistic
delays between the time at the detector and the pulsar proper time (see, e.g., Equation 1 of [4]_).
The resulting complex series :math:`h'(t)` is low-pass filtered (effectively a band-pass filter on
the two-sided complex data) using a high order Butterworth filter, often with a knee frequency of
0.25 Hz, and down-sampled, via averaging, to a far lower sample rate than the original raw data. In
general gravitational-wave detector data is sampled at 16384 Hz, and often the heterodyned data is
downsampled to 1/60 Hz (one sample per minute).

Reading and writing
-------------------

The :class:`~cwinpy.data.HeterodynedData` object can be written to and read from both ASCII and HDF5
file formats. The ASCII read/writer is mainly to access legacy data as produced by the
`lalapps_heterodyne_pulsar` code within `LALSuite <https://doi.org/10.7935/GT1W-FZ16>`_. However, it
is advisable to use the HDF5 routines if purely using CWInPy for analyses. The
:class:`~cwinpy.data.HeterodynedData` class has both a :meth:`~cwinpy.data.HeterodynedData.read`
and :meth:`~cwinpy.data.HeterodynedData.write` method based on those for a GWPy `TimesSeries
<https://gwpy.github.io/docs/stable/timeseries/io.html>`_ object.

ASCII
^^^^^

The :class:`~cwinpy.data.HeterodynedData` class will automatically detect an ASCII file provided it
has the file extension ``.txt`` or, if it is gzipped, ``.txt.gz``. Any comments in the file taken
from lines starting with a ``#`` or ``%`` will be stored in the object. The format of data required
in an ASCII text file is shown for with the :class:`~cwinpy.data.HeterodynedData` API below. Given a
text file called ``data.txt``, it could either be read in as:

.. code-block:: python

    >>> from cwinpy.data import HeterodynedData
    >>> data = HeterodynedData("data.txt")

or using the :meth:`~cwinpy.data.HeterodynedData.read` method with:

.. code-block:: python

    >>> data = HeterodynedData.read("data.txt")

If you have an instance of a :class:`~cwinpy.data.HeterodynedData` class called, e.g., ``het``, then
writing to an ASCII file can be done using the :meth:`~cwinpy.data.HeterodynedData.write` method
with:

.. code-block:: python

    >>> het.write("data.txt")

If the file you are writing to already exists it will be overwritten.

A disadvantage of using the ASCII format is that you cannot stored additional metadata such as the
detector, pulsar parameter file used, and any injected signal, whereas this *will* be stored in a HDF5
file.

HDF5
^^^^

`HDF5 <https://support.hdfgroup.org/HDF5/doc/H5.intro.html#Intro-WhatIs>`_ is a flexible binary
storage format. In the context of a :class:`~cwinpy.data.HeterodynedData` object, it will store both
the complex time series of data and metadata about the object, such as the detector and contents of
the pulsar parameter file used for heterodyning. The HDF5 file will be detected if it has the file
extension ``.hdf``, ``.hdf5``, or ``.h5``. A HDF5 file called ``data.hdf5`` can be read in with,
e.g.:

.. code-block:: python

    >>> from cwinpy.data import HeterodynedData
    >>> data = HeterodynedData("data.hdf5")

or

.. code-block:: python

    >>> data = HeterodynedData.read("data.hdf5")

If you want to apply non-default keyword arguments for processing the data after it's read in, e.g.,
setting a window for the :meth:`~cwinpy.data.HeterodynedData.compute_running_median` method, then
the former method is recommended as the standard keywords can be passed to the class. Otherwise, you
have to explicitly run the processing methods that you require.

If you have an instance of a :class:`~cwinpy.data.HeterodynedData` class called, e.g., ``het``, then
writing to a HDF5 file can be done using the :meth:`~cwinpy.data.HeterodynedData.write` method
with:

.. code-block:: python

    >>> het.write("data.hdf5")

The `"Dataset" <http://docs.h5py.org/en/stable/high/dataset.html#>`_ within the HDF5 file within
which the data is stored is called ``HeterodynedData``. So, the data could be read in just using the
Python `h5py <https://www.h5py.org/>`_ package using:

.. code-block:: python

    >>> import h5py
    >>> hfile = h5py.File("data.hdf5", "r")
    >>> dataset = hfile["HeterodynedData"]
    >>> # the time series data can be found in:
    >>> timeseries = dataset[()]
    >>> # the metadata can be found as a dictionary with:
    >>> metadata = dict(dataset.attrs)
    >>> hfile.close()

If the file you are writing to already exists the, by default it will not be overwritten and an
exception will be raised. To overwrite a file use:

.. code-block:: python

    >>> het.write("data.hdf5", overwrite=True)

Combining data
--------------

Merging data
^^^^^^^^^^^^

If you have multiple heterodyned data files for a given detector and pulsar it might make sense to
combine them into one :class:`~cwinpy.data.HeterodynedData` object. The
:meth:`~cwinpy.data.HeterodynedData.read` method can in fact be passed a list of heterodyned data
files, and provided that they are compatible (i.e., they are for the same detector, the same pulsar,
and the same frequency scale factor) they will be merged into one using the
:meth:`~cwinpy.data.HeterodynedData.merge` method. By default the resulting object will be sorted in
ascending time order even if the input files were not sorted in time order. If you had two files
``data1.hdf5`` and ``data2.hdf5`` they could be read in and merged with:

.. code-block:: python

    >>> from cwinpy.data import HeterodynedData
    >>> data = HeterodynedData.read(["data1.hdf5", "data2.hdf5"])

Data from multiple detectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may want to store data for one pulsar, but from multiple detectors, or multiple frequency scale
factor data streams. This can be done using the :class:`~cwinpy.data.MultiHeterodynedData` class.
You can use the :meth:`~cwinpy.data.MultiHeterodynedData.add_data` method to add in multiple data
sets, which are stored in a dictionary-like format keyed by the detectors. If you had heterodyned
data files for H1 and L1 called ``dataH1.hdf5`` and ``dataL1.hdf5``, respectively, they could be
stored with:

.. code-block:: python

    >>> from cwinpy.data import MultiHeterodynedData
    >>> data = MultiHeterodynedData()
    >>> data.add_data("dataH1.hdf5")
    >>> data.add_data("dataL1.hdf5")

This type of object is used when performing :ref:`parameter estimation<Known pulsar parameter
estimation>` for a single pulsar with data from multiple detectors.

Data API
--------

.. autoclass:: cwinpy.data.MultiHeterodynedData
   :members:

.. autoclass:: cwinpy.data.HeterodynedData
   :members:

Data references
===============

.. [1] `J. Aasi et al
   <https://ui.adsabs.harvard.edu/#abs/2014ApJ...785..119A/abstract>`_,
   *Astrophys. J.*, **785**, 119 (2014)

.. [2] `B. P. Abbott et al
   <https://ui.adsabs.harvard.edu/#abs/2017ApJ...839...12A/abstract>`_,
   *Astrophys. J.*, **839**, 12 (2017)

.. [3] `R. Dupius & G. Woan
   <https://ui.adsabs.harvard.edu/#abs/2005PhRvD..72j2002D/abstract>`_,
   *Phys. Rev. D*, **72**, 102002 (2005)

.. [4] `G. Hobbs, R. T. Edwards, R. N. Manchester
   <https://ui.adsabs.harvard.edu/abs/2006MNRAS.369..655H/abstract>`_,
   *Mon. Not. R. Astron. Soc.*, **369**, 655 (2006)

.. [5] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
    <https:arxiv.org/abs/1705.08978v1>`_ (2017)

.. [6] https://github.com/joferkington/oost_paper_code/blob/master/utilities.py and
   https://stackoverflow.com/a/22357811/1862861

.. [7] Boris Iglewicz and David Hoaglin (1993), `"Volume 16: How to Detect and
   Handle Outliers"
   <https://hwbdocuments.env.nm.gov/Los%20Alamos%20National%20Labs/TA%2054/11587.pdf>`_,
   The ASQC Basic References in Quality Control:
   Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
