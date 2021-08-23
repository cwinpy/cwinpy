Plotting results
================

The outputs of parameter estimation performed by CWInPy are in the form of bilby
:class:`~bilby.core.result.Result` or :class:`~bilby.core.grid.Grid` objects, for posterior samples
or posteriors evaluated on a grid, respectively. The :class:`~bilby.core.result.Result` object
itself has various methods for plotting the posteriors:

* :meth:`~bilby.core.result.Result.plot_corner`
* :meth:`~bilby.core.result.Result.plot_marginals`
* :meth:`~bilby.core.result.Result.plot_single_density`

However, CWInPy also provides a :class:`~cwinpy.plot.Plot` class for producing plots. This makes
use of various plotting functions from the `PESummary <https://docs.ligo.org/lscsoft/pesummary/>`_
package [1]_ and `corner.py <https://corner.readthedocs.io/en/latest/>`_ [2]_ and can overplot
posterior results from multiple different detectors/samplers. A selection of example plots are
shown below, which are selected in :class:`~cwinpy.plot.Plot` class using the ``plottype``
argument:

* "hist": produce a histogram of posterior samples for a single parameters (the ``kde`` option can
  be used to also overplot a kernel density estimate of the posterior);
* "kde": produce a kernel density estimate using posterior samples for a single parameter;
* "corner": produce what is known as a corner, or triangle, plot of posterior distributons for two or
  more parameters in this case produced with the
  `corner.py <https://corner.readthedocs.io/en/latest/>`_ [2]_ package;
* "triangle": a different style of triangle plot (as produce by the
  `PESummary <https://docs.ligo.org/lscsoft/pesummary/>`_ package [1]_) that can only be used for
  a pair of parameters;
* "reverse_triangle": another style of triangle plot (as produce by the
  `PESummary <https://docs.ligo.org/lscsoft/pesummary/>`_ package [1]_) that has the 1D
  distributions to the left/below rather than to the right/above the 2D distribution plot;
* "contour": a contour plot that can be produced only for a pair of parameters.

The examples are based on the outputs of the
:ref:`Multiple detectors, software injection (circular polarisation)` example comparing CWInPy with
the ``lalapps_pulsar_parameter_estimation_nested`` code [3]_.

Example: plotting 1D marginal posteriors 
----------------------------------------

If we have a single bilby :class:`~bilby.core.result.Result` containing posterior samples
saved to a file called ``cwinpy.hdf5`` and want to plot the posterior on the :math:`h_0`
parameter this can be done with:

.. code-block:: python

   from cwinpy.plot import Plot
   plot = Plot("cwinpy.hdf5", parameters="h0", plottype="hist", kde=True)
   plot.plot()
   plot.save("cwinpy.png", dpi=150)

which will produce:

.. thumbnail:: images/plotting_example1.png
   :width: 600px
   :align: center

In the above code the ``kde=True`` keyword was used to also plot a kernel density estimate (KDE) of
the posterior as well as a histogram. To just plot the histogram the ``kde`` keyword could be
removed. To just plot a KDE without the histogram ``plottype="kde"`` can instead be used.

Example: over-plotting multiple 1D posteriors
---------------------------------------------

If we have multiple posterior results (e.g., from different detectors or from runs using different
samplers) that we want to compare these can be overplotted on the same figure. To do this we can
pass a dictionary pointing to the various results. In this case we will overplot the
:math:`\cos{\iota}` posterior results of running on a simulated pulsar signal using CWInPy to
produce both posterior samples and an evaluation of the posterior on a grid, and using the
``lalapps_pulsar_parameter_estimation_nested`` code, with:

.. code-block:: python

   from cwinpy.plot import Plot
   plot = Plot(
       {"CWInPy": "cwinpy.hdf5", "lppen": "lppen.hdf5", "grid": "grid.json"},
       parameters="cosiota",
       plottype="kde",
       pulsar="simulation.par",
   )
   plot.plot()
   plot.save("cwinpy.png", dpi=150)

which will produce:

.. thumbnail:: images/plotting_example2.png
   :width: 600px
   :align: center

In this example the ``pulsar`` keyword has been supplied pointing to a TEMPO(2)-style pulsar
parameter file that contains the true *simulated* signal parameters, which are then overplotted
with a vertical black line. To not overplot the true values this argument would be removed.

Example: plotting 2D posteriors
-------------------------------

We can also produce plots of pairs of parameters that include 1D marginalised posterior
distributions and 2D joint posterior distributions (over-plotting multiple results as ):

Corner plot
^^^^^^^^^^^

A "corner" plot, as produced with the `corner.py <https://corner.readthedocs.io/en/latest/>`_ [2]_
package, of a pair of parameters (in this case :math:`h_0` versus :math:`\cos{\iota}`) can be
generated with, e.g.,:

.. code-block:: python

   from cwinpy.plot import Plot
   plot = Plot(
       {"CWInPy": "cwinpy.hdf5", "lppen": "lppen.hdf5", "grid": "grid.json"},
       parameters=["h0", "cosiota"],
       plottype="corner",
       pulsar="simulation.par",
   )
   plot.plot()
   plot.save("cwinpy.png", dpi=150)

which will produce:

.. thumbnail:: images/plotting_example3.png
   :width: 600px
   :align: center

.. note::

    By default, a 2D distribution from any results in a :class:`~bilby.core.grid.Grid` will not be
    included. To add these the :meth:`~cwinpy.plot.Plot.plot` method can be passed the
    ``grid2d=True`` argument.

Triangle plot
^^^^^^^^^^^^^

A different style of corner/triangle plot can be generated with, e.g.,:

.. code-block:: python

   from cwinpy.plot import Plot
   plot = Plot(
       {"CWInPy": "cwinpy.hdf5", "lppen": "lppen.hdf5", "grid": "grid.json"},
       parameters=["h0", "cosiota"],
       plottype="triangle",
       pulsar="simulation.par",
   )
   plot.plot()
   plot.save("cwinpy.png", dpi=150)

and will produce:

.. thumbnail:: images/plotting_example4.png
   :width: 600px
   :align: center

.. note::

   This plot may take a bit of time to produce due to performing a bounded 2D kernel density
   estimate.

Reverse triangle plot
^^^^^^^^^^^^^^^^^^^^^

Another style of corner plot, with the 1D marginal distributions shown below the 2D joint posterior
plot, is a "reverse triangle" plot, which can be generated with, e.g.,:

.. code-block:: python

   from cwinpy.plot import Plot
   plot = Plot(
       {"CWInPy": "cwinpy.hdf5", "lppen": "lppen.hdf5", "grid": "grid.json"},
       parameters=["h0", "cosiota"],
       plottype="reverse_triangle",
       pulsar="simulation.par",
   )
   plot.plot()
   plot.save("cwinpy.png", dpi=150)

and will produce:

.. thumbnail:: images/plotting_example5.png
   :width: 600px
   :align: center

Contour plot
^^^^^^^^^^^^

A plot just containing the 2D posterior distribution, without the 1D marginal distributions, can be
produced with:

.. code-block:: python

   from cwinpy.plot import Plot
   plot = Plot(
       {"CWInPy": "cwinpy.hdf5", "lppen": "lppen.hdf5", "grid": "grid.json"},
       parameters=["h0", "cosiota"],
       plottype="contour",
       pulsar="simulation.par",
   )
   plot.plot()
   plot.save("cwinpy.png", dpi=150)

and will produce:

.. thumbnail:: images/plotting_example6.png
   :width: 600px
   :align: center

Example: plotting multiple posteriors
-------------------------------------

If wanting to plot posteriors for more than two parameters then only the "corner" plot option can
be used. If we wanted to plot :math:`h_0`, :math:`\cos{\iota}`, :math:`\psi` and :math:`\phi_0`
we could do:

.. code-block:: python

   from cwinpy.plot import Plot
   plot = Plot(
       {"CWInPy": "cwinpy.hdf5", "lppen": "lppen.hdf5", "grid": "grid.json"},
       parameters=["h0", "cosiota", "psi", "phi0"],
       plottype="corner",
       pulsar="simulation.par",
   )
   plot.plot()
   plot.save("cwinpy.png", dpi=150)

to produce:

.. thumbnail:: images/plotting_example7.png
   :width: 600px
   :align: center

Plotting API
------------

.. autoclass:: cwinpy.plot.Plot
   :members:

Plotting references
-------------------

.. [1] `C. Hoy & V. Raymond <https://doi.org/10.1016/j.softx.2021.100765>`_, *SoftwareX*, **15**, 100765 (2021)

.. [2] `D. Foreman-Mackey <https://doi.org/10.21105/joss.00024>`_, *JOSS*, **1**, 2, 24, (2016)

.. [3]  M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
   <https:arxiv.org/abs/1705.08978v1>`_ (2017)