###############
Code comparison
###############

Here we document comparisons between the analysis produced using CWInPy compared with the
`LALSuite
<https://git.ligo.org/lscsoft/lalsuite>`_ code ``lalapps_pulsar_parameter_estimation_nested``,
and the full pipeline generation code ``lalapps_knope``, which
are described in [1]_.

The codes will be run on identical data and in configurations that are as close as possible.

Parameter estimation comparison
===============================

Links to the various direct comparisons of pulsar parameter estimation are given below:

.. toctree::
   :maxdepth: 1

   single-detector-noise-only
   single-detector-software-injection-linear
   single-detector-software-injection-circular
   multi-detector-noise-only
   multi-detector-software-injection-linear
   multi-detector-software-injection-circular
   single-detector-noise-only-two-harmonics
   single-detector-software-injection-two-harmonics
   single-detector-O1-data
   single-detector-O1-data-restricted-prior
   single-detector-O1-data-hardware-injection
   multi-detector-O1-data
   multi-detector-O1-data-hardware-injection

Posterior credible interval checks
==================================

Another test of the code is to check that the posterior credible intervals resulting from any
analysis correctly ascribe probability, i.e., that they are "well calibrated" [2]_. To do this one
can create a set of simulated signal with true parameters drawn from a particular prior, and then
use the code to sample the posterior probability distribution over the parameter space for each
parameter using the *same* prior. Using the one-dimensional marginalised posteriors for each
parameter, and each simulation, one can then find the credible interval in which the known true
signal parameter lies. If the credible intervals are correct you would expect, e.g., to find the
true parameter in the 1% credible interval for 1% of the simulations, the true parameter within the
50% credible interval for 50% of the simulations, etc. A plot of the credible interval versus the
percentage of true signal values found within that credible interval is known within the
gravitational-wave community colloquially as a "PP plot" (see, e.g., Section VC of [3]_). This is
also more generally known as "Simulated-based calibration" [4]_.

These tests have been performed in the output of ``cwinpy_knope`` by generating a set of simulated
signals (using the :class:`cwinpy.pe.testing.PEPPPlotsDAG`) to be analysed. After all the
individual simulations have been analysed a PP plot is generated using :meth:`cwinpy.pe.testing.
generate_pp_plots` (which itself uses functions from `bilby
<https://lscsoft.docs.ligo.org/bilby/index.html>`_).

Single harmonic signal
----------------------

We produce PP plots for the case of a signal from the :math:`l=m=2` mass quadrupole of a pulsar,
where emission would be at twice the rotation frequency and defined by the parameters :math:`h_0`,
:math:`\phi_0`, :math:`\iota` and :math:`\psi`. A Python file to run such an analysis for 250
simulated signals using :class:`~cwinpy.pe.testing.PEPPPlotsDAG` is shown below_. This also
shows the priors used for the generation of signal parameters and their recovery.

.. note::

   When drawing parameters from the :math:`h_0` prior a maximum cut-off (``maxamp`` in the code)
   that is lower than the upper range of the prior is used. This is to ensure that posteriors are
   not truncated by the upper end of the prior in this case. However, this should not bias the
   recovered credible intervals due to the :math:`h_0` prior being uniform and extended well above
   the ``maxamp`` value.

.. thumbnail:: ppplot_2f.png
   :width: 600px
   :align: center

The distributions of signal-to-noise ratios for these simulations is:

.. thumbnail:: snrs_2f.png
   :width: 400px
   :align: center

Dual harmonic signal
--------------------

We produce PP plots for the case of a signal from the :math:`l=2, m=1,2` mass quadrupole of a
pulsar, where emission would be at both once and twice the rotation frequency and defined by the
parameters :math:`C_{12}`, :math:`C_{22}`, :math:`\Phi_{12}`, :math:`\Phi_{22}`, :math:`\iota` and
:math:`\psi`. A Python file to run such an analysis for 250 simulated signals using
:class:`~cwinpy.knope.testing.KnopePPPlotsDAG` is shown below_. This also shows the priors used for
the generation of signal parameters and their recovery.

.. note::

   When drawing parameters from the :math:`C_{12}` and :math:`C_{22}` priors a maximum cut-off
   (``maxamp`` in the code) that is lower than the upper range of the prior is used. This is to
   ensure that posteriors are not truncated by the upper end of the prior in this case. However,
   this should not bias the recovered credible intervals due to the :math:`C_{12}` and
   :math:`C_{22}` priors being uniform and extended well above the ``maxamp`` value.

.. thumbnail:: ppplot_1f2f.png
   :width: 600px
   :align: center

The distributions of signal-to-noise ratios for these simulations is:

.. thumbnail:: snrs_1f2f.png
   :width: 400px
   :align: center

.. _below:

.. literalinclude:: scripts/pptest_one_harmonic.py
   :language: python
   :lines: 1-29,33-42,45-48,50-52

.. literalinclude:: scripts/pptest_two_harmonics.py
   :language: python
   :lines: 1-45,49-58,61-64,66-68

Pipeline comparison
===================

We can compare the results of the full pipeline produced by the `LALSuite
<https://git.ligo.org/lscsoft/lalsuite>`_ code ``lalapps_knope`` with that produced using the CWInPy
code ``cwinpy_knope_pipeline``. We will do this comparison by analysing the set of :ref:`<Hardware
injections>` and analysis of real pulsar data using open data from the two LIGO detectors during the
`first advanced LIGO observing run <https://www.gw-openscience.org/O1/>`_ (O1). 

O1 hardware injections
----------------------

To analyse the 15 hardware injections in O1 using the ``lalapps_knope`` pipeline the following
configuration file (named ``lalapps_knope_O1injections.ini``) has been used:

.. literalinclude:: knope/lalapps_knope_O1injections.ini

In this case the included segment lists have been made using the following code:

.. code-block:: python

   from cwinpy.heterodyne import generate_segments
   from cwinpy.info import HW_INJ_SEGMENTS, RUNTIMES

   for det in ["H1", "L1"]:
       start, end = RUNTIMES["O1"][det]
       _ = generate_segments(
           starttime=start,
           endtime=end,
           includeflags=HW_INJ_SEGMENTS["O1"][det]["includesegments"],
           excludeflags=HW_INJ_SEGMENTS["O1"][det]["excludesegments"],
           usegwosc=True,
           writesegments=f"{det}segments.txt",
       )

This has then been submitted (on the `UWM Nemo computing cluster
<https://cgca.uwm.edu/datacenter.html>`_) with:

>>> lalapps_knope lalapps_knope_O1injections.ini

To perform the analysis using CWInPy, the :ref:`<Quick setup>` has been used:

>>> cwinpy_knope_pipeline --run O1 --hwinj --incoherent --output /home/matthew/cwinpy_knope/O1injections --accounting-group-tag ligo.dev.o1.cw.targeted.bayesian

.. note::

   Because these analyses used LVK computing resources the
   ``accounting_group`` / ``--accounting-group-tag`` inputs have had to be set.

In terms of `wall-clock time
<https://en.wikipedia.org/wiki/Elapsed_real_time#:~:text=Elapsed%20real%20time%2C%20real%20time,at%20which%20the%20task%20started.>`_
the ``lalapps_knope`` and ``cwinpy_knope_pipeline`` pipelines took 32 hours 8 mins and 13 hours 1
min, respectively (differences here could in part relate to availability of cluster nodes at the
time of running). In terms of total CPU hours used by all the jobs for the ``lalapps_knope`` and
``cwinpy_knope_pipeline`` pipelines these took approximately 43.4 days and 27.9 days, respectively.

.. note::

   To gather than total time for all the jobs in a Condor DAG I have used:

   .. code-block:: bash

      condor_history matthew -constraint "DAGManJobId == <dagman_id>" -limit <num> -af RemoteWallClockTime | paste -s -d+ - | bc

   where `<dagman_id>` is the ID of the DAGMan job, which can be found in the `.dagman.log` file.

Heterodyned data comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compare the heterodyned data we can look at the power spectra obtained using ``lalapps_knope``
and ``cwinpy_knope_pipeline``. The following code has been used to produce these spectra:

.. code-block:: python
    
    import lal
    from cwinpy import HeterodynedData
    from matplotlib import pyplot as plt

    lalappsbase = "/home/matthew/lalapps_knope/O1injections/{det}/JPULSAR{num}/data/fine/2f/fine-{det}-{span}.txt.gz"
    cwinpybase = "/home/matthew/cwinpy_knope/O1injections/{det}/heterodyne_JPULSAR{num}_{det}_2_{span}.hdf5"

    numbers = [f"{i:02d}" for i in range(15)]

    timespans = {
        "lalapps": {"H1": "1126051217-1137254417", "L1": "1126051217-1137254417"},
        "cwinpy": {"H1": "1129136736-1137253524", "L1": "1126164689-1137250767"},
    }

    for det in ["H1", "L1"]:
        fig, axs = plt.subplots(5, 3, figsize=(20, 18))

        # loop over pulsars
        for num, ax in zip(numbers, axs.flat):
            # read in heterodyned data
            ck = HeterodynedData(cwinpybase.format(det=det, num=num, span=timespans["cwinpy"][det]))
            lk = HeterodynedData(lalappsbase.format(det=det, num=num, span=timespans["lalapps"][det]))

            # plot median power spectrum of data
            lk.power_spectrum(remove_outliers=True, dt=int(lal.DAYSID_SI * 10), label="lalapps", lw=3, color="k", ax=ax)
            ck.power_spectrum(remove_outliers=True, dt=int(lal.DAYSID_SI * 10), label="cwinpy", alpha=0.8, ls="--", ax=ax)
            ax.set_title(f"PULSAR{num}")

            if int(num) < 12:
                ax.xaxis.set_visible(False)
    
        fig.tight_layout()
        fig.savefig(f"hwinj_comparison_spectrum_{det}.png", dpi=200)

giving the following spectra for H1:

.. thumbnail:: hwinj_comparison_spectrum_H1.png
   :width: 600px
   :align: center

and L1:

.. thumbnail:: hwinj_comparison_spectrum_L1.png
   :width: 600px
   :align: center

Injection parameter comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compare the final parameter estimation between the ``lalapps_knope`` and
``cwinpy_knope_pipeline`` to following code has been used:

.. code-block:: python

    import os
    from cwinpy.plot import Plot
    from cwinpy.info import HW_INJ
    from gwpy.plot.colors import GW_OBSERVATORY_COLORS

    basedir = "/home/matthew"

    cwinpydir = os.path.join(basedir, "cwinpy_knope", "O1injections", "results")
    lppndir = os.path.join(basedir, "lalapps_knope", "O1injections", "posterior_samples")

    for pulnum in range(15):
        pulsar = f"JPULSAR{pulnum:02d}"

        for det in ["H1", "L1"]:
            results = {}
            colors = {}

            for a in ["cwinpy", "lalapps"]:
                if a == "cwinpy":
                    resfile = os.path.join(cwinpydir, pulsar, f"cwinpy_pe_{det}_{pulsar}_result.hdf5")
                    colors[a] = GW_OBSERVATORY_COLORS[det]
                else:
                    resfile = os.path.join(lppndir, pulsar, det, "2f", f"posterior_samples_{pulsar}.hdf")
                    colors[a] = "grey"
                results[a] = resfile

            p = Plot(
                results,
                parameters=["h0", "iota", "phi0", "psi"],
                plottype="corner",
                pulsar=HW_INJ["O1"]["hw_inj_files"][pulnum],
                untrig="cosiota"
            )

            fig = p.plot(colors=colors)
            p.fig.suptitle(f"{det}: {pulsar}")
            p.save(f"{pulsar}_{det}_plot.png", dpi=200)

            fig.clf()
            fig.close()

which produces the following plots:

.. thumbnail:: JPULSAR00_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR00_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR01_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR01_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR02_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR02_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR03_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR03_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR04_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR04_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR05_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR05_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR06_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR06_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR07_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR07_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR08_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR08_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR09_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR09_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR10_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR10_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR11_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR11_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR12_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR12_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR13_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR13_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

.. thumbnail:: JPULSAR14_H1_plot.png
   :width: 275px
   :align: left
   :group: injections

.. thumbnail:: JPULSAR14_L1_plot.png
   :width: 275px
   :align: right
   :group: injections

These show very consistent posteriors produced by both codes, which extract the parameters as
expected. They would not be expected to be identical due to a range of differences including:
``lalapps_knope`` performed the heterodyne in two stages rather than the one used by
``cwinpy_knope_pipeline``; ``lalapps_knope`` uses a low-pass filter with knee frequency of 0.25 Hz
rather than the default 0.1 Hz used by ``cwinpy_knope_pipeline``; the outlier vetoing for the two
codes is different; the default "chunking" of the heterodyned data into stationary segments used
during parameter estimation is different between the codes.

Comparison References
=====================

.. [1]  M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
   <https:arxiv.org/abs/1705.08978v1>`_ (2017).
.. [2] `A. P. Dawid
   <https://www.tandfonline.com/doi/abs/10.1080/01621459.1982.10477856>`_, *Journal of the
   American Statistical Association*, **77**, 379 (1982).
.. [3] `J. Veitch et al.
   <https://ui.adsabs.harvard.edu/abs/2015PhRvD..91d2003V/abstract>`_,
   *PRD*, **91**, 042003 (2015)
.. [4] S. Talts, M. Betancourt, D. Simpson, A. Vehtari & A. Gelman, `arXiv:1804.06788
   <https:arxiv.org/abs/1804.06788>`_ (2018).
