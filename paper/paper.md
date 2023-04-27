---
title: 'CWInPy: A Python package for inference with continuous gravitational-wave signals from pulsars'
tags:
  - gravitational waves
  - Python
  - pulsars
authors:
  - name: Matthew Pitkin
    orcid: 0000-0003-4548-526X
    affiliation: "1, 2"
affiliations:
  - name: Department of Physics, Lancaster University, Lancaster, UK, LA1 4YB
    index: 1
  - name: School of Physics and Astronomy, University of Glasgow, University Avenue, Glasgow, UK, G12 8QQ
    index: 2
date: 6 June 2022
bibliography: paper.bib
---

# Summary

Continuous quasi-monochromatic gravitational-waves are expected to be emitted from non-axisymmetric
rapidly rotating neutron stars [see, e.g., @2022arXiv220606447R for a review]. There are thought to
be on the order of $10^8 - 10^9$ neutron stars within the Milky Way [@2010A&A510A23S]. At the time
of writing, around 3000 such stars, have been found through electromagnetic (primarily radio)
observations of their highly regular pulsing profiles [@ATNF]. These are known as pulsars. A
proportion of these pulsars with rotation frequencies $\gtrsim 10\,$Hz make enticing targets for the
current and future generation of ground-based gravitational-wave observatories such as LIGO, Virgo
and KAGRA [@2019ApJ87910A]. The detection of gravitational-waves from such a source would reveal
information on the size of any non-axisymmetry, or more colloquially a "mountain", on the star.
This in turn provides valuable and novel information on neutron star matter and structure [see,
e.g., @2015Lasky].

Detecting the very weak gravitational-wave signal from a pulsar requires the coherent integration of
long (months-to-years) gravitational-wave data sets from multiple detectors. CWInPy implements an
analysis pipeline enabling the user to search for and characterise these signals. It implements the
preprocessing of the time-domain strain, $h(t)$, measured by gravitational-wave detectors for a
user-provided set of pulsars [defined through Tempo2-style [parameter
files](https://cwinpy.readthedocs.io/en/latest/pipelines.html#source-parameter-specification), @tempo2; @tempo2manual]; this
entails heterodyning the data with the expected signal phase evolution, followed by aggressive
filtering and down-sampling of the data [@2005PhRvD72j2002D]. These much compressed datasets are
then used to perform Bayesian inference on the unknown signal parameters, including the
gravitational-wave amplitude.

CWInPy can be used to perform this full pipeline or implement the data preprocessing stages and
inferences stages separately. These are all accessible via convenient command line executables
(making use of configuration files) or, equivalently, through a Python API. The pipelines provided by
CWInPy can perform analyses on an individual machine, but the amount of data being processed
generally requires that the preprocessing be parallelised over multiple machines. Therefore, CWInPy
will create jobs that can be submitted over a computing cluster running the
[HTCondor](https://htcondor.readthedocs.io/) job management system [@condor-practice] via the
[`htcondor`](https://htcondor.readthedocs.io/en/latest/) package. These can also be run over the
[Open Science Grid](https://opensciencegrid.org/) [@Pordes_2007] using open gravitational-wave data provided via
the [Gravitational-wave Open Science Center](https://gwosc.org/) (GWOSC)
[@2015JPhCS.610a2021V; @RICHABBOTT2021100658].

For the heterodyne preprocessing stage, CWInPy makes use of algorithms written in `C` within
LALSuite [@lalsuite] to calculate the phase evolution of each pulsar, which must account for the
slowly changing rotation frequency and include Doppler and relativistic effects related to the
position and motion of the detector with respect to the pulsar. These are accessible within Python
through a SWIG interface to LALSuite [@WETTE2020100634]. The Bayesian inference stage makes use of
the [`bilby`](https://lscsoft.docs.ligo.org/bilby/) package [@2019ApJS24127A], which provides a
convenient interface to a wide variety of packages for using the Markov Chain Monte Carlo [MCMC;
see, e.g., @2017ARA&A55213S for a review of MCMC with particular reference to astronomy] or nested
sampling algorithms [see, e.g., @nestedsampling; @2021arXiv210109675B for reviews]. By default,
CWInPy uses the [`dynesty`](https://dynesty.readthedocs.io/) package [@2020MNRAS.493.3132S] for
inference using nested sampling, producing both posterior probability distributions for the
parameters of interest and the Bayesian evidence for the data given the signal model.

# Statement of need

CWInPy is designed to supersede the current analysis pipeline, known as `lalapps_knope`
[@2017arXiv170508978P], largely based on executables written in `C`, which has been used for several
searches in LIGO and Virgo data [@2017ApJ83912A; @2019ApJ87910A; @2020ApJ902L21A; @2021arXiv211113106T].
The reasons behind CWInPy's development, and its enhancements over existing software, include:

* the Python API allows easy access to the full range of functionality from data preprocessing to
  source parameter estimation, with greater ability to control various aspects of the analysis;
* the parallelisation of the analysis for multiple pulsars over long observing runs has been changed
  to improve its efficiency and robustness;
* the intermediate preprocessed data products (the heterodyned data files) have their own
  `HeterodynedData` class, based on a [GWPy](https://gwpy.github.io)
  [`TimeSeries`](https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries.html)
  [@MACLEOD2021100657], and can be written to and read from
  [HDF5](https://support.hdfgroup.org/HDF5/whatishdf5.html) files; this class includes full
  information on the data provenance, useful plotting routines in both time- and frequency-domain,
  and access to data statistics;
* the use of [`bilby`](https://lscsoft.docs.ligo.org/bilby/) for the signal parameter estimation
  allows easy access to a wide range of packages for Bayesian inference;
* the pipeline can be run over the Open Science Grid, allowing a wider range of computing resources
  to be used;
* the pipeline can make use of open gravitational-wave data provided over
  [CVMFS](https://gwosc.org/cvmfs/) from the [Gravitational-wave Open Science Center
  (GWOSC)](https://gwosc.org/) [@2015JPhCS.610a2021V; @RICHABBOTT2021100658];
* it provides tools to generate simulated signals injected in real or fake data for individual
  sources, or for user specified populations of sources;
* it provides tools to perform hierarchical inference on the underlying ellipticity distribution of
  a population of pulsars [@2018PhRvD98f3001P];
* it provides tools for determining the significance of a recovered signal using the sky-shifting
  method [@2020PhRvD.102l3027I];
* the pulsar phase evolution can be calculated using the standard pulsar timing package Tempo2
  [@tempo2] via the [`libstempo`](https://github.com/vallis/libstempo) Python package
  [@2020ascl.soft02017V] in addition to being able to use the independent routines within LALSuite
  [@lalsuite].

# Validation

To validate CWInPy, extensive testing against previous codes has been performed and is provided with
the [documentation](https://cwinpy.readthedocs.io/en/latest/). CWInPy has been used to successfully
extract simulated signals directly added into the LIGO data [@2017PhRvD95f2002B] as well as multiple
signals simulated via software.

The inference of source parameters has been validated through the use of simulation-based
calibration [@2018arXiv180406788T], with the posterior probability distributions shown to provide
well-calibrated [@wellcalibrated] credible intervals.

As evidence for this consistency, \autoref{fig:posteriors} shows the posterior probability
distributions—plotted using [matplotlib](https://matplotlib.org/) [@matplotlib], and
[corner.py](https://corner.readthedocs.io/) [@corner], via an interface with PESummary, [@pesummary]—for the four unknown parameters of a simulated
gravitational-wave signal from a pulsar. The figure compares the posterior samples extracted using
CWInPy (with the [`dynesty`](https://dynesty.readthedocs.io/) sampler) with those calculated (again
with CWInPy) over a uniform grid in the parameters space and those produced by the previously used
`lalapps_pulsar_parameter_estimation_nested` code [@2017arXiv170508978P]. Consistent posteriors, and
also Bayesian odds values comparing the evidence for the signal model versus the data containing
pure noise, are found.

![The posterior probability distributions for the parameters of a simulated gravitational-wave signal from a pulsar injected into Gaussian noise.\label{fig:posteriors}](multi_detector_software_injection_linear_corner.png)

In \autoref{fig:heterodyne} CWInPy has been used to heterodyne simulated data containing a
gravitational-wave signal and an insignificant amount of noise (to show the the method does not
corrupt the signal in any way). The simulated data has been generated using software that is largely
independent of CWInPy. The solid lines show the heterodyned time series as produced by CWInPy,
whereas the dashed lines show the theoretical expectation for the heterodyned signal, which provide
a very good match.

![The real and imaginary components of the heterodyned time series for a simulated signal generated using the heterodyne pipeline in CWInPy.\label{fig:heterodyne}](example1_plot.png)

# Usage

The main pipelines provided by CWInPy are accessible using command line executables that require a
configuration file. The full pipeline, which must be run as an [HTCondor directed acyclic
graph](https://htcondor.readthedocs.io/en/latest/users-manual/dagman-workflows.html) (DAG), can be
run with the `cwinpy_knope_pipeline` executable. The heterodyne preprocessing stage can be run using
`cwinpy_heterodyne` or, if running for long stretches of data and multiple pulsars, using a HTCondor
DAG via `cwinpy_heterodyne_pipeline`. The latter should be used for most practical purposes, while
the former is mainly useful for testing purposes. The parameter estimation stage can be run using
`cwinpy_pe` or, if running for multiple pulsars, using an HTCondor DAG via `cwinpy_pe_pipeline`.
Full details of all the required configuration file settings are given in the
[documentation](https://cwinpy.readthedocs.io/).

## Quick setup

Both the `cwinpy_knope_pipeline` and the `cwinpy_heterodyne_pipeline` executables have several
command line arguments that can be used for quickly setting up analyses using open data from GWOSC.
These rely on the user having access to a computer, or cluster of computers, with HTCondor installed
and CVMFS set up with access to the data. To run the analysis the user just needs to have
Tempo2-style pulsar ephemeris files for any pulsars they wish to search for. If one had an ephemeris
file for, e.g., PSR J0740+6620, called `J0740+6620.par`, then the pipeline could be run over all
data from the first observing run of Advanced LIGO (O1) [@RICHABBOTT2021100658], using

```bash
$ cwinpy_knope_pipeline \
--run O1 \
--pulsar J0740+6620.par \
--output /home/usr/analysis
```

where `/home/usr/analysis` can be substituted for the required final location of the files output by
the analysis.

If you do not have access to a pulsar ephemeris file, an ephemeris from the [ATNF Pulsar
Catalogue](https://www.atnf.csiro.au/research/pulsar/psrcat/) [@ATNF] can be used by instead just
specifying the name of the pulsar (you have to trust that the ephemeris provides a coherent timing
solution over the gravitational-wave data period). The ephemeris is extracted from the catalogue
using [`psrqpy`](https://psrqpy.readthedocs.io/) [@psrqpy]. For example, to search for PSR
J0737-3039A using the [ATNF Pulsar Catalogue](https://www.atnf.csiro.au/research/pulsar/psrcat/)
data in LIGO data from the second observing run, one could use:

```bash
$ cwinpy_knope_pipeline \
--run O2 \
--pulsar J0737-3039A \
--output /home/usr/analysis
```

In both the above cases the `--pulsar` command can be given multiple times to input multiple
pulsars. With these "quick setup" options there is no further control over the pipeline, so default
parameters are used in all cases. If more control is required then using a configuration file is
highly recommended.

The quick setup can be used to analyse all the signal hardware injections for each observation run
[@2017PhRvD95f2002B] by supplying the `--hwinj` command instead of `--pulsar`.

# Availability, documentation and development

CWInPy is installable under Linux and MacOS using `pip` via [PyPI](https://pypi.org/project/cwinpy/)
or using `conda` via [conda-forge](https://anaconda.org/conda-forge/cwinpy). Full documentation of
the executables and Python API is available on [Read the Docs](https://cwinpy.readthedocs.io/).
Development is currently performed in the
[git.ligo.org/cwinpy/cwinpy](https://git.ligo.org/cwinpy/cwinpy) git repository, which is openly
viewable, but for which write access is only available within the LIGO-Virgo-KAGRA collaborations.
The master branch is also mirrored on [GitHub](https://github.com/cwinpy/cwinpy). Feedback, bug
reports, or development suggestions are welcome and can be contributed via Github
[issues](https://github.com/cwinpy/cwinpy/issues), the [discussion
forum](https://github.com/cwinpy/cwinpy/discussions), or via
[email](mailto:contact+cw-software-cwinpy-3315-issue-@support.ligo.org).

# Acknowledgements

The author acknowledges support from the UK Science & Technology Facilities Council under grant
number [ST/V001213/1](https://gtr.ukri.org/projects?ref=ST%2FV001213%2F1). The development of this
package relies on the original work of Réjean Dupuis and Graham Woan and also has benefited
massively from discussions with the LIGO-Virgo-KAGRA continuous waves working group and the
developers of the bilby software package.

# References
