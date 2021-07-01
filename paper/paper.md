---
title: 'CWInPy: a Python package for inference with continuous gravitational-wave signals from pulsars'
tags:
  - Python
  - pulsars
authors:
  - name: Matthew Pitkin
    orcid: 0000-0003-4548-526X
    affiliation: 1
affiliations:
  - name: Department of Physics, Lancaster University, Lancaster, UK, LA1 4YB
    index: 1
date: 30 June 2021
bibliography: paper.bib
---

# Summary

Continuous quasi-monochromatic gravitational-waves are expected to be emitted from non-axisymmetric
rapidly rotating neutron stars. There are thought to be on the order of $10^9$ neutron stars within
the Milky Way. At the time of writing, around 3000 such stars, known as pulsars, have been found
through electromagnetic (and primarily radio) observations of their highly regular pulsing profiles
[@ATNF]. A proportion of these pulsars with rotation frequencies $\gtrsim 10\,$Hz make enticing
targets for the current and future generation of ground-based gravitational-wave observatories such
as LIGO, Virgo and KAGRA. The detection of gravitational-waves from such a source would reveal
information on the size of the non-axisymmetry, or more colloquially the "mountain", on the star.
This in turn provides valuable and novel information on neutron star matter and structure.

To detect the very weak gravitational-wave signal from a pulsar requires the coherent integration of
long (months-to-years) gravitational-wave data sets including multiple detectors. CWInPy implements
an analysis pipeline enabling the user search for and characterise these signals. It implements the
preprocessing of gravitational-wave detector data for a user-provided set of pulsars; this entails
heterodyning the data with the expected signal phase evolution, followed by aggressive filtering and
down-sampling of the data [@2005PhRvD72j2002D]. These much compressed datasets are then used to
perform Bayesian inference on the unknown signal parameters including the gravitational-wave
amplitude.

CWInPy can be used to perform this full pipeline or implement the data preprocessing stages and
inferences stages seperately. These are all accessible via convenient command line executables
(making use of configuration files) or equivalently through a Python API. The pipelines provided by
CWInPy can perform analyses on an individual machine, but the amount of data being processed
generally requires that the preprocessing be parallelised over multiple machines. Therefore, CWInPy
will create jobs submittable over a computing cluster running the HTCondor job managment system.
These can also be run over the Open Science Grid. 

For the heterodyne preprocessing stage, CWInPy makes use of algorithms written in `C` within
LALSuite [@lalsuite] to calculate the phase evolution of each pulsar, which must account for the
changing rotation frequency and Doppler and relativistic effects related to the position and motion
of the detector with respect to the pulsar. These are accessible within Python through a SWIG
interface to LALSuite [@WETTE2020100634]. The Bayesian inference stage makes use of the
[`bilby`](https://lscsoft.docs.ligo.org/bilby/) package [@2019ApJS24127A], which provides a
convienent interface to a wide variety of packages for using the MCMC or nested sampling algorthm.
By default CWInPy uses the [`dynesty`](https://dynesty.readthedocs.io/) package
[@2020MNRAS.493.3132S] for inference, producing both posterior probability distributions for the
parameters of interest and the Bayesian evidence for the signal model given the data.

CWInPy is designed to supercede the current analysis pipeline, known as `lalapps_knope`
[@2017arXiv170508978P], largely based on executables written in `C`. There reasons and enhancements
behind CWInPy's development:

* the Python API allows easy access to a wider range of functionality and can be used 
* the intermediate data products (the heterodyned data files) have their own `HeterodynedData`
  class, based on a GWPy `TimeSeries` [@MACLEOD2021100657], and can be written to and read from HDF5
  files, which includes full information on the data provenance; 
* future integration to use models within standard pulsar timing packages such as, TEMPO2 (via the
  `libstempo` Python package), or `PINT`, will be simpler.
* the use of `bilby` allows easy access to a wide range of packages for Bayesian inference;
* the pipeline has been design so that it can be run over the Open Science Grid;
* it provides tools to generate simulated signals injected in real or fake data for individual
  sources, or user specified populations of sources;
* it provides tools to perform hierarchical inference in the underlying ellipticity distribution of
  a population of pulsars [@2018PhRvD98f3001P]. 

# Validation

To validate CWInPy extensive testing against previous codes has been performed and is provided with
the documentation. CWInPy has been used to successfully extract simulated signals directly added
into the LIGO data as well as multiple signals simulated via software.

The inference of source parameters has been validated through the use of simulation-based
calibration, with the posteriors probability distributions provided well-calibrated credible
intervals.

# Availability, documentation and development

CWInPy is installable under Linux and MacOS via `pip` through
[PyPI](https://pypi.org/project/cwinpy/) or using `conda` via
[conda-forge](https://anaconda.org/conda-forge/cwinpy). Full documentation of the exectubles and
Python API is avialable on [Read the Docs](https://cwinpy.readthedocs.io/). Development is currently
performed on a private repository, but the master branch is mirrored on GitHub. Feedback, bug
reports, or development suggestions are welcome and can be contributed via Github
[issues](https://github.com/cwinpy/cwinpy/issues), the [discussion
forum](https://github.com/cwinpy/cwinpy/discussions), or via
[email](mailto:contact+cw-software-cwinpy-3315-issue-@support.ligo.org).

# References
