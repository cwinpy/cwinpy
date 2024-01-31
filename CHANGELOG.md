# Notable changes between versions

## [1.2.0] 2024-01-31

Changes for this release:

- Remove support for Python < 3.9 and require PESummary >= 1.1.1 (!170).
- Suppress SWIGLAL and htcondor warning when importing cwinpy in IPython (!171).
- Fix evaluation of `noise_log_likelihood` for the power law distribution in the hierarchical module (!169).

## [1.1.0] 2024-01-02

Changes for this release:

- Add script to generate results summary pages (!155).
- Switch to using SciTokens for authentication when accessing proprietary frames over CVMFS (!165).
- Allow `periodic_restart_time` to be passed in a DAG pipeline configuration file (!166).
- Remove use of `getenv=True` in HTCondor files (!160).
- Fix writing out heterodyned data to HDF5 files when standard deviations are supplied (!164).
- Sort frame cache files by GPS time (!162).
- Update deprecated htcondor functions (!161).
- No longer use temporary files when using the "Quick Setup" pipeline (!158).
- Make the [arby](https://arby.readthedocs.io/en/latest/) package a full requirement of CWInPy (!156).

## [1.0.2] 2023-06-17

Changes for this release:

- Allow merging of heterodyned files when some are empty (!153)

## [1.0.1] 2023-06-14

This release provides a minor bug fix (!151) and fixes a intermittent failure of the test suite (!152).

## [1.0.0] 2023-06-05

Changes for this release:

- Add the ability to compute the likelihood using reduced order quadrature (!119)
- Allow the heterodyned pipeline to ignore frame files that fail to be read (!142)
- Switch the default dynesty sampling method back to `"rwalk"` (!144)
- Use the `solar-system-ephemerides` package to provide ephemeris files (!140)
- Allow  the `HeterodynedData.power_spectrum` method to plot amplitude spectral density (and fix
  scalings) (!146, !147)
- Allow Bayesian blocks segmentation of a `HeterodynedData` object to be recalculated for data that
  is read in from an existing file (!141, !143)

## [0.10.0] 2023-05-11

Changes for this release:

- Unpinned matplotlib version, but require `pesummary` to be v1.0.0 or greater (!139)
- Switched from using `eval` to [`simpleeval`](https://github.com/danthedeckie/simpleeval) for safe configuration parsing (!138)
- Added the ability to add aliases to `PulsarParameters` values (!136)
- Update references to `gw-openscience.org` to be `gwosc.org` (!135)

## [0.9.3] 2023-03-17

Changes for this release:

- Update the bilby requirement to v2.0.1 or greater (!133) 

## [0.9.2] 2022-11-22

This release fixes a broken build in the previous release.

## [0.9.1] 2022-11-21

Changes for this release:

- Added checks for `Constraint` objects in priors and resolved duplicate construction of `PriorDict` (!130)
- Move project build metadata from `setup.cfg` to `pyproject.toml` (!131)

## [0.9.0] 2022-09-28

Major changes for this release:

- Allow the use of a "transient-continuous" signal model (!108)
- Fix how the minimum Bayesian Block chunk length is set in `HeterodynedData` (!114)
- Use SciPy filter functions for performing filtering during the heterodyne stage (!112)
- Move `likelihood.py` into the `pe` submodule (!116)
- Allow `pulsarfiles` configuration file option for `cwinpy_pe_pipeline` that is consistent with the `knope` and `heterodyne` pipeline (!117)
- Fix bug that now mean that PE will work for frequency/frequency derivative parameters (!120)
- Add the ability to use the Einstein Telescope (ET) detector ASD for simulated noise (!121)
- Exclude the log likelihood and log prior from the `Plot` posterior plots by default (!126)

## [0.8.0] 2022-05-27

The release has major changes, including some that are backwards incompatible. The major changes are:

- Allow pipelines to be given ISO format start and end dates as well as GPS times (!106)
- Greatly speed-up the running median calculation for `HeterodynedData` (!93)
- Greatly speed-up the Bayesian Blocks calculation for `HeterodynedData` (!94)
- Fix bugs that now allow non-GR parameters to be estimated (!99, !100)
- Correctly deal with input frame data that is 32-bit floats when heterodyning (!98)
- Add pipeline to perform sky-shifting analysis (!72)
- Set the default dynesty sampler method to be `rslice` (!91)
- Fixes to allow running pipelines on the OSG (!70, !101)
- Switch from using pycondor to HTCondor Python package (!89)
- Remove dependencies on bilby_pipe package (!90)
- (**Backwards incompatible**) Change `_dag` suffix on pipeline scripts to `_pipeline` (!71)

## [0.7.2] 2021-10-29

Changes for this release:

- Fix bug when combining multiple heterodyned time series during heterodyne pipeline (!69)

## [0.7.1] 2021-10-25

Changes for this release:

- Fix bug in resample rate usage when resuming a heterodyne analysis (!68)

## [0.7.0] 2021-10-22

Changes for this release:

- Add the `cwinpy_knope` and `cwinpy_knope_dag` scripts for running the full heterodyne and PE pipeline (and generate HTCondor DAGs) (!60)
- Move from using the `PulsarParametersPy` class from LALPulsar to a version within CWInPy itself (!66)
- Allow use of TEMPO2 (via libstempo) for calculating the phase (!63)
- Add `Plot` class to allow plotting of various posteriors (!62)
- Allow parameter estimation to be performed both coherently for multiple detectors and for the individual detectors when submitting a PE DAG job (!61)
- Use GWOSC data find server for finding local paths of CVMFS data rather than hardcoding these (!65)

Note: several of of these changes is are major updates and may break some backwards compatibility.
There may still be some minor bugs in the implementation and there are more comparison tests to
perform, but the avoid further unwieldy MRs these have been put into a release. This release also
requires a development version of LALSuite, so a Conda installable release of this version will not
be available.

## [0.6.0] 2021-06-22

Changes for this release:

- Add command line interface for data heterodyning, including generating HTCondor DAGs (!28)
- Add the ability to merge `HeterodynedData` HDF5 files (!51)
- Allow heterodynes to use pulsar ephemerides from the ATNF pulsar catalogue (!53)
- Switch the default save format of parameter estimation outputs to be HDF5 files rather than JSON (!54)
- Add heterodyned signal simulation class into CWInPy rather than using LALSuite version (!46, !55)
- Add a histogram distribution to the hierarchical analysis (!48)
- Fix how the KDE bandwidth is calculated in the hierarchical analysis (!49)
- Change the way packaging/versioning of the code is done (!52)

**Important note**: HDF5 files created from `HeterodynedData` objects using earlier versions of
CWInPy will no longer be compatible with v0.6.0 and will fail to be read.

## [0.5.0] 2021-03-15

Changes for this release:

- Added a logo (!29).
- Add Python implementation of data heterodyning (!25).
- Fixes and changes to the hierarchical analysis (!35, !40, !41, !43).
- Add example of hierarchical analysis to the documentation page (!42).

## [0.4.3] 2020-11-18

Changes for this release:

- Further updates to deal with changes to bilby_pipe pipeline API (!27)

## [0.4.2] 2020-11-18

Changes for this release:

- Updates to deal with changes to bilby_pipe's Condor API (!26).
- Allow simulations to set the start time, end time and time step of the data (!23).
- Allow hierarchical analysis to work in ellipticity as well as mass quadrupole (!24).

## [0.4.1] 2020-06-22

Changed for this release:

- Add an API for generating simulations of pulsar populations (!15).
- Changes the way that HTCondor jobs are set up and the directory structure used, allowing for file transfer and use over the OSG (!15).
- Add power law distribution to allowed hierarchical model distributions (!21).
- Change Gaussian distribution in the hierarchical models to take in a Dirichlet prior on the Gaussian mode weights (!22).

## [0.3.1] 2020-06-02

Changes for this release:

- Allow Earth and Sun ephemeris files, and time correction files, to be explicitly given to the `HeterodynedData` object. Additionally allow these files to be passed via the `cwinpy_pe` script and DAG generation (!20).

## [0.3.0] 2020-05-19

Changes for this release:

- Add the use of `pre-commits` for developers (!10).
- Rename the `cwinpy_knope` script to `cwinpy_pe` (!13). This is a **backwards incompatible** API change.
- Allow `HeterodynedData` objects to be written to and read from HDF5 files (!14).
- Fix a bug that cause irregularly sampled time stamps in a `HeterodynedData` object to be overwritten with regularly sampled values.
- Minor fixes to the hierarchical analysis (!11 and !12).
