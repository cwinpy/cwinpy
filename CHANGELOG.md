# Notable changes between versions

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
