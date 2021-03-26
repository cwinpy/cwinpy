# Notable changes between versions

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
