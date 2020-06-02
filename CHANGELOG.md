# Notable changes between versions

## [0.3.1] 2019-06-02

Changes for this release:

- Allow Earth and Sun ephemeris files, and time correction files, to be explicitly given to the `HeterodynedData` object. Additionally allow these files to be passed via the `cwinpy_pe` script and DAG generation (!20).

## [0.3.0] 2019-05-19

Changes for this release:

- Add the use of `pre-commits` for developers (!10).
- Rename the `cwinpy_knope` script to `cwinpy_pe` (!13). This is a **backwards incompatible** API change.
- Allow `HeterodynedData` objects to be written to and read from HDF5 files (!14).
- Fix a bug that cause irregularly sampled time stamps in a `HeterodynedData` object to be overwritten with regularly sampled values.
- Minor fixes to the hierarchical analysis (!11 and !12)
