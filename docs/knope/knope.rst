##############################
Known pulsar analysis pipeline
##############################

CWInPy can be used to run an end-to-end pipeline to search for gravitational waves from known
pulsars in data from ground-based gravitational-wave detectors. This "known pulsar pipeline" is
known here as "knope".

The pipeline has two main stages:

#. data preprocessing involving heterodyning, filtering and down-sampling raw gravitational-wave
   detector strain time-series;
#. using the processed data to estimate the posterior probability
   distributions of the unknown gravitational-wave signal parameter via Bayesian inference.

These stages can be run seperately via command line executables, or via their Python APIs, but can
be run together using the interface documented here. Comprehensive documentation for the
:ref:`heterodyne<Heterodyning data>` and :ref:`parameter estimation<Known pulsar parameter
estimation>` is provided elsewhere on this site and will be required for running the full pipeline.