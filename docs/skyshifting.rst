############
Sky-shifting
############

CWInPy can be used to produce posterior probability distributions for the unknown signal parameters
of a continuous gravitational-wave source, i.e., a pulsar. These allow you to constrain, for
example, the gravitational-wave amplitude, or orientation of the source. However, before making
inferences about a particular source's parameters it is useful to be sure that the signal you are
observing is actually astrophysical in origin. Gravitational-wave detector data contains many
non-astrophysical artifacts and also its properties vary over time, i.e., the noise is the data is
not drawn from a purely Gaussian `non-stationary process
<https://en.wikipedia.org/wiki/Stationary_process>`_ (see, e.g., [1]_). Some of these artifacts are narrow-band
spectral line features, which can wander across the frequency band of an expected signal and
potentially mimic a signal to some extent [2]_.

Sky-shifting process:

perform "coarse" heterodyne, i.e., heterodyne the data without accounting for any Doppler
corrections (just use terms of the Taylor expansion in the frequency evolution). Filter and
downsample the data, but making sure the filter is wide enough to accommodate the Doppler modulation
of the source.

randomly generate a number of new sky locations in the same ecliptic hemisphere as the source. For
each of these new locations, perform an additional heterodyne of the "coarse" data, using the
expected Doppler modulation for that position.

perform parameter estimation for each 

Example
=======

An easy way to test the sky-shifting analysis is by looking at one of the hardware injection
signals. The ``cwinpy_skyshift_pipeline`` script below set up the analysis to run on O1 data for the
``PULSAR03`` injection with 500 sky-shifts. By default this will run with both the LIGO detectors,
H1 and L1, with parameter estimation performed both coherently with both detectors and on each of
the individual detectors.

.. code-block: bash

   cwinpy_skyshift_pipeline --run O1 --pulsar PULSAR03 --nshifts 500 --accounting-group-tag aluk.dev.o1.cw.targeted.bayesian

Sky-shifting references
=======================

.. [1] `D. Davis *et al*
    <https://ui.adsabs.harvard.edu/abs/2021CQGra..38m5014D/abstract>`_, *CQG*, **38**, 135014 (2021).

.. [2] `P. Covas *et al*
    <https://ui.adsabs.harvard.edu/abs/2018PhRvD..97h2002C/abstract>`_, *PRD*, **97**, 082002 (2018).