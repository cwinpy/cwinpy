#!/usr/bin/env python

"""
An example estimating the posterior on h0 for some fake data in a targeted
pulsar-type search.
"""

import numpy as np
from bilby.core.prior import PriorDict, Uniform
from cwinpy import HeterodynedData, TargetedPulsarLikelihood
from matplotlib import pyplot as pl

# create a fake pulsar parameter file
parcontent = """\
PSRJ     J0123+3456
RAJ      01:23:45.6789
DECJ     34:56:54.321
F0       567.89
F1       -1.2e-12
PEPOCH   56789
H0       5.12e-25
COSIOTA  0.3
PSI      1.1
PHI0     2.4
"""

# add content to the par file
parfile = "J0123+3456.par"
with open(parfile, "w") as fp:
    fp.write(parcontent)

# create some fake heterodyned data
detector = "H1"  # the detector to use
times = np.linspace(1000000000.0, 1000086340.0, 1440)  # times
het = HeterodynedData(
    times=times,
    inject=True,
    par=parfile,
    injpar=parfile,
    fakeasd=1e-24,
    detector=detector,
)

# set prior on h0
priors = dict()
priors["h0"] = Uniform(0.0, 1e-24, "h0")

like = TargetedPulsarLikelihood(het, PriorDict(priors))

N = 500
h0s = np.linspace(0.0, 1e-24, N)
post = np.zeros(N)

for i, h0 in enumerate(h0s):
    params = {"h0": h0}
    like.parameters = params
    post[i] = like.log_likelihood() + priors["h0"].ln_prob(h0)

pl.plot(h0s, np.exp(post - np.max(post)), "b")
pl.axvline(het.injpar["H0"])
pl.show()
