#!/usr/bin/env python

"""
An example estimating the posterior on h0 for some fake data in a targeted
pulsar-type search using the pe interface.
"""

import numpy as np
from bilby.core.prior import Uniform
from cwinpy.pe import pe
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

# use pe to create the data and sample on a grid
detector = "H1"  # the detector to use
times = np.linspace(1000000000.0, 1000086340.0, 1440)  # times
asd = 1e-24

# set prior on h0
priors = dict()
priors["h0"] = Uniform(0.0, 1e-24, "h0")
h0s = np.linspace(0.0, 1e-24, 500)  # h0 values to evaluate at

run = pe(
    detector=detector,
    fake_times=times,
    par_file=parfile,
    inj_par=parfile,
    fake_asd=asd,
    grid=True,
    grid_kwargs={"grid_size": {"h0": h0s}},
    prior=priors,
)

pl.plot(h0s, np.exp(run.grid.ln_posterior - np.max(run.grid.ln_posterior)), "b")
pl.axvline(run.hetdata["H1"][0].par["H0"])
pl.show()
