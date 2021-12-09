#!/usr/bin/env python

"""
Run analysis to create two test pulsars and posterior samples to use
in the testing of the hierarchical code.
"""

import os
from collections import OrderedDict

import bilby
import numpy as np
from bilby.core.prior import PriorDict, Uniform
from cwinpy import HeterodynedData, TargetedPulsarLikelihood

# create a two fake pulsar parameter files
parcontent1 = """\
PSRJ     J0123+3456
RAJ      01:23:45.6789
DECJ     34:56:54.321
F0       567.89
F1       -1.2e-12
PEPOCH   56789
H0       0.0
COSIOTA  0.3
PSI      1.1
PHI0     2.4
DIST     2.0
"""

parcontent2 = """\
PSRJ     J2345-0123
RAJ      23:45:00.123
DECJ     -01:23:45.098
F0       212.09
F1       -6.7e-13
PEPOCH   56789
H0       0.0
COSIOTA  -0.4
PSI      0.1
PHI0     1.3
DIST     1.5
"""

label = "hierarchical_test_set"
outdir = "data"

if not os.path.isdir(outdir):
    os.makedirs(outdir)

detector = "H1"  # the detector to use
asd = [1e-24, 8e-25]  # noise amplitude spectral densities
times = np.linspace(1000000000.0, 1000086340.0, 1440)  # times
hets = []

# add content to the par file and create fake data
for i, content in enumerate([parcontent1, parcontent2]):
    parfile = os.path.join(outdir, "{}_{}.par".format(label, i))
    with open(parfile, "w") as fp:
        fp.write(content)

    # create some fake heterodyned data
    hets.append(
        HeterodynedData(times=times, par=parfile, fakeasd=asd[i], detector=detector)
    )

    # output the data
    hetfile = os.path.join(outdir, "{}_{}_data.txt".format(label, i))
    np.savetxt(
        hetfile,
        np.vstack((hets[i].times.value, hets[i].data.real, hets[i].data.imag)).T,
    )

# create priors
phi0range = [0.0, np.pi]
psirange = [0.0, np.pi / 2.0]
cosiotarange = [-1.0, 1.0]
q22range = [0.0, 1e38]
# h0range = [0., 1e-23]

# set prior for lalapps_pulsar_parameter_estimation_nested
priorfile = os.path.join(outdir, "{}_prior.txt".format(label))
priorcontent = """Q22 uniform {} {}
PHI0 uniform {} {}
PSI uniform {} {}
COSIOTA uniform {} {}
"""
with open(priorfile, "w") as fp:
    fp.write(priorcontent.format(*(q22range + phi0range + psirange + cosiotarange)))

# set prior for bilby
priors = OrderedDict()
priors["q22"] = Uniform(q22range[0], q22range[1], "q22", latex_label=r"$Q_{22}$")
priors["phi0"] = Uniform(
    phi0range[0], phi0range[1], "phi0", latex_label=r"$\phi_0$", unit="rad"
)
priors["psi"] = Uniform(
    psirange[0], psirange[1], "psi", latex_label=r"$\psi$", unit="rad"
)
priors["cosiota"] = Uniform(
    cosiotarange[0], cosiotarange[1], "cosiota", latex_label=r"$\cos{\iota}$"
)

Nlive = 1024  # number of nested sampling live points

# run bilby
for i, het in enumerate(hets):
    # set the likelihood for bilby
    likelihood = TargetedPulsarLikelihood(het, PriorDict(priors))

    thislabel = "{}_{}".format(label, i)

    # run bilby
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="cpnest",
        nlive=Nlive,
        outdir=outdir,
        label=thislabel,
        use_ratio=False,
    )
