#!/usr/bin/env python

"""
Compare cwinpy with lalapps_pulsar_parameter_estimation_nested for noise-only
data for a single detector using reflective and circular boundaries, and
comparing cpnest with dynesty
"""

import os
import subprocess as sp
import numpy as np
import corner
from collections import OrderedDict
from cwinpy import HeterodynedData
from cwinpy import TargetedPulsarLikelihood
import bilby
from bilby.core.prior import Uniform, PriorDict
from bilby.core.grid import Grid
from matplotlib import pyplot as pl
from lalapps.pulsarpputils import pulsar_nest_to_posterior
from astropy.utils.data import download_file
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager

# comparison function
from comparitors import comparisons


# URL for ephemeris files
DOWNLOAD_URL = "https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/src/{}"

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

label = "single_detector_noise_only_boundaries"
outdir = "outputs"

if not os.path.isdir(outdir):
    os.makedirs(outdir)

# add content to the par file
parfile = os.path.join(outdir, "{}.par".format(label))
with open(parfile, "w") as fp:
    fp.write(parcontent)

# create some fake heterodyned data
detector = "H1"  # the detector to use
asd = 1e-24  # noise amplitude spectral density
times = np.linspace(1000000000.0, 1000086340.0, 1440)  # times
het = HeterodynedData(times=times, par=parfile, fakeasd=asd, detector=detector)

# output the data
hetfile = os.path.join(outdir, "{}_data.txt".format(label))
np.savetxt(hetfile, np.vstack((het.times, het.data.real, het.data.imag)).T)

# create priors
phi0range = [0.0, 2.0 * np.pi]
psirange = [0.0, 2.0 * np.pi]
cosiotarange = [-1.0, 1.0]
h0range = [0.0, 1e-23]

# set prior for bilby
priors = OrderedDict()
priors["h0"] = Uniform(
    h0range[0], h0range[1], "h0", latex_label=r"$h_0$", boundary="reflective"
)
priors["phi0"] = Uniform(
    phi0range[0],
    phi0range[1],
    "phi0",
    latex_label=r"$\phi_0$",
    unit="rad",
    boundary="periodic",
)
priors["psi"] = Uniform(
    psirange[0],
    psirange[1],
    "psi",
    latex_label=r"$\psi$",
    unit="rad",
    boundary="periodic",
)
priors["cosiota"] = Uniform(
    cosiotarange[0],
    cosiotarange[1],
    "cosiota",
    latex_label=r"$\cos{\iota}$",
    boundary="reflective",
)


Nlive = 1024  # number of nested sampling live points

# set ephemeris files
efile = download_file(DOWNLOAD_URL.format("earth00-40-DE405.dat.gz"), cache=True)
sfile = download_file(DOWNLOAD_URL.format("sun00-40-DE405.dat.gz"), cache=True)
tfile = download_file(DOWNLOAD_URL.format("te405_2000-2040.dat.gz"), cache=True)

# set the likelihood for bilby
likelihood = TargetedPulsarLikelihood(het, PriorDict(priors))

# run bilby with cpnest
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="cpnest",
    nlive=Nlive,
    outdir=outdir,
    label=label,
    use_ratio=False,
)

# run bilby with dynesty
resultdynesty = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=Nlive,
    outdir=outdir,
    label=label,
    use_ratio=False,
)

# evaluate the likelihood on a grid
gridpoints = 35
grid_size = dict()
for p in priors.keys():
    grid_size[p] = np.linspace(
        np.min(result.posterior[p]), np.max(result.posterior[p]), gridpoints
    )

# grid = Grid(likelihood, PriorDict(priors), grid_size=grid_size)

# output comparisons
# comparisons(label, outdir, grid, priors, cred=0.9)

# plot results
fig = result.plot_corner(save=False, parameters=list(priors.keys()), color="b")
fig = corner.corner(
    resultdynesty.samples,
    fig=fig,
    color="r",
    bins=50,
    smooth=0.9,
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    fill_contours=True,
    hist_kwargs={"density": True},
)
axes = fig.get_axes()
# axidx = 0
# for p in priors.keys():
#    axes[axidx].plot(grid.sample_points[p], np.exp(grid.marginalize_ln_posterior(
#    not_parameter=p) - grid.log_evidence), 'k--')
#    axidx += 5

# custom legend
legend_elements = [
    Line2D([], [], color="r", label="dynesty"),
    Line2D([], [], color="b", label="cpnest"),
]  # ,
#                   Line2D([], [], color='k', ls='--', label='cwinpy (grid)')]
font = font_manager.FontProperties(family="monospace")
leg = axes[3].legend(
    handles=legend_elements, loc="upper right", frameon=False, prop=font, handlelength=3
)
for line in leg.get_lines():
    line.set_linewidth(1.0)

fig.savefig(os.path.join(outdir, "{}_corner.png".format(label)), dpi=200)
