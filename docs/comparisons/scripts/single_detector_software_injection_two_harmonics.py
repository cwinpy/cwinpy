#!/usr/bin/env python

"""
Compare cwinpy with lalapps_pulsar_parameter_estimation_nested for data from a
single detector containing a software injection with components at two
harmonics.
"""

import os
import subprocess as sp
from collections import OrderedDict

import corner
import h5py
import matplotlib.font_manager as font_manager
import numpy as np
from astropy.utils.data import download_file
from bilby.core.prior import Uniform
from comparitors import comparisons_two_harmonics
from cwinpy import HeterodynedData
from cwinpy.pe import pe
from lalinference import LALInferenceHDF5PosteriorSamplesDatasetName
from lalinference.io import read_samples
from matplotlib.lines import Line2D

# URL for ephemeris files
DOWNLOAD_URL = "https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/{}"

# create a fake pulsar parameter file
parcontent = """\
PSRJ     J0123+3456
RAJ      01:23:45.6789
DECJ     34:56:54.321
F0       567.89
F1       -1.2e-12
PEPOCH   56789
C21      1.1e-25
C22      1.6e-25
COSIOTA  -0.3
PSI      0.9
PHI21    1.8
PHI22    3.6
"""

injection_parameters = OrderedDict()
injection_parameters["c21"] = 1.1e-25
injection_parameters["c22"] = 1.6e-25
injection_parameters["phi21"] = 1.8
injection_parameters["phi21"] = 3.6
injection_parameters["psi"] = 0.9
injection_parameters["cosiota"] = -0.3

label = "single_detector_software_injection_two_harmonics"
outdir = "outputs"

if not os.path.isdir(outdir):
    os.makedirs(outdir)

# add content to the par file
parfile = os.path.join(outdir, "{}.par".format(label))
with open(parfile, "w") as fp:
    fp.write(parcontent)

# create some fake heterodyned data
detector = "H1"  # the detector to use
asds = [1e-24, 2e-24]  # noise amplitude spectral densities
times = np.linspace(1000000000.0, 1000086340.0, 1440)  # times
harmonics = [1, 2]

seed = np.random.RandomState(896231)

hetfiles = []
for harmonic, asd in zip(harmonics, asds):
    het = HeterodynedData(
        times=times,
        par=parfile,
        injpar=parfile,
        inject=True,
        fakeasd=asd,
        detector=detector,
        freqfactor=harmonic,
        fakeseed=seed,
    )

    # output the data
    hetfile = os.path.join(
        outdir, "{}_{}_{}_data.txt".format(label, detector, harmonic)
    )
    het.write(hetfile)
    hetfiles.append(hetfile)

# create priors
phi21range = [0.0, 2.0 * np.pi]
phi22range = [0.0, 2.0 * np.pi]
psirange = [0.0, np.pi / 2.0]
cosiotarange = [-1.0, 1.0]
c21range = [0.0, 1e-23]
c22range = [0.0, 1e-23]

# set prior for lalapps_pulsar_parameter_estimation_nested
priorfile = os.path.join(outdir, "{}_prior.txt".format(label))
priorcontent = """C21 uniform {} {}
C22 uniform {} {}
PHI21 uniform {} {}
PHI22 uniform {} {}
PSI uniform {} {}
COSIOTA uniform {} {}
"""
with open(priorfile, "w") as fp:
    fp.write(
        priorcontent.format(
            *(c21range + c22range + phi21range + phi22range + psirange + cosiotarange)
        )
    )

# set prior for bilby
priors = OrderedDict()
priors["c21"] = Uniform(c21range[0], c21range[1], "c21", latex_label=r"$C_{21}$")
priors["c22"] = Uniform(c22range[0], c22range[1], "c22", latex_label=r"$C_{22}$")
priors["phi21"] = Uniform(
    phi21range[0], phi21range[1], "phi21", latex_label=r"$\Phi_{21}$", unit="rad"
)
priors["phi22"] = Uniform(
    phi22range[0], phi22range[1], "phi22", latex_label=r"$\Phi_{22}$", unit="rad"
)
priors["psi"] = Uniform(
    psirange[0], psirange[1], "psi", latex_label=r"$\psi$", unit="rad"
)
priors["cosiota"] = Uniform(
    cosiotarange[0], cosiotarange[1], "cosiota", latex_label=r"$\cos{\iota}$"
)

# run lalapps_pulsar_parameter_estimation_nested
try:
    execpath = os.environ["CONDA_PREFIX"]
except KeyError:
    raise KeyError(
        "Please work in a conda environment with lalsuite and cwinpy installed"
    )

execpath = os.path.join(execpath, "bin")

lppen = os.path.join(execpath, "lalapps_pulsar_parameter_estimation_nested")
n2p = os.path.join(execpath, "lalinference_nest2pos")

Nlive = 1024  # number of nested sampling live points
Nmcmcinitial = 0  # set to 0 so that prior samples are not resampled

outfile = os.path.join(outdir, "{}_nest.hdf".format(label))

# set ephemeris files
efile = download_file(DOWNLOAD_URL.format("earth00-40-DE405.dat.gz"), cache=True)
sfile = download_file(DOWNLOAD_URL.format("sun00-40-DE405.dat.gz"), cache=True)
tfile = download_file(DOWNLOAD_URL.format("te405_2000-2040.dat.gz"), cache=True)

# set the command line arguments
runcmd = " ".join(
    [
        lppen,
        "--verbose",
        "--input-files",
        ",".join(hetfiles),
        "--detectors",
        detector,
        "--par-file",
        parfile,
        "--prior-file",
        priorfile,
        "--Nlive",
        "{}".format(Nlive),
        "--harmonics",
        ",".join([str(harmonic) for harmonic in harmonics]),
        "--Nmcmcinitial",
        "{}".format(Nmcmcinitial),
        "--outfile",
        outfile,
        "--ephem-earth",
        efile,
        "--ephem-sun",
        sfile,
        "--ephem-timecorr",
        tfile,
    ]
)

p = sp.Popen(runcmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
out, err = p.communicate()

# convert nested samples to posterior samples
outpost = os.path.join(outdir, "{}_post.hdf".format(label))
runcmd = " ".join([n2p, "-p", outpost, outfile])
with sp.Popen(
    runcmd,
    stdout=sp.PIPE,
    stderr=sp.PIPE,
    shell=True,
    bufsize=1,
    universal_newlines=True,
) as p:
    for line in p.stdout:
        print(line, end="")

# get posterior samples
post = read_samples(outpost, tablename=LALInferenceHDF5PosteriorSamplesDatasetName)
lp = len(post["C21"])
postsamples = np.zeros((lp, len(priors)))
for i, p in enumerate(priors.keys()):
    postsamples[:, i] = post[p.upper()]

# get evidence
hdf = h5py.File(outpost, "r")
a = hdf["lalinference"]["lalinference_nest"]
evsig = a.attrs["log_evidence"]
evnoise = a.attrs["log_noise_evidence"]
hdf.close()

# run bilby via the pe interface
runner = pe(
    data_file_1f=hetfiles[0],
    data_file_2f=hetfiles[1],
    par_file=parfile,
    prior=priors,
    detector=detector,
    sampler="dynesty",
    sampler_kwargs={"Nlive": Nlive, "walks": 60},
    outdir=outdir,
    label=label,
)

result = runner.result

# output comparisons
comparisons_two_harmonics(label, outdir, priors, cred=0.9)

# plot results
fig = result.plot_corner(save=False, parameters=list(priors.keys()), color="b")
fig = corner.corner(
    postsamples,
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

# custom legend
legend_elements = [
    Line2D([], [], color="r", label="lalapps_pulsar_parameter_estimation_nested"),
    Line2D([], [], color="b", label="cwinpy_pe"),
]
font = font_manager.FontProperties(family="monospace")
leg = axes[3].legend(
    handles=legend_elements, loc="upper right", frameon=False, prop=font, handlelength=3
)
for line in leg.get_lines():
    line.set_linewidth(1.0)

fig.savefig(os.path.join(outdir, "{}_corner.png".format(label)), dpi=150)
