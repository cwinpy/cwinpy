#!/usr/bin/env python

import copy
import os
import shutil
import subprocess as sp

import numpy as np
from bilby.core.prior import PriorDict, Sine, Uniform

from cwinpy.heterodyne import heterodyne
from cwinpy.parfile import PulsarParameters
from cwinpy.pe import pe
from cwinpy.plot import Plot
from cwinpy.utils import LAL_EPHEMERIS_URL, download_ephemeris_file

# generate some fake data using lalpulsar_Makefakedata_v5
mfd = shutil.which("lalpulsar_Makefakedata_v5")

fakedatadir = "fake_data"
fakedatadetector = "H1"
fakedatachannel = f"{fakedatadetector}:FAKE_DATA"

fakedatastart = 1000000000
fakedataduration = 864000

os.makedirs(fakedatadir, exist_ok=True)

fakedatabandwidth = 8  # Hz
sqrtSn = 2e-23  # noise amplitude spectral density
fakedataname = "FAKEDATA"

# create one pulsar to inject
fakepulsarpar = []

# requirements for Makefakedata pulsar input files
pulsarstr = """\
Alpha = {alpha}
Delta = {delta}
Freq = {f0}
f1dot = {f1}
f2dot = {f2}
refTime = {pepoch}
h0 = {h0}
cosi = {cosi}
psi = {psi}
phi0 = {phi0}
"""

# pulsar parameters
f0 = 6.9456 / 2.0  # source rotation frequency (Hz)
f1 = -9.87654e-11 / 2.0  # source rotational frequency derivative (Hz/s)
f2 = 2.34134e-18 / 2.0  # second frequency derivative (Hz/s^2)
alpha = 0.0  # source right ascension (rads)
delta = 0.5  # source declination (rads)
pepoch = 1000000000  # frequency epoch (GPS)

# GW parameters
h0 = 5.5e-25  # GW amplitude
phi0 = 1.0  # GW initial phase (rads)
iota = 1.47
cosiota = np.cos(iota)  # cosine of inclination angle
psi = 0.5  # GW polarisation angle (rads)

mfddic = {
    "alpha": alpha,
    "delta": delta,
    "f0": 2 * f0,
    "f1": 2 * f1,
    "f2": 2 * f2,
    "pepoch": pepoch,
    "h0": h0,
    "cosi": cosiota,
    "psi": psi,
    "phi0": phi0,
}

fakepulsarpar = PulsarParameters()
fakepulsarpar["PSRJ"] = "J0000+0000"
fakepulsarpar["H0"] = h0
fakepulsarpar["PHI0"] = phi0 / 2.0
fakepulsarpar["PSI"] = psi
fakepulsarpar["IOTA"] = iota
fakepulsarpar["F"] = [f0, f1, f2]
fakepulsarpar["RAJ"] = alpha
fakepulsarpar["DECJ"] = delta
fakepulsarpar["PEPOCH"] = pepoch
fakepulsarpar["EPHEM"] = "DE405"
fakepulsarpar["UNITS"] = "TDB"

fakepardir = "par_dir"
os.makedirs(fakepardir, exist_ok=True)
fakeparfile = os.path.join(fakepardir, "J0000+0000.par")
fakepulsarpar.pp_to_par(fakeparfile)

injfile = os.path.join(fakepardir, "inj.dat")
with open(injfile, "w") as fp:
    fp.write("[Pulsar 1]\n")
    fp.write(pulsarstr.format(**mfddic))
    fp.write("\n")

# set ephemeris files
efile = download_ephemeris_file(LAL_EPHEMERIS_URL.format("earth00-40-DE405.dat.gz"))
sfile = download_ephemeris_file(LAL_EPHEMERIS_URL.format("sun00-40-DE405.dat.gz"))

cmds = [
    "-F",
    fakedatadir,
    f"--outFrChannels={fakedatachannel}",
    "-I",
    fakedatadetector,
    "--sqrtSX={0:.1e}".format(sqrtSn),
    "-G",
    str(fakedatastart),
    f"--duration={fakedataduration}",
    f"--Band={fakedatabandwidth}",
    "--fmin",
    "0",
    f'--injectionSources="{injfile}"',
    f"--outLabel={fakedataname}",
    f'--ephemEarth="{efile}"',
    f'--ephemSun="{sfile}"',
    "--randSeed=1234",  # for reproducibility
]

# run makefakedata
sp.run([mfd] + cmds)

# resolution
res = 1 / fakedataduration

# set priors for PE
priors = PriorDict()
priors["h0"] = Uniform(0, 1e-23, name="h0", latex_label="$h_0$")
priors["phi0"] = Uniform(0, np.pi, name="phi0", latex_label=r"$\phi_0$", unit="rad")
priors["psi"] = Uniform(0, np.pi / 2, name="psi", latex_label=r"$\psi$", unit="rad")
priors["iota"] = Sine(name="iota", latex_label=r"$\iota$")

priors["f0"] = Uniform(
    f0 - 5 * res, f0 + 5 * res, name="f0", latex_label=r"$f$", unit="Hz"
)
priors["f1"] = Uniform(
    f1 - 5 * res**2,
    f1 + 5 * res**2,
    name="f1",
    latex_label=r"$\dot{f}$",
    unit="Hz/s",
)

# perform heterodyne with f0 and f1 offset from true values
segments = [(fakedatastart, fakedatastart + fakedataduration)]

fulloutdir = os.path.join(fakedatadir, "heterodyne_output")

offsetpar = copy.deepcopy(fakepulsarpar)
offsetpar["F"] = [
    fakepulsarpar["F0"] - 2.3 * res,
    fakepulsarpar["F1"] + 1.8 * res**2,
]
offsetparfile = os.path.join(fakepardir, "J0000+0000_offset.par")
offsetpar.pp_to_par(offsetparfile)

inputkwargs = dict(
    starttime=segments[0][0],
    endtime=segments[-1][-1],
    pulsarfiles=offsetparfile,
    segmentlist=segments,
    framecache=fakedatadir,
    channel=fakedatachannel,
    freqfactor=2,
    stride=86400 // 2,
    resamplerate=1 / 60,
    includessb=True,
    includebsb=True,
    includeglitch=True,
    output=fulloutdir,
    label="heterodyne_{psr}_{det}_{freqfactor}.hdf5",
)

het = heterodyne(**inputkwargs)

peoutdir = os.path.join(fakedatadir, "pe_output")
pelabel = "nonroqoffset"

pekwargs = {
    "outdir": peoutdir,
    "label": pelabel,
    "prior": priors,
    "likelihood": "studentst",
    "detector": fakedatadetector,
    "data_file": list(het.outputfiles.values()),
    "par_file": copy.deepcopy(offsetpar),
}

# run PE without ROQ
pestandard = pe(**copy.deepcopy(pekwargs))

# set ROQ likelihood
ntraining = 2000
pekwargs["roq"] = True
pekwargs["roq_kwargs"] = {"ntraining": ntraining}
pekwargs["label"] = "roqoffset"

# run PE with ROQ
peroq = pe(**pekwargs)

# plot results
plot = Plot(
    {"Standard likelihood": pestandard.result, "ROQ likelihood": peroq.result},
    pulsar=fakeparfile,
)

plot.plot()
plot.save(os.path.join(peoutdir, "posteriors.png"), dpi=200)
