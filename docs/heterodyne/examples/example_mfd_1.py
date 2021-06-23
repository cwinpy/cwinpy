#!/usr/bin/env python

import shutil
import subprocess as sp

from astropy.utils.data import download_file
from cwinpy.utils import LAL_EPHEMERIS_URL
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy

# set data start, duration and bandwidth
fakedatastart = 1000000000
fakedataduration = 86400  # 1 day in seconds
fakedatabandwidth = 8  # 8 Hz

parfiles = ["J0123+0123.par", "J0404-0404.par"]

# create injection files for lalapps_Makefakedata_v5
# requirements for Makefakedata pulsar input files
isolatedstr = """\
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

binarystr = """\
orbitasini = {asini}
orbitPeriod = {period}
orbitTp = {Tp}
orbitArgp = {argp}
orbitEcc = {ecc}
"""

injfile = "inj.dat"
fp = open(injfile, "w")

for i, parfile in enumerate(parfiles):
    p = PulsarParametersPy(parfile)
    fp.write("[Pulsar {}]\n".format(i + 1))

    # set parameters (multiply freqs/phase by 2)
    mfddic = {
        "alpha": p["RAJ"],
        "delta": p["DECJ"],
        "f0": 2 * p["F0"],
        "f1": 2 * p["F1"],
        "f2": 2 * p["F2"],
        "pepoch": p["PEPOCH"],
        "h0": p["H0"],
        "cosi": p["COSIOTA"],
        "psi": p["PSI"],
        "phi0": 2 * p["PHI0"],
    }
    fp.write(isolatedstr.format(**mfddic))

    if p["BINARY"] is not None:
        mfdbindic = {
            "asini": p["A1"],
            "Tp": p["T0"],
            "period": p["PB"],
            "argp": p["OM"],
            "ecc": p["ECC"],
        }
        fp.write(binarystr.format(**mfdbindic))

    fp.write("\n")
fp.close()

# set ephemeris files
efile = download_file(LAL_EPHEMERIS_URL.format("earth00-40-DE405.dat.gz"), cache=True)
sfile = download_file(LAL_EPHEMERIS_URL.format("sun00-40-DE405.dat.gz"), cache=True)

# set detector
detector = "H1"
channel = "{}:FAKE_DATA".format(detector)

# set noise amplitude spectral density (use a small value to see the signal clearly)
sqrtSn = 1e-29

# set Makefakedata commands
cmds = [
    "-F",
    ".",
    "--outFrChannels={}".format(channel),
    "-I",
    detector,
    "--sqrtSX={0:.1e}".format(sqrtSn),
    "-G",
    str(fakedatastart),
    "--duration={}".format(fakedataduration),
    "--Band={}".format(fakedatabandwidth),
    "--fmin",
    "0",
    '--injectionSources="{}"'.format(injfile),
    "--outLabel=FAKEDATA",
    '--ephemEarth="{}"'.format(efile),
    '--ephemSun="{}"'.format(sfile),
]

# run makefakedata
sp.run([shutil.which("lalapps_Makefakedata_v5")] + cmds)
