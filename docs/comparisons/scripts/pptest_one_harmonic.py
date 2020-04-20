#!/usr/bin/env python

"""
Create HTCondor Dag to run a P-P test for the standard four parameters
of a pulsar gravitational-wave signal, assuming emission at twice the rotation
frequency: h0, iota, psi and phi0.
"""

import bilby
import numpy as np
from cwinpy.pe.testing import PEPPPlotsDAG

# set the priors
prior = {}
prior["h0"] = bilby.core.prior.Uniform(minimum=0.0, maximum=1e-22, latex_label="$h_0$")
prior["phi0"] = bilby.core.prior.Uniform(
    name="phi0", minimum=0.0, maximum=np.pi, latex_label=r"$\phi_0$", unit="rad"
)
prior["iota"] = bilby.core.prior.Sine(
    name="iota", minimum=0.0, maximum=np.pi, latex_label=r"$\iota$", unit="rad"
)
prior["psi"] = bilby.core.prior.Uniform(
    name="psi", minimum=0.0, maximum=np.pi / 2, latex_label=r"$\psi$", unit="rad"
)

# Maximum amplitude for any of the injection signal (below the prior maximum)
maxamp = 2.5e-24
det = "H1"  # detector and noise ASD
ninj = 250  # number of simulated signals
basedir = "/home/sismp2/pptest/fourparameters"  # base directory
accuser = "matthew.pitkin"
accgroup = "aluk.dev.o3.cw.targeted"
sampler = "dynesty"
numba = True
getenv = True
freqrange = (100.0, 200.0)

run = PEPPPlotsDAG(
    prior,
    ninj=ninj,
    maxamp=maxamp,
    detector=det,
    accountuser=accuser,
    accountgroup=accgroup,
    sampler=sampler,
    numba=numba,
    getenv=getenv,
    freqrange=freqrange,
    basedir=basedir,
    submit=True,  # submit the DAG
)
