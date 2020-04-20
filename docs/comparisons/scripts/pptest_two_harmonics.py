#!/usr/bin/env python

"""
Create HTCondor Dag to run a P-P test for the six parameters of a pulsar
gravitational-wave signal, assuming emission at both once and twice the
rotation frequency: c21, c22, iota, psi, phi21, and phi22.
"""

import bilby
import numpy as np
from cwinpy.pe.testing import PEPPPlotsDAG

# set the priors
prior = {}
prior["c21"] = bilby.core.prior.Uniform(
    minimum=0.0, maximum=1e-22, latex_label="$C_{21}$"
)
prior["c22"] = bilby.core.prior.Uniform(
    minimum=0.0, maximum=1e-22, latex_label="$C_{22}$"
)
prior["phi21"] = bilby.core.prior.Uniform(
    name="phi21",
    minimum=0.0,
    maximum=2.0 * np.pi,
    latex_label=r"$\Phi_{21}$",
    unit="rad",
)
prior["phi22"] = bilby.core.prior.Uniform(
    name="phi22",
    minimum=0.0,
    maximum=2.0 * np.pi,
    latex_label=r"$\Phi_{22}$",
    unit="rad",
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
basedir = "/home/sismp2/pptest/sixparameters"  # base directory
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
