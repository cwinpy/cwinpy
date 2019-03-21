#!/usr/bin/env python

"""
Compare cwinpy with lalapps_pulsar_parameter_estimation_nested for
data for multiple detectors (H1, L1 and V1) containing a software injection
with close-to-linear polarisation.
"""

import os
import subprocess as sp
import numpy as np
import corner
from collections import OrderedDict
from cwinpy import HeterodynedData, MultiHeterodynedData
from cwinpy import TargetedPulsarLikelihood
import bilby
from bilby.core.prior import Uniform, PriorDict
from matplotlib import pyplot as pl
from lalapps.pulsarpputils import pulsar_nest_to_posterior

# create a fake pulsar parameter file
parcontent = """\
PSRJ     J0123+3456
RAJ      01:23:45.6789
DECJ     34:56:54.321
F0       567.89
F1       -1.2e-12
PEPOCH   56789
H0       1.1e-25
COSIOTA  -0.05
PSI      0.5
PHI0     1.2
"""

injection_parameters = OrderedDict()
injection_parameters['h0'] = 1.1e-25
injection_parameters['phi0'] = 1.2
injection_parameters['psi'] = 0.5
injection_parameters['cosiota'] = -0.05

label = 'multi_detector_software_injection_linear'
outdir = 'outputs'

if not os.path.isdir(outdir):
    os.makedirs(outdir)

# add content to the par file
parfile = os.path.join(outdir, '{}.par'.format(label))
with open(parfile, 'w') as fp:
    fp.write(parcontent)

# create some fake heterodyned data
detectors = ['H1', 'L1', 'V1']  # the detector to use
asd = 1e-24  # noise amplitude spectral density
times = np.linspace(1000000000., 1000086340., 1440)  # times
het = {}
hetfiles = []
for detector in detectors:
    het[detector] = HeterodynedData(times=times, par=parfile, injpar=parfile,
                                    inject=True, fakeasd=asd, detector=detector)

    # output the data
    hetfile = os.path.join(outdir, '{}_{}_data.txt'.format(label, detector))
    np.savetxt(hetfile, np.vstack((het[detector].times,
                                   het[detector].data.real,
                                   het[detector].data.imag)).T)
    hetfiles.append(hetfile)

mhet = MultiHeterodynedData(het)

# create priors
phi0range = [0., np.pi]
psirange = [0., np.pi/2.]
cosiotarange = [-1., 1.]
h0range = [0., 1e-23]

# set prior for lalapps_pulsar_parameter_estimation_nested
priorfile = os.path.join(outdir, '{}_prior.txt'.format(label))
priorcontent = """H0 uniform {} {}
PHI0 uniform {} {}
PSI uniform {} {}
COSIOTA uniform {} {}
"""
with open(priorfile, 'w') as fp:
    fp.write(priorcontent.format(*(h0range + phi0range + psirange + cosiotarange)))

# set prior for bilby
priors = OrderedDict()
priors['h0'] = Uniform(h0range[0], h0range[1], 'h0', latex_label=r'$h_0$')
priors['phi0'] = Uniform(phi0range[0], phi0range[1], 'phi0', latex_label=r'$\phi_0$', unit='rad')
priors['psi'] = Uniform(psirange[0], psirange[1], 'psi', latex_label=r'$\psi$', unit='rad')
priors['cosiota'] = Uniform(cosiotarange[0], cosiotarange[1], 'cosiota', latex_label=r'$\cos{\iota}$')

# run lalapps_pulsar_parameter_estimation_nested
try:
    execpath = os.environ['CONDA_PREFIX']
except KeyError:
    raise KeyError("Please work in a conda environment with lalsuite and cwinpy installed")

execpath = os.path.join(execpath, 'bin')

lppen = os.path.join(execpath, 'lalapps_pulsar_parameter_estimation_nested')
n2p = os.path.join(execpath, 'lalinference_nest2pos') 

Nlive = 1024  # number of nested sampling live points
Nmcmcinitial = 0  # set to 0 so that prior samples are not resampled
tolerance = 0.1   # nested sampling stopping criterion (0.1 is default value)
priorsamples = 40000  # number of samples from the prior

outfile = os.path.join(outdir, '{}_nest.hdf'.format(label))

# set the command line arguments
runcmd = ' '.join([lppen,
                   '--verbose',
                   '--input-files', ','.join(hetfiles),
                   '--detectors', ','.join(detectors),
                   '--par-file', parfile,
                   '--prior-file', priorfile,
                   '--Nlive', '{}'.format(Nlive),
                   '--Nmcmcinitial', '{}'.format(Nmcmcinitial),
                   '--outfile', outfile,
                   '--ephem-earth', os.path.join(os.environ['CONDA_PREFIX'], 'share', 'lalpulsar', 'earth00-40-DE405.dat.gz'),
                   '--ephem-sun', os.path.join(os.environ['CONDA_PREFIX'], 'share', 'lalpulsar', 'sun00-40-DE405.dat.gz'), 
                   '--ephem-timecorr', os.path.join(os.environ['CONDA_PREFIX'], 'share', 'lalpulsar', 'te405_2000-2040.dat.gz')])

#print(runcmd)
with sp.Popen(runcmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, bufsize=1, universal_newlines=True) as p:
    for line in p.stderr:
        print(line, end='')

# convert nested samples to posterior samples
outpost = os.path.join(outdir, '{}_post.hdf'.format(label))
runcmd = ' '.join([n2p, '-p', outpost, outfile])
with sp.Popen(runcmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, bufsize=1, universal_newlines=True) as p:
    for line in p.stdout:
        print(line, end='')

# get posterior samples
post, evsig, evnoise = pulsar_nest_to_posterior(outpost)
lp = len(post['h0'].samples)
postsamples = np.zeros((lp, len(priors)))
for i, p in enumerate(priors.keys()):
    postsamples[:,i] = post[p].samples[:,0]

# set the likelihood for bilby
likelihood = TargetedPulsarLikelihood(mhet, PriorDict(priors))

# run bilby
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='cpnest', nlive=Nlive,
    outdir=outdir, label=label)

# compare evidences
comparefile = os.path.join(outdir, '{}_compare.txt'.format(label))
with open(comparefile, 'w') as fp:
    fp.write('Evidence (lalapps_pulsar_parameter_estimation_nested): {}\n'.format(evsig))
    fp.write('Evidence (cwinpy): {}\n'.format(result.log_evidence))
    fp.write('Noise evidence (lalapps_pulsar_parameter_estimation_nested): {}\n'.format(evnoise))
    fp.write('Noise evidence (cwinpy): {}\n'.format(result.log_noise_evidence))

# plot results
fig = result.plot_corner(save=False, parameters=injection_parameters)
fig = corner.corner(postsamples, fig=fig, color='r', bins=50, smooth=0.9,
                    quantiles=[0.16, 0.84],
                    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                    fill_contours=True, hist_kwargs={'density': True})

fig.savefig(os.path.join(outdir, '{}_corner.png'.format(label)), dpi=300)

