#!/usr/bin/env python

"""
Compare cwinpy with lalapps_pulsar_parameter_estimation_nested for noise-only
data for a single detector.
"""

import os
import subprocess as sp
import numpy as np
from scipy.stats import ks_2samp
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
import h5py


# URL for ephemeris files
DOWNLOAD_URL = 'https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/src/{}'

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

label = 'single_detector_noise_only'
outdir = 'outputs'

if not os.path.isdir(outdir):
    os.makedirs(outdir)

# add content to the par file
parfile = os.path.join(outdir, '{}.par'.format(label))
with open(parfile, 'w') as fp:
    fp.write(parcontent)

# create some fake heterodyned data
detector = 'H1'  # the detector to use
asd = 1e-24  # noise amplitude spectral density
times = np.linspace(1000000000., 1000086340., 1440)  # times
het = HeterodynedData(times=times, par=parfile, fakeasd=asd, detector=detector)

# output the data
hetfile = os.path.join(outdir, '{}_data.txt'.format(label))
np.savetxt(hetfile, np.vstack((het.times, het.data.real, het.data.imag)).T)

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

# set ephemeris files
efile = download_file(DOWNLOAD_URL.format('earth00-40-DE405.dat.gz'), cache=True)
sfile = download_file(DOWNLOAD_URL.format('sun00-40-DE405.dat.gz'), cache=True)
tfile = download_file(DOWNLOAD_URL.format('te405_2000-2040.dat.gz'), cache=True)

# set the command line arguments
runcmd = ' '.join([lppen,
                   '--verbose',
                   '--input-files', hetfile,
                   '--detectors', detector,
                   '--par-file', parfile,
                   '--prior-file', priorfile,
                   '--Nlive', '{}'.format(Nlive),
                   '--Nmcmcinitial', '{}'.format(Nmcmcinitial),
                   '--outfile', outfile,
                   '--ephem-earth', efile,
                   '--ephem-sun', sfile, 
                   '--ephem-timecorr', tfile])

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

# get "information gain"
info = h5py.File(outpost)['lalinference']['lalinference_nest'].attrs['information_nats']
everr = np.sqrt(info/Nlive)  # the uncertainty on the evidence

# set the likelihood for bilby
likelihood = TargetedPulsarLikelihood(het, PriorDict(priors))

# run bilby
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='cpnest', nlive=Nlive,
    outdir=outdir, label=label)
#result = bilby.core.result.read_in_result(label=label, outdir=outdir)

# evaluate the likelihood on a grid
gridpoints = 35
grid_size = dict()
evdiff = 0.
for p in priors.keys():
    grid_size[p] = np.linspace(np.min(result.posterior[p]), np.max(result.posterior[p]), gridpoints)

grid = Grid(likelihood, PriorDict(priors), grid_size=grid_size)
grid_evidence = grid.log_evidence

# compare evidences
comparefile = os.path.join(outdir, '{}_compare.txt'.format(label))
filetxt = """\
.. csv-table:: Evidence table
   :widths: auto
   :header: "Method", ":math:`\\\\ln{{(Z)}}`", ":math:`\\\\ln{{(Z)}}` noise", ":math:`\\\\ln{{}}` Odds"

   "``lalapps_pulsar_parameter_estimation_nested``", "{0:.3f}", "{1:.3f}", "{2:.3f}±{3:.3f}"
   "cwinpy", "{4:.3f}", "{5:.3f}", "{6:.3f}±{7:.3f}"
   "cwinpy (grid)", "{8:.3f}", "", "{9:.3f}"

.. csv-table:: Parameter table
   :widths: auto
   :header: "Method", ":math:`h_0`", ":math:`\\\\phi_0` (rad)", ":math:`\\\\psi` (rad)", ":math:`\\\\cos{{\\\\iota}}`"

   "``lalapps_pulsar_parameter_estimation_nested``", "{10:.2f}±{11:.2f}×10\ :sup:`{12:d}`", "{13:.2f}±{14:.2f}", "{15:.2f}±{16:.2f}", "{17:.2f}±{18:.2f}"
   "cwinpy", "{19:.2f}±{20:.2f}×10\ :sup:`{21:d}`", "{22:.2f}±{23:.2f}", "{24:.2f}±{25:.2f}", "{26:.2f}±{27:.2f}"

.. csv-table:: Maximum a-posteriori
   :widths: auto
   :header: "Method", ":math:`h_0`", ":math:`\\\\phi_0` (rad)", ":math:`\\\\psi` (rad)", ":math:`\\\\cos{{\\\\iota}}`", ":math:`\\\\ln{{(L)}}` max"

   "``lalapps_pulsar_parameter_estimation_nested``", "{28:.2f}×10\ :sup:`{29:d}`", "{30:.2f}", "{31:.2f}", "{32:.2f}", "{33:.2f}"
   "cwinpy", "{34:.2f}×10\ :sup:`{35:d}`", "{36:.2f}", "{37:.2f}", "{38:.2f}", "{39:.2f}"

Minimum K-S test p-value: {40:.4f}
"""

values = 41*[None]
values[0:4] = evsig, evnoise, (evsig - evnoise), everr
values[4:8] = result.log_evidence, result.log_noise_evidence, (result.log_evidence - result.log_noise_evidence), result.log_evidence_err
values[8:10] = grid_evidence, (grid_evidence - result.log_noise_evidence)

# output parameter means and standard deviations
idx = 10
for method in ['lalapps', 'cwinpy']:
    for p in priors.keys():
        mean = post[p].samples[:,0].mean() if method == 'lalapps' else result.posterior[p].mean()
        std = post[p].samples[:,0].std() if method == 'lalapps' else result.posterior[p].std()
        if p == 'h0':
            exponent = int(np.floor(np.log10(mean)))
            mean = mean / 10**exponent
            std = std / 10**exponent
            values[idx] = mean
            values[idx + 1] = std
            values[idx + 2] = exponent
            idx += 3
        else:
            values[idx] = mean
            values[idx + 1] = std
            idx += 2

# output parameter maximum a-posteriori points
maxidx = (result.posterior['log_likelihood'] + result.posterior['log_prior']).idxmax() 
for method in ['lalapps', 'cwinpy']:
    for p in priors.keys():
        maxpval = post.maxP[1][p] if method == 'lalapps' else result.posterior[p][maxidx]
        if p == 'h0':
            exponent = int(np.floor(np.log10(maxpval)))
            values[idx] = maxpval / 10**exponent
            values[idx + 1] = exponent
            idx += 2
        else:
            values[idx] = maxpval
            idx += 1
    values[idx] = post.maxP[1]['logl'] if method == 'lalapps' else result.posterior['log_likelihood'][maxidx]
    idx += 1

# calculate the Kolmogorov-Smirnov test for each 1d marginalised distribution
# from the two codes, and output the minimum p-value of the KS test statistic
# over all parameters
values[idx] = np.inf
for p in priors.keys():
    _, pvalue = ks_2samp(post[p].samples[:,0], result.posterior[p])
    if pvalue < values[idx]:
        values[idx] = pvalue

with open(comparefile, 'w') as fp:
    fp.write(filetxt.format(*values))

# plot results
fig = result.plot_corner(save=False, parameters=list(priors.keys()))
fig = corner.corner(postsamples, fig=fig, color='r', bins=50, smooth=0.9,
                    quantiles=[0.16, 0.84],
                    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                    fill_contours=True, hist_kwargs={'density': True})
axes = fig.get_axes()
axidx = 0
for p in priors.keys():
    axes[axidx].plot(grid.sample_points[p], np.exp(grid.marginalize_ln_posterior(
    not_parameter=p) - grid_evidence), 'k--')
    axidx += 5

fig.savefig(os.path.join(outdir, '{}_corner.png'.format(label)), dpi=300)
