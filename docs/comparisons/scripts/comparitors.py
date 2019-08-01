"""
Output various comparitors between lalapps_pulsar_parameter_estimation_nested
runs and cwinpy runs
"""

import os
import h5py
import numpy as np
from scipy.stats import ks_2samp, combine_pvalues
from scipy.spatial.distance import jensenshannon
from lalapps.pulsarpputils import pulsar_nest_to_posterior
from bilby.core.result import read_in_result


# comparison rst table information
FILETEXT = """\
.. csv-table:: Evidence table
   :widths: auto
   :header: "Method", ":math:`\\\\ln{{(Z)}}`", ":math:`\\\\ln{{(Z)}}` noise", ":math:`\\\\ln{{}}` Odds"

   "``lalapps_pulsar_parameter_estimation_nested``", "{0:.3f}", "{1:.3f}", "{2:.3f}±{3:.3f}"
   "``cwinpy``", "{4:.3f}", "{5:.3f}", "{6:.3f}±{7:.3f}"
   "``cwinpy`` (grid)", "{8:.3f}", "", "{9:.3f}"

.. csv-table:: Parameter table
   :widths: auto
   :header: "Method", ":math:`h_0`", ":math:`\\\\phi_0` (rad)", ":math:`\\\\psi` (rad)", ":math:`\\\\cos{{\\\\iota}}`"

   "``lalapps_pulsar_parameter_estimation_nested``", "{10:.2f}±{11:.2f}×10\ :sup:`{12:d}`", "{13:.2f}±{14:.2f}", "{15:.2f}±{16:.2f}", "{17:.2f}±{18:.2f}"
   "{19:d}% credible intervals", "[{20:.2f}, {21:.2f}]×10\ :sup:`{22:d}`", "[{23:.2f}, {24:.2f}]", "[{25:.2f}, {26:.2f}]", "[{27:.2f}, {28:.2f}]"
   "``cwinpy``", "{29:.2f}±{30:.2f}×10\ :sup:`{31:d}`", "{32:.2f}±{33:.2f}", "{34:.2f}±{35:.2f}", "{36:.2f}±{37:.2f}"
   "{38:d}% credible intervals", "[{39:.2f}, {40:.2f}]×10\ :sup:`{41:d}`", "[{42:.2f}, {43:.2f}]", "[{44:.2f}, {45:.2f}]", "[{46:.2f}, {47:.2f}]"

.. csv-table:: Maximum a-posteriori
   :widths: auto
   :header: "Method", ":math:`h_0`", ":math:`\\\\phi_0` (rad)", ":math:`\\\\psi` (rad)", ":math:`\\\\cos{{\\\\iota}}`", ":math:`\\\\ln{{(L)}}` max"

   "``lalapps_pulsar_parameter_estimation_nested``", "{48:.2f}×10\ :sup:`{49:d}`", "{50:.2f}", "{51:.2f}", "{52:.2f}", "{53:.2f}"
   "``cwinpy``", "{54:.2f}×10\ :sup:`{55:d}`", "{56:.2f}", "{57:.2f}", "{58:.2f}", "{59:.2f}"

| Combined K-S test p-value: {60:.4f}
| Maximum Jensen-Shannon divergence: {61:.4f}
"""


def credible_interval(samples, ci=0.90):
    # get the given percentage (minimum) credible interval
    sortedsamples = sorted(samples)

    lowbound = sortedsamples[0]
    highbound = sortedsamples[-1]
    cregion = highbound - lowbound
    lsam = len(sortedsamples)
    cllen = int(lsam*ci)
    for j in range(lsam-cllen):
        if sortedsamples[j + cllen] - sortedsamples[j] < cregion:
            lowbound = sortedsamples[j]
            highbound = sortedsamples[j + cllen]
            cregion = highbound - lowbound

    return lowbound, highbound


def comparisons(label, outdir, grid, priors, cred=0.9):
    """
    Perform comparisons of the evidence, parameter values, confidence
    intervals, and Kolmogorov-Smirnov test between samples produced with
    lalapps_pulsar_parameter_estimation_nested and cwinpy.
    """

    lppenfile = os.path.join(outdir, '{}_post.hdf'.format(label))

    # read in posterior samples from lalapps_pulsar_parameter_estimation_nested
    post, evsig, evnoise = pulsar_nest_to_posterior(lppenfile)

    # get uncertainty on ln(evidence)
    info = h5py.File(lppenfile)['lalinference']['lalinference_nest'].attrs['information_nats']
    nlive = h5py.File(lppenfile)['lalinference']['lalinference_nest'].attrs['number_live_points']
    everr = np.sqrt(info/nlive)  # the uncertainty on the evidence

    # read in cwinpy results
    result = read_in_result(outdir=outdir, label=label)

    # comparison file
    comparefile = os.path.join(outdir, '{}_compare.txt'.format(label))

    # get grid-based evidence
    grid_evidence = grid.log_evidence

    # set values to output
    values = 62*[None]
    values[0:4] = evsig, evnoise, (evsig - evnoise), everr
    values[4:8] = result.log_evidence, result.log_noise_evidence, result.log_bayes_factor, result.log_evidence_err
    values[8:10] = grid_evidence, (grid_evidence - result.log_noise_evidence)

    # output parameter means standard deviations, and credible intervals
    idx = 10
    for method in ['lalapps', 'cwinpy']:
        values[idx + 9] = int(cred * 100)
        for p in priors.keys():
            samples = post[p].samples[:,0] if method == 'lalapps' else result.posterior[p]
            mean = samples.mean()
            std = samples.std()
            low, high = credible_interval(samples, ci=cred)
            if p == 'h0':
                exponent = int(np.floor(np.log10(mean)))
                values[idx] = mean / 10**exponent
                values[idx + 1] = std / 10**exponent
                values[idx + 2] = exponent
                values[idx + 10] = low / 10**exponent
                values[idx + 11] = high / 10**exponent
                values[idx + 12] = exponent
                idx += 3
            else:
                values[idx] = mean
                values[idx + 1] = std
                values[idx + 10] = low
                values[idx + 11] = high
                idx += 2
        idx += 10

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
        values[idx] = post.maxP[0][0] if method == 'lalapps' else result.posterior['log_likelihood'][maxidx]
        idx += 1

    # calculate the Kolmogorov-Smirnov test for each 1d marginalised distribution,
    # and the Jensen-Shannon divergence, from the two codes. Output the 
    # combined p-value of the KS test statistic over all parameters, and the
    # maximum Jensen-Shannon divergence over all parameters.
    values[idx] = np.inf
    pvalues = []
    jsvalues = []
    for p in priors.keys():
        _, pvalue = ks_2samp(post[p].samples[:,0], result.posterior[p])
        pvalues.append(pvalue)

        # calculate J-S divergence
        bins = np.linspace(
            np.min([np.min(post[p].samples[:,0]), np.min(result.posterior[p])]),
            np.max([np.max(post[p].samples[:,0]), np.max(result.posterior[p])]),
            100
        )

        hp, _ = np.histogram(post[p].samples[:,0], bins=bins, density=True)
        hq, _ = np.histogram(result.posterior[p], bins=bins, density=True)
        jsvalues.append(jensenshannon(hp, hq)**2)

    values[idx] = combine_pvalues(pvalues)[1]
    idx += 1
    values[idx] = np.max(jsvalues)

    with open(comparefile, 'w') as fp:
        fp.write(FILETEXT.format(*values))
