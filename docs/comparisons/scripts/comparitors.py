"""
Output various comparitors between lalapps_pulsar_parameter_estimation_nested
runs and cwinpy runs
"""

import os

import bilby
import cwinpy
import h5py
import numpy as np
from bilby.core.result import read_in_result
from lalinference import LALInferenceHDF5PosteriorSamplesDatasetName
from lalinference.io import read_samples
from scipy.spatial.distance import jensenshannon
from scipy.stats import combine_pvalues, gaussian_kde, ks_2samp

# comparison rst table information
FILETEXT = """\
.. csv-table:: Evidence table
   :widths: auto
   :header: "Method", ":math:`\\\\ln{{(Z)}}`", ":math:`\\\\ln{{(Z)}}` noise", ":math:`\\\\ln{{}}` Odds"

   "``lalapps_pulsar_parameter_estimation_nested``", "{0:.3f}", "{1:.3f}", "{2:.3f}±{3:.3f}"
   "``cwinpy_pe``", "{4:.3f}", "{5:.3f}", "{6:.3f}±{7:.3f}"
   "``cwinpy_pe`` (grid)", "{8}", "", "{9}"

.. csv-table:: Parameter table
   :widths: auto
   :header: "Method", ":math:`h_0`", ":math:`\\\\phi_0` (rad)", ":math:`\\\\psi` (rad)", ":math:`\\\\cos{{\\\\iota}}`"

   "``lalapps_pulsar_parameter_estimation_nested``", "{10:.2f}±{11:.2f}×10\ :sup:`{12:d}`", "{13:.2f}±{14:.2f}", "{15:.2f}±{16:.2f}", "{17:.2f}±{18:.2f}"
   "{19:d}% credible intervals", "[{20:.2f}, {21:.2f}]×10\ :sup:`{22:d}`", "[{23:.2f}, {24:.2f}]", "[{25:.2f}, {26:.2f}]", "[{27:.2f}, {28:.2f}]"
   "``cwinpy_pe``", "{29:.2f}±{30:.2f}×10\ :sup:`{31:d}`", "{32:.2f}±{33:.2f}", "{34:.2f}±{35:.2f}", "{36:.2f}±{37:.2f}"
   "{38:d}% credible intervals", "[{39:.2f}, {40:.2f}]×10\ :sup:`{41:d}`", "[{42:.2f}, {43:.2f}]", "[{44:.2f}, {45:.2f}]", "[{46:.2f}, {47:.2f}]"

.. csv-table:: Maximum a-posteriori
   :widths: auto
   :header: "Method", ":math:`h_0`", ":math:`\\\\phi_0` (rad)", ":math:`\\\\psi` (rad)", ":math:`\\\\cos{{\\\\iota}}`", ":math:`\\\\ln{{(L)}}` max"

   "``lalapps_pulsar_parameter_estimation_nested``", "{48:.2f}×10\ :sup:`{49:d}`", "{50:.2f}", "{51:.2f}", "{52:.2f}", "{53:.2f}"
   "``cwinpy_pe``", "{54:.2f}×10\ :sup:`{55:d}`", "{56:.2f}", "{57:.2f}", "{58:.2f}", "{59:.2f}"

| Combined K-S test p-value: {60:.4f}
| Maximum Jensen-Shannon divergence: {61:.4f}

| CWInPy version: {62:s}
| bilby version: {63:s}
"""


# comparison rst table information for two harmonics
FILETEXTTWOHARMONICS = """\
.. csv-table:: Evidence table
   :widths: auto
   :header: "Method", ":math:`\\\\ln{{(Z)}}`", ":math:`\\\\ln{{(Z)}}` noise", ":math:`\\\\ln{{}}` Odds"

   "``lalapps_pulsar_parameter_estimation_nested``", "{0:.3f}", "{1:.3f}", "{2:.3f}±{3:.3f}"
   "``cwinpy_pe``", "{4:.3f}", "{5:.3f}", "{6:.3f}±{7:.3f}"

.. csv-table:: Parameter table
   :widths: auto
   :header: "Method", ":math:`C_{{21}}`", ":math:`C_{{22}}`", ":math:`\\\\Phi_{{21}}` (rad)", ":math:`\\\\Phi_{{22}}` (rad)", ":math:`\\\\psi` (rad)", ":math:`\\\\cos{{\\\\iota}}`"

   "``lalapps_pulsar_parameter_estimation_nested``", "{8:.2f}±{9:.2f}×10\ :sup:`{10:d}`", "{11:.2f}±{12:.2f}×10\ :sup:`{13:d}`", "{14:.2f}±{15:.2f}", "{16:.2f}±{17:.2f}", "{18:.2f}±{19:.2f}", "{20:.2f}±{21:.2f}"
   "{22:d}% credible intervals", "[{23:.2f}, {24:.2f}]×10\ :sup:`{25:d}`", "[{26:.2f}, {27:.2f}]×10\ :sup:`{28:d}`", "[{29:.2f}, {30:.2f}]", "[{31:.2f}, {32:.2f}]", "[{33:.2f}, {34:.2f}]", "[{35:.2f}, {36:.2f}]"
   "``cwinpy_pe``", "{37:.2f}±{38:.2f}×10\ :sup:`{39:d}`", "{40:.2f}±{41:.2f}×10\ :sup:`{42:d}`", "{43:.2f}±{44:.2f}", "{45:.2f}±{46:.2f}", "{47:.2f}±{48:.2f}", "{49:.2f}±{50:.2f}"
   "{51:d}% credible intervals", "[{52:.2f}, {53:.2f}]×10\ :sup:`{54:d}`", "[{55:.2f}, {56:.2f}]×10\ :sup:`{57:d}`", "[{58:.2f}, {59:.2f}]", "[{60:.2f}, {61:.2f}]", "[{62:.2f}, {63:.2f}]", "[{64:.2f}, {65:.2f}]"

.. csv-table:: Maximum a-posteriori
   :widths: auto
   :header: "Method", ":math:`C_{{21}}`", ":math:`C_{{22}}`", ":math:`\\\\Phi_{{21}}` (rad)", ":math:`\\\\Phi_{{22}}` (rad)", ":math:`\\\\psi` (rad)", ":math:`\\\\cos{{\\\\iota}}`", ":math:`\\\\ln{{(L)}}` max"

   "``lalapps_pulsar_parameter_estimation_nested``", "{66:.2f}×10\ :sup:`{67:d}`", "{68:.2f}×10\ :sup:`{69:d}`", "{70:.2f}", "{71:.2f}", "{72:.2f}", "{73:.2f}", "{74:.2f}"
   "``cwinpy_pe``", "{75:.2f}×10\ :sup:`{76:d}`", "{77:.2f}×10\ :sup:`{78:d}`", "{79:.2f}", "{80:.2f}", "{81:.2f}", "{82:.2f}", "{83:.2f}"

| Combined K-S test p-value: {84:.4f}
| Maximum Jensen-Shannon divergence: {85:.4f}

| CWInPy version: {86:s}
| bilby version: {87:s}
"""


def credible_interval(samples, ci=0.90):
    # get the given percentage credible interval about the median
    return np.quantile(samples, [0.5 - 0.5 * ci, 0.5 + 0.5 * ci])


def comparisons(label, outdir, grid, priors, cred=0.9):
    """
    Perform comparisons of the evidence, parameter values, confidence
    intervals, and Kolmogorov-Smirnov test between samples produced with
    lalapps_pulsar_parameter_estimation_nested and cwinpy.
    """

    lppenfile = os.path.join(outdir, "{}_post.hdf".format(label))

    # get posterior samples
    post = read_samples(
        lppenfile, tablename=LALInferenceHDF5PosteriorSamplesDatasetName
    )

    # get uncertainty on ln(evidence)
    info = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "information_nats"
    ]
    nlive = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "number_live_points"
    ]
    evsig = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "log_evidence"
    ]
    evnoise = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "log_noise_evidence"
    ]
    everr = np.sqrt(info / nlive)  # the uncertainty on the evidence

    # read in cwinpy results
    result = read_in_result(outdir=outdir, label=label)

    # comparison file
    comparefile = os.path.join(outdir, "{}_compare.txt".format(label))

    # get grid-based evidence
    if grid is not None:
        grid_evidence = grid.log_evidence

    # set values to output
    values = 64 * [None]
    values[0:4] = evsig, evnoise, (evsig - evnoise), everr
    values[4:8] = (
        result.log_evidence,
        result.log_noise_evidence,
        result.log_bayes_factor,
        result.log_evidence_err,
    )
    if grid is not None:
        values[8] = "{0:.3f}".format(grid_evidence)
        values[9] = "{0:.3f}".format(grid_evidence - result.log_noise_evidence)
    else:
        values[8:10] = ("N/A", "N/A")  # no values supplied

    # output parameter means standard deviations, and credible intervals
    idx = 10
    for method in ["lalapps", "cwinpy"]:
        values[idx + 9] = int(cred * 100)
        for p in priors.keys():
            samples = post[p.upper()] if method == "lalapps" else result.posterior[p]

            # convert iota to cos(iota)
            if p == "iota":
                samples = np.cos(samples)

            mean = samples.mean()
            std = samples.std()
            low, high = credible_interval(samples, ci=cred)
            if p == "h0":
                exponent = int(np.floor(np.log10(mean)))
                values[idx] = mean / 10 ** exponent
                values[idx + 1] = std / 10 ** exponent
                values[idx + 2] = exponent
                values[idx + 10] = low / 10 ** exponent
                values[idx + 11] = high / 10 ** exponent
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
    maxidx = (
        result.posterior["log_likelihood"] + result.posterior["log_prior"]
    ).idxmax()
    maxidxlppen = (post["logL"] + post["logPrior"]).argmax()
    for method in ["lalapps", "cwinpy"]:
        for p in priors.keys():
            maxpval = (
                post[p.upper()][maxidxlppen]
                if method == "lalapps"
                else result.posterior[p][maxidx]
            )
            if p == "h0":
                exponent = int(np.floor(np.log10(maxpval)))
                values[idx] = maxpval / 10 ** exponent
                values[idx + 1] = exponent
                idx += 2
            else:
                values[idx] = maxpval
                idx += 1
        if result.use_ratio:
            # convert likelihood ratio back to likelihood
            values[idx] = (
                post["logL"][maxidxlppen]
                if method == "lalapps"
                else (
                    result.posterior["log_likelihood"][maxidx]
                    + result.log_noise_evidence
                )
            )
        else:
            values[idx] = (
                post["logL"][maxidxlppen]
                if method == "lalapps"
                else result.posterior["log_likelihood"][maxidx]
            )
        idx += 1

    # calculate the Kolmogorov-Smirnov test for each 1d marginalised distribution,
    # and the Jensen-Shannon divergence, from the two codes. Output the
    # combined p-value of the KS test statistic over all parameters, and the
    # maximum Jensen-Shannon divergence over all parameters.
    values[idx] = np.inf
    pvalues = []
    jsvalues = []
    for p in priors.keys():
        _, pvalue = ks_2samp(post[p.upper()], result.posterior[p])
        pvalues.append(pvalue)

        # calculate J-S divergence (use Gaussian KDE)
        bins = np.linspace(
            np.min([np.min(post[p.upper()]), np.min(result.posterior[p])]),
            np.max([np.max(post[p.upper()]), np.max(result.posterior[p])]),
            100,
        )

        hp = gaussian_kde(post[p.upper()]).pdf(bins)
        hq = gaussian_kde(result.posterior[p]).pdf(bins)
        jsvalues.append(jensenshannon(hp, hq) ** 2)

    values[idx] = combine_pvalues(pvalues)[1]
    idx += 1
    values[idx] = np.max(jsvalues)

    values[idx + 1] = cwinpy.__version__
    values[idx + 2] = bilby.__version__

    with open(comparefile, "w") as fp:
        fp.write(FILETEXT.format(*values))


def comparisons_two_harmonics(label, outdir, priors, cred=0.9):
    """
    Perform comparisons of the evidence, parameter values, confidence
    intervals, and Kolmogorov-Smirnov test between samples produced with
    lalapps_pulsar_parameter_estimation_nested and cwinpy.
    """

    lppenfile = os.path.join(outdir, "{}_post.hdf".format(label))

    # get posterior samples
    post = read_samples(
        lppenfile, tablename=LALInferenceHDF5PosteriorSamplesDatasetName
    )

    # get uncertainty on ln(evidence)
    info = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "information_nats"
    ]
    nlive = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "number_live_points"
    ]
    evsig = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "log_evidence"
    ]
    evnoise = h5py.File(lppenfile)["lalinference"]["lalinference_nest"].attrs[
        "log_noise_evidence"
    ]
    everr = np.sqrt(info / nlive)  # the uncertainty on the evidence

    # read in cwinpy results
    result = read_in_result(outdir=outdir, label=label)

    # comparison file
    comparefile = os.path.join(outdir, "{}_compare.txt".format(label))

    # set values to output
    values = 88 * [None]
    values[0:4] = evsig, evnoise, (evsig - evnoise), everr
    values[4:8] = (
        result.log_evidence,
        result.log_noise_evidence,
        result.log_bayes_factor,
        result.log_evidence_err,
    )

    # output parameter means standard deviations, and credible intervals
    idx = 8
    for method in ["lalapps", "cwinpy"]:
        values[idx + 14] = int(cred * 100)
        for p in priors.keys():
            samples = post[p.upper()] if method == "lalapps" else result.posterior[p]

            # convert iota to cos(iota)
            if p == "iota":
                samples = np.cos(samples)

            mean = samples.mean()
            std = samples.std()
            low, high = credible_interval(samples, ci=cred)
            if p in ["c21", "c22"]:
                exponent = int(np.floor(np.log10(mean)))
                values[idx] = mean / 10 ** exponent
                values[idx + 1] = std / 10 ** exponent
                values[idx + 2] = exponent
                values[idx + 15] = low / 10 ** exponent
                values[idx + 16] = high / 10 ** exponent
                values[idx + 17] = exponent
                idx += 3
            else:
                values[idx] = mean
                values[idx + 1] = std
                values[idx + 15] = low
                values[idx + 16] = high
                idx += 2
        idx += 15

    # output parameter maximum a-posteriori points
    maxidx = (
        result.posterior["log_likelihood"] + result.posterior["log_prior"]
    ).idxmax()
    maxidxlppen = (post["logL"] + post["logPrior"]).argmax()
    for method in ["lalapps", "cwinpy"]:
        for p in priors.keys():
            maxpval = (
                post[p.upper()][maxidxlppen]
                if method == "lalapps"
                else result.posterior[p][maxidx]
            )
            if p in ["c21", "c22"]:
                exponent = int(np.floor(np.log10(maxpval)))
                values[idx] = maxpval / 10 ** exponent
                values[idx + 1] = exponent
                idx += 2
            else:
                values[idx] = maxpval
                idx += 1
        if result.use_ratio:
            values[idx] = (
                post["logL"][maxidxlppen]
                if method == "lalapps"
                else (
                    result.posterior["log_likelihood"][maxidx]
                    + result.log_noise_evidence
                )
            )
        else:
            values[idx] = (
                post["logL"][maxidxlppen]
                if method == "lalapps"
                else result.posterior["log_likelihood"][maxidx]
            )
        idx += 1

    # calculate the Kolmogorov-Smirnov test for each 1d marginalised distribution,
    # and the Jensen-Shannon divergence, from the two codes. Output the
    # combined p-value of the KS test statistic over all parameters, and the
    # maximum Jensen-Shannon divergence over all parameters.
    values[idx] = np.inf
    pvalues = []
    jsvalues = []
    for p in priors.keys():
        _, pvalue = ks_2samp(post[p.upper()], result.posterior[p])
        pvalues.append(pvalue)

        # calculate J-S divergence
        bins = np.linspace(
            np.min([np.min(post[p.upper()]), np.min(result.posterior[p])]),
            np.max([np.max(post[p.upper()]), np.max(result.posterior[p])]),
            100,
        )

        hp, _ = np.histogram(post[p.upper()], bins=bins, density=True)
        hq, _ = np.histogram(result.posterior[p], bins=bins, density=True)
        jsvalues.append(jensenshannon(hp, hq) ** 2)

    values[idx] = combine_pvalues(pvalues)[1]
    idx += 1
    values[idx] = np.max(jsvalues)

    values[idx + 1] = cwinpy.__version__
    values[idx + 2] = bilby.__version__

    with open(comparefile, "w") as fp:
        fp.write(FILETEXTTWOHARMONICS.format(*values))
