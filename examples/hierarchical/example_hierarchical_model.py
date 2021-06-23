#!/usr/bin/env python

"""
This example uses shows using the CWInPy hierarchical inference infrastructure
to fit a both a non-parameteric histogram style distribution and a Gaussian
distribution to some data consisting of "posterior" samples on amplitude from
multiple objects. To compare the CWInPy code to an independent source we use
the same data as in the example from PosteriorStacker
https://github.com/JohannesBuchner/PosteriorStacker (posterior values there are
velocities, which in our case have been shifted by 100 to make everything
positive as required for the ampitude parameter in MassQuadrupoleDistribution).
The differences between this example and that given in PosteriorStacker are:
    * use of numerical integration of KDEs of posterior samples rather than
      calculating the expectation value
    * use of dynesty as the sample rather than UltraNest

Both of these could easily be changed to match.
"""

import numpy as np
import pandas as pd
from bilby.core.prior import Uniform
from bilby.core.result import Result, ResultList
from cwinpy.hierarchical import MassQuadrupoleDistribution
from matplotlib import pyplot as plt

# values from https://github.com/JohannesBuchner/PosteriorStacker/blob/main/tutorial/gendata.py
# (+100 to make positive)
values = np.array([15, 4, 2, 11, 1, -2, -1, -14, -39, -3]) + 100.0
values_lo = np.array([7, 16, 6, 3, 6, 5, 10, 6, 11, 13])
values_hi = np.array([7, 15, 8, 3, 6, 6, 10, 7, 14, 14])
n_data = len(values)

reslist = []

# set outer edges of distribution (shifted by 100 from the values of -80 and 80)
low = 20.0
high = 180.0

# draw "posterior samples" for each object based on the above values
for i in range(n_data):
    # draw normal random points
    u = np.random.normal(size=400)
    v = values[i] + np.where(u < 0, u * values_lo[i], u * values_hi[i])

    # put data into bilby Result object
    res = Result(
        search_parameter_keys=["Q22"],
        posterior=pd.DataFrame({"Q22": v}),
        log_evidence=0.0,
        priors={"Q22": Uniform(low, high, name="Q22")},
    )

    reslist.append(res)

data = ResultList(reslist)

### Use histogram model ###  # noqa: E266

# set the parameters for the histogram distribution
distkwargs = {"low": low, "high": high, "nbins": 11}

intmethod = "numerical"
sampler_kwargs = {
    "nlive": 500,
    "gzip": True,
    "outdir": "example_distribution",
    "label": "histogram",
}

mqdhist = MassQuadrupoleDistribution(
    data=data,
    distribution="histogram",
    distkwargs=distkwargs,
    sampler_kwargs=sampler_kwargs,
    integration_method=intmethod,
    nsamples=500,  # number of Q22 samples to store/use,
)

# perform sampling
reshist = mqdhist.sample()

### Use Gaussian model ###  # noqa: E266

# set the parameter priors for the Gaussian distribution (equivlant to those in
# https://github.com/JohannesBuchner/PosteriorStacker/blob/main/posteriorstacker.py)
distkwargs = {
    "mus": Uniform(low, low + ((high - low) * 3), name="mu"),
    "sigmas": Uniform(
        0.5, (high - low) * 3, name="sigma"
    ),  # allowing this to have a minimum at zero causes issues when using the "numerical" method
}

sampler_kwargs = {
    "nlive": 500,
    "gzip": True,
    "outdir": "example_distribution",
    "label": "gaussian",
    "sample": "unif",  # uniform sampling
}

mqdgauss = MassQuadrupoleDistribution(
    data=data,
    distribution="gaussian",
    distkwargs=distkwargs,
    sampler_kwargs=sampler_kwargs,
    integration_method=intmethod,
    nsamples=500,  # number of Q22 samples to store/use,
)

# perform sampling
resgauss = mqdgauss.sample()

### Plot results ###  # noqa: E266

fig, ax = plt.subplots(figsize=(5, 3))

# add posterior predictive plot for Gaussian distribution
q22values = np.linspace(low, high, 400)
for m in mqdgauss.posterior_predictive(q22values, nsamples=100):
    ax.plot(q22values, m, color="r", alpha=0.1)

# get maximum a-posteriori value
idxmax = np.argmax(
    (resgauss.posterior["log_likelihood"] + resgauss.posterior["log_prior"]).values
)
mumax = resgauss.posterior["mu0"][idxmax]
sigmamax = resgauss.posterior["sigma0"][idxmax]
ax.plot(
    q22values,
    mqdgauss._distribution.pdf(q22values, {"mu0": mumax, "sigma0": sigmamax}),
    color="r",
    lw=2,
    label="Gaussian model",
)

# plot error bars on histogram bin values (equivalent to
# https://github.com/JohannesBuchner/PosteriorStacker#visualising-the-results)
bins_lo = mqdhist._distribution.binedges[:-1]
bins_hi = mqdhist._distribution.binedges[1:]

# add in final weight
postshist = reshist.posterior.iloc[:, 0:10].values
finalbin = 1.0 - np.sum(postshist, axis=1)
postshist = np.hstack((postshist, np.atleast_2d(finalbin).T))

# get bounds containing 68% of probability about the median
lo_hist = np.quantile(postshist, 0.15865525393145707, axis=0)
mid_hist = np.quantile(postshist, 0.5, axis=0)
hi_hist = np.quantile(postshist, 0.8413447460685429, axis=0)

ax.errorbar(
    x=(bins_hi + bins_lo) / 2,
    xerr=(bins_hi - bins_lo) / 2,
    y=postshist.mean(axis=0) / (bins_hi - bins_lo),
    yerr=[
        (mid_hist - lo_hist) / (bins_hi - bins_lo),
        (hi_hist - mid_hist) / (bins_hi - bins_lo),
    ],
    marker="o",
    linestyle=" ",
    color="k",
    label="Histogram model",
    capsize=2,
)
ax.set_xlabel(r"$Q_{22}$")
ax.set_ylabel("Probability density")
ax.legend(loc="best")

fig.savefig("hierarchical_example.png", dpi=200, bbox_inches="tight")
