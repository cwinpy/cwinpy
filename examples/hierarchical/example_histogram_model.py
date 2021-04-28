#!/usr/bin/env python

"""
This example uses shows using the CWInPy hierarchical inference infrastructure
to fit a non-parameteric histogram style distribution to some data. To compare
the CWInPy code to an independent source we use the same data as in the
example from PosteriorStacker https://github.com/JohannesBuchner/PosteriorStacker
(values are shifted by 100 to make everything positive as required for the
MassQuadrupoleDistribution). The differences between this example and that
given in PosteriorStacker are:
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

# values from https://github.com/JohannesBuchner/PosteriorStacker/blob/main/tutorial/gendata.py (+100 to make positive)
values = np.array([15, 4, 2, 11, 1, -2, -1, -14, -39, -3]) + 100.0
values_lo = np.array([7, 16, 6, 3, 6, 5, 10, 6, 11, 13])
values_hi = np.array([7, 15, 8, 3, 6, 6, 10, 7, 14, 14])

n_data = len(values)

reslist = []

# set outer edges of distribution (shifted by 100 from the values of -80 and 80)
low = 20.0
high = 180.0

for i in range(n_data):
    # draw normal random points
    u = np.random.normal(size=400)
    v = values[i] + np.where(u < 0, u * values_lo[i], u * values_hi[i])

    res = Result(
        search_parameter_keys=["Q22"],
        posterior=pd.DataFrame({"Q22": v}),
        log_evidence=0.0,
        priors={"Q22": Uniform(low, high, name="Q22")},
    )

    reslist.append(res)

data = ResultList(reslist)

# set the parameters for the histogram distribution
distkwargs = {"low": low, "high": high, "nbins": 11}

intmethod = "numerical"
sampler_kwargs = {
    "nlive": 500,
    "gzip": True,
    "outdir": "histogram_distribution",
    "label": intmethod,
}

mqd = MassQuadrupoleDistribution(
    data=data,
    distribution="histogram",
    distkwargs=distkwargs,
    sampler_kwargs=sampler_kwargs,
    integration_method=intmethod,
    nsamples=500,  # number of Q22 samples to store/use,
    prependzero=False,  # this is required in this example
)

# perform sampling
res = mqd.sample()

# create corner plot
res.plot_corner()

# plot error bars on histogram bin values (equivalent to
# https://github.com/JohannesBuchner/PosteriorStacker#visualising-the-results)
bins_lo = mqd._distribution.binedges[:-1]
bins_hi = mqd._distribution.binedges[1:]

# add in final weight
posts = res.posterior.iloc[:, 0:10].values
finalbin = 1.0 - np.sum(posts, axis=1)
posts = np.hstack((posts, np.atleast_2d(finalbin).T))

# get bounds containing 68% of probability about the median
lo_hist = np.quantile(posts, 0.15865525393145707, axis=0)
mid_hist = np.quantile(posts, 0.5, axis=0)
hi_hist = np.quantile(posts, 0.8413447460685429, axis=0)

plt.errorbar(
    x=(bins_hi + bins_lo) / 2,
    xerr=(bins_hi - bins_lo) / 2,
    y=posts.mean(axis=0) / (bins_hi - bins_lo),
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
plt.xlabel(r"$Q_{22}$")
plt.ylabel("Probability density")
plt.legend(loc="best")

plt.show()
