from itertools import permutations

import numpy as np
from bilby.core.result import Result


def results_odds(results, oddstype="svn", scale="log10"):
    """
    Calculate the logarithm of the Bayesian odds between models given a set o
    evidence values. The type of odds can be one of the following:

    - "svn": signal vs. noise, i.e., the odds between a coherent signal in one,
      or multiple detectors.
    - "cvi": coherent vs. incoherent, i.e., for multiple detectors this is the
      odds between a coherent signal in add detectors and an incoherent signal
      between detectors _or_ noise.

    which are calculated from equations (30) and (32) of [1]_, respectively.

    Parameters
    ----------
    results: Result, dict
        A :class:`bilby.core.result.Result` or dictionary of
        :class:`bilby.core.result.Result` objects containing the output of
        parameter estimation of a signal for one or multiple detectors. These
        should each contain the attributes ``log_10_evidence`` and
        ``log_10_noise_evidence`` with the former providing the base-10
        logarithm for the signal model and the latter the base-10 logarithm of
        the data being consistent with noise. In inputting a dictionary, it
        should be keyed by two-character detector names, e.g., "H1", for the
        results from individual detectors, or the string "joint", "coherent" or
        a concatenation of all detector names for a coherent multi-detector
        result.
    oddstype: str
        The particular odds that should be calculated.
    scale: str:
        A flag saying whether the output should be in the base-10 logarithm
        ``"log10"`` (the default), or the natural logarithm ``"ln"``.
    """

    if not isinstance(results, (Result, dict)):
        raise TypeError("result must be a Result object or list of Result objects")

    if isinstance(results, Result):
        log10odds = results.log_10_evidence - results.log_10_noise_evidence
    else:
        # list of detectors
        dets = [det for det in results if len(det) == 2]

        if len(dets) == len(results) and len(results) > 1:
            raise RuntimeError("No 'coherent' multi-detector result is given")

        # get the key that contains the coherent multi-detector results
        coherentname = [("".join(detperm)).lower() for detperm in permutations(dets)]
        coherentname.extend(["joint", "coherent"])

        for key in results:
            if key.lower() in coherentname:
                break
        else:
            raise KeyError("No 'coherent' multi-detector result is given")

        coherentZ = results[key].log_10_evidence

        if oddstype == "svn":
            log10odds = coherentZ - results[key].log_10_noise_evidence
        else:
            # get the denominator of the coherent vs incoherent odds
            denom = 0.0
            for rkey in results:
                if rkey != key:
                    denom += np.logaddexp(
                        results[rkey].log_10_evidence,
                        results[rkey].log_10_noise_evidence,
                    )

            log10odds = coherentZ - denom

    return log10odds if scale == "log10" else log10odds / np.log10(np.exp(1))
