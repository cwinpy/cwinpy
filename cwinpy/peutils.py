from itertools import permutations
from pathlib import Path

import numpy as np
from bilby.core.result import Result, read_in_result


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
    results: Result, dict, str
        A :class:`~bilby.core.result.Result` or dictionary of
        :class:`~bilby.core.result.Result` objects (or file path to a file
        containing such an object) containing the output of parameter
        estimation of a signal for one or multiple detectors. These should each
        contain the attributes ``log_10_evidence`` and
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

    if not isinstance(results, (str, Path, Result, dict)):
        raise TypeError("result must be a Result object or list of Result objects")

    if isinstance(results, (str, Path, Result)):
        result = read_in_result_wrapper(results)

        log10odds = result.log_10_evidence - result.log_10_noise_evidence
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

        result = read_in_result_wrapper(results[key])

        coherentZ = result.log_10_evidence

        if oddstype == "svn":
            log10odds = coherentZ - result.log_10_noise_evidence
        else:
            # get the denominator of the coherent vs incoherent odds
            denom = 0.0
            for rkey in results:
                result = read_in_result_wrapper(results[rkey])

                if rkey != key:
                    denom += np.logaddexp(
                        result.log_10_evidence,
                        result.log_10_noise_evidence,
                    )

            log10odds = coherentZ - denom

    return log10odds if scale == "log10" else log10odds / np.log10(np.exp(1))


def read_in_result_wrapper(res):
    """
    Check if argument is a :class:`~bilby.core.result.Result` object or a path
    to a file containing a :class:`~bilby.core.result.Result` object. If the
    former, just return the result, otherwise try reading in the file and
    return the loaded object.

    Parameters
    ----------
    res: str, Result
        A string with a path to a :class:`~bilby.core.result.Result` object.

    Returns
    -------
    result: Result
        The read in :class:`~bilby.core.result.Result` object or the original
        :class:`~bilby.core.result.Result` object.
    """

    if isinstance(res, Result):
        return res

    try:
        result = read_in_result(res)
    except (ValueError, TypeError, OSError):
        raise IOError(f"Could not read in results file '{res}'")

    return result


def skyshift_results():
    pass
