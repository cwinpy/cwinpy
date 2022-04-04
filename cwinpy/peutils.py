import re
from copy import deepcopy
from itertools import permutations
from pathlib import Path

import numpy as np
from bilby.core.result import Result, read_in_result

from .data import MultiHeterodynedData


def results_odds(results, oddstype="svn", scale="log10", **kwargs):
    """
    Calculate the logarithm of the Bayesian odds between models given a set of
    evidence values. The type of odds can be one of the following:

    * "svn": signal vs. noise, i.e., the odds between a coherent signal in one,
      or multiple detectors.
    * "cvi": coherent vs. incoherent, i.e., for multiple detectors this is the
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
        the data being consistent with noise. If inputting a dictionary, it
        should be keyed by two-character detector names, e.g., "H1", for the
        results from individual detectors, or the string "joint", "coherent" or
        a concatenation of all detector names for a coherent multi-detector
        result. Alternatively, ``results`` can be a directory, within which it
        is assumed that each subdirectory is named after a pulsar and contains
        results files with the format
        ``{fnamestr}_{det}_{psrname}_result.[hdf5,json]``, where the default
        ``fnamestr`` is ``cwinpy_pe``, ``det`` is the two-character detector
        name, or concantenation of multiple detector names, and ``psrname`` is
        the same as the directory name. In this case a dictionary of odds
        values, keyed to the pulsar name, will be calculated and returned.
    oddstype: str
        The particular odds that should be calculated.
    scale: str:
        A flag saying whether the output should be in the base-10 logarithm
        ``"log10"`` (the default), or the natural logarithm ``"ln"``.
    """

    if not isinstance(results, (str, Path, Result, dict)):
        raise TypeError("result must be a Result object or list of Result objects")

    if isinstance(results, Result):
        log10odds = results.log_10_evidence - results.log_10_noise_evidence

        return log10odds if scale == "log10" else log10odds / np.log10(np.e)
    else:
        if not isinstance(results, dict):
            respath = Path(results)

            if respath.is_file():
                result = read_in_result_wrapper(results)
                log10odds = result.log_10_evidence - result.log_10_noise_evidence
            elif respath.is_dir():
                # iterate through directories
                resfiles = {}

                for rd in respath.iterdir():
                    if rd.is_dir():
                        dname = rd.name

                        # check directory contains results objects
                        for ext in ["hdf5", "json"]:
                            fnamestr = kwargs.get("fnamestr", "cwinpy_pe")
                            fnamematch = f"{fnamestr}_*_{dname}_result.{ext}"
                            rfiles = list(rd.glob(fnamematch))
                            if len(rfiles) > 0:
                                break

                        if len(rfiles) > 0:
                            resfiles[dname] = {}

                            for rf in rfiles:
                                # extract detector name
                                detmatch = re.search(
                                    f"{fnamestr}_(.*?)_{dname}", str(rf)
                                )
                                if detmatch is None:
                                    raise RuntimeError(
                                        f"{rd} contains incorrectly named results file '{rf}'"
                                    )

                                # set files
                                resfiles[dname][detmatch.group(1)] = rf

                if len(resfiles) == 0:
                    raise ValueError(f"{results} contains no valid results files")
        else:
            resfiles = {"dummyname": results}

        logodds = {}

        for pname, resultd in resfiles.items():
            # list of detectors
            dets = [det for det in resultd if len(det) == 2]

            if len(dets) == len(resultd) and len(resultd) > 1:
                raise RuntimeError("No 'coherent' multi-detector result is given")

            # get the key that contains the coherent multi-detector results
            coherentname = [
                ("".join(detperm)).lower() for detperm in permutations(dets)
            ]
            coherentname.extend(["joint", "coherent"])

            for key in resultd:
                if key.lower() in coherentname:
                    break
            else:
                raise KeyError("No 'coherent' multi-detector result is given")

            result = read_in_result_wrapper(resultd[key])

            coherentZ = result.log_10_evidence

            if oddstype == "svn":
                log10odds = coherentZ - result.log_10_noise_evidence
            else:
                # get the denominator of the coherent vs incoherent odds
                denom = 0.0
                for rkey in resultd:
                    if rkey != key:
                        result = read_in_result_wrapper(resultd[rkey])

                        denom += np.logaddexp(
                            result.log_10_evidence,
                            result.log_10_noise_evidence,
                        )

                log10odds = coherentZ - denom

            if pname == "dummyname":
                return log10odds if scale == "log10" else log10odds / np.log10(np.e)
            else:
                logodds[pname] = (
                    log10odds if scale == "log10" else log10odds / np.log10(np.e)
                )

    return logodds


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


def optimal_snr(res, het, par=None, which="posterior"):
    """
    Calculate the optimal matched filter signal-to-noise ratio for a signal in
    given data based on the posterior samples. This can either be the
    signal-to-noise ratio for the maximum a-posteriori sample or for the
    maximum likelihood sample.

    Parameters
    ----------
    res: str, Result
        The path to a :class:`~bilby.core.result.Result` object file or a
        :class:`~bilby.core.result.Result` object itself containing posterior
        samples and priors.
    het: str, dict, HeterodynedData, MultiHeterodynedData
        The path to a :class:`~cwinpy.data.HeterodynedData` object file or a
        :class:`~cwinpy.data.HeterodynedData` object itself containing the
        heterodyned data that was used for parameter estimation. Or, a
        dictionary (keyed to detector names) containing individual paths to or
        :class:`~cwinpy.data.HeterodynedData` objects.
    par: str, PulsarParameters
        If the heterodyned data provided with ``het`` does not contain a pulsar
        parameter (``.par``) file, then it can be specified here.
    which: str
        A string stating whether to calculate the SNR using the maximum
        a-posteriori (``"posterior"``) or maximum likelihood (``"likelihood"``)
        sample.

    Returns
    -------
    snr: float
        The matched filter signal-to-noise ratio.
    """

    # get results
    resdata = read_in_result_wrapper(res)

    post = resdata.posterior
    prior = resdata.priors

    hetdata = MultiHeterodynedData(het, par=par)

    if hetdata.pars[0] is None:
        raise ValueError("No pulsar parameter file is given")

    # store copy of pulsar parameter file
    par = deepcopy(hetdata.pars[0])

    # get index of required sample
    if which.lower() == "likelihood":
        idx = post.log_likelihood.argmax()
    elif which.lower() == "posterior":
        idx = (post.log_likelihood + post.log_prior).argmax()
    else:
        raise ValueError("'which' must be 'posterior' or 'likelihood'")

    # update parameter file with signal values
    for key in prior:
        par[key.upper()] = post[key].iloc[idx]

    # get snr
    snr = hetdata.signal_snr(par)

    return snr
