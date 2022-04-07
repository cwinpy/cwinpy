import re
from copy import deepcopy
from itertools import permutations
from pathlib import Path

import numpy as np
from bilby.core.result import Result, read_in_result
from matplotlib import pyplot as plt

from .data import HeterodynedData, MultiHeterodynedData
from .parfile import PulsarParameters
from .utils import get_psr_name


def results_odds(results, oddstype="svn", scale="log10", **kwargs):
    """
    Calculate the logarithm of the Bayesian odds between models given a set of
    evidence values. The type of odds can be one of the following:

    * "svn": signal vs. noise, i.e., the odds between a coherent signal in one,
      or multiple detectors, and the data being consistent with noise.
    * "cvi": coherent vs. incoherent, i.e., for multiple detectors this is the
      odds between a coherent signal in all detectors and an incoherent signal
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

    Returns
    -------
    log10odds: float, dict
        If using a single result this will be a float giving the requested
        log-odds value. If a directory of results is used, then this this will
        a dictionary of log-odds values for each source in the directory keyed
        on the source's name (based on the sub-directory name containing the
        result).
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

                return log10odds if scale == "log10" else log10odds / np.log10(np.e)
            elif respath.is_dir():
                resfiles = find_results_files(
                    respath, fnamestr=kwargs.get("fnamestr", "cwinpy_pe")
                )

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


def optimal_snr(res, het, par=None, det=None, which="posterior", remove_outliers=False):
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
        samples and priors. Alternatively, this can be a directory containing
        sub-directories, named by source name, that themselves contain results.
        In this case the SNRs for all sources will be calculated.
    het: str, dict, HeterodynedData, MultiHeterodynedData
        The path to a :class:`~cwinpy.data.HeterodynedData` object file or a
        :class:`~cwinpy.data.HeterodynedData` object itself containing the
        heterodyned data that was used for parameter estimation. Or, a
        dictionary (keyed to detector names) containing individual paths to or
        :class:`~cwinpy.data.HeterodynedData` objects. Alternatively, this can
        be a directory containing heterodyned data for all sources given in the
        ``res`` argument.
    par: str, PulsarParameters
        If the heterodyned data provided with ``het`` does not contain a pulsar
        parameter (``.par``) file, then the file or directory containing the
        file(s) can be specified here.
    det: str
        If passing results containing multiple detectors and joint detector
        analyses, but only requiring SNRs for an individual detector, then this
        can be specified here. By default, SNR for all detectors will be
        calculated.
    which: str
        A string stating whether to calculate the SNR using the maximum
        a-posteriori (``"posterior"``) or maximum likelihood (``"likelihood"``)
        sample.
    remove_outliers: bool
        Set when to remove outliers from data before calculating the SNR. This
        defaults to False.

    Returns
    -------
    snr: float, dict
        The matched filter signal-to-noise ratio(s). This is a float if a
        single source and single detector are requested. If a single source is
        requested, but multiple detectors, then this will be a dictionary keyed
        by detector prefix. If multiple sources are requested, then this will
        be a dictionary keyed on source name.
    """

    # get posterior results files
    if isinstance(res, (str, Path)):
        if Path(res).is_dir():
            resfiles = find_results_files(res)
        else:
            resfiles = {"dummyname": {"dummydet": res}}
    elif isinstance(res, Result):
        resfiles = {"dummyname": {"dummydet": res}}
    else:
        raise TypeError("res should be a file/directory path or Result object")

    # get heterodyned data files
    if isinstance(het, (str, Path)):
        if Path(het).is_dir():
            hetfiles = find_heterodyned_files(het)

            if len(hetfiles) == 0:
                raise ValueError(f"No heterodyned files could be found in {het}")

            # check for consistent sources
            if "dummyname" not in resfiles:
                respsrs = set(resfiles.keys())
                hetpsrs = (
                    set(hetfiles[0].keys())
                    if isinstance(hetfiles, tuple)
                    else set(hetfiles.keys())
                )

                if not respsrs.issubset(hetpsrs):
                    raise RuntimeError(
                        "Heterodyned data files are not present for all required sources"
                    )
        else:
            hetfiles = {"dummyname": {"dummydet": het}}
    elif isinstance(het, (HeterodynedData, MultiHeterodynedData, dict)):
        hetfiles = {"dummyname": {"dummydet": het}}
    else:
        raise TypeError("het should be a file/directory path or HeterodynedData object")

    # get par files
    if isinstance(par, (str, Path)):
        if Path(par).is_dir():
            parfiles = {
                get_psr_name(PulsarParameters(pf)): pf for pf in Path(par).glob("*.par")
            }
            for pname in parfiles:
                if pname not in resfiles:
                    parfiles[pname] = None
    else:
        parfiles = {psr: par for psr in resfiles}

    snrs = {}

    for psr in resfiles:
        # get results
        if "dummydet" in resfiles[psr]:
            resdata = {"dummydet": read_in_result_wrapper(resfiles[psr]["dummydet"])}
            inddets = ["dummydet"]
            muldets = []
        else:
            dets = list(resfiles[psr].keys()) if det is None else [det]
            resdata = {d: read_in_result_wrapper(resfiles[psr][d]) for d in dets}

            # get individual detectors and multi-detectors (assuming two-character detector strings)
            inddets = [d for d in dets if len(d) == 2]
            muldets = [
                re.findall("..", d) for d in dets if len(d) > 3 and not len(d) % 2
            ]

            if len(muldets) > 0:
                for d in set([d for dl in muldets for d in dl]):
                    if d not in inddets:
                        inddets.append(d)

                mhd = MultiHeterodynedData(
                    par=parfiles[psr],
                    bbminlength=np.inf,
                    remove_outliers=remove_outliers,
                )

        if which.lower() not in ["likelihood", "posterior"]:
            raise ValueError("which must be 'likelihood' or 'posterior'")

        snrs[psr] = {}

        # get individual detector SNRs
        for d in inddets:
            # bbminlength is inf, so no Bayesian Blocks is performed
            mhddet = MultiHeterodynedData(
                par=parfiles[psr], bbminlength=np.inf, remove_outliers=remove_outliers
            )

            if isinstance(hetfiles, tuple):
                if not all([d in hf[psr] for hf in hetfiles]):
                    # no heterodyned data available so skip
                    continue

                for hf in hetfiles:
                    mhddet.add_data(hf[psr][d])

                    if len(muldets) > 0:
                        mhd.add_data(hf[psr][d])
            else:
                if isinstance(hetfiles[psr][d], (MultiHeterodynedData, dict)):
                    for hd in hetfiles[psr][d]:
                        if isinstance(hd, HeterodynedData):
                            mhddet.add_data(hd)
                        else:
                            mhddet.add_data(hetfiles[psr][d][hd])
                else:
                    mhddet.add_data(hetfiles[psr][d])

                    if len(muldets) > 0:
                        mhd.add_data(hetfiles[psr][d])

            if mhddet.pars[0] is None:
                raise ValueError("No pulsar parameter file is given")

            if d in resdata:
                # get snrs for individual detectors
                post = resdata[d].posterior
                prior = resdata[d].priors

                # store copy of pulsar parameter file
                parc = deepcopy(mhddet.pars[0])

                # get index of required sample
                idx = (
                    post.log_likelihood.argmax()
                    if which.lower() == "likelihood"
                    else (post.log_likelihood + post.log_prior).argmax()
                )

                # update parameter file with signal values
                for key in prior:
                    parc[key.upper()] = post[key].iloc[idx]

                # get snr
                snrs[psr][d] = mhddet.signal_snr(parc)

        # get multidetector SNRs
        for ds in muldets:
            mhdmulti = MultiHeterodynedData(
                par=parfiles[psr], bbminlength=np.inf, remove_outliers=remove_outliers
            )

            for d in ds:
                for hd in mhd[d]:
                    mhdmulti.add_data(hd)

            # get snrs for individual detectors
            post = resdata["".join(ds)].posterior
            prior = resdata["".join(ds)].priors

            # store copy of pulsar parameter file
            parc = deepcopy(mhddet.pars[0])

            # get index of required sample
            idx = (
                post.log_likelihood.argmax()
                if which.lower() == "likelihood"
                else (post.log_likelihood + post.log_prior).argmax()
            )

            # update parameter file with signal values
            for key in prior:
                parc[key.upper()] = post[key].iloc[idx]

            # get snr
            snrs[psr]["".join(ds)] = mhdmulti.signal_snr(parc)

    if len(snrs) == 1:
        if len(list(snrs.keys())) == 1:
            # dictionary contains a single value
            return [item for p in snrs for item in snrs[p].values()][0]
        else:
            # dictionary for different detectors
            return snrs[list(snrs.keys())[0]]
    else:
        if all([len(item) == 1 for item in snrs.values()]):
            # only a single detector per source
            return {p: list(snrs[p].values())[0] for p in snrs}
        else:
            return snrs


def find_results_files(resdir, fnamestr="cwinpy_pe"):
    """
    Given a directory, go through all subdirectories and check if they contain
    results from cwinpy_pe. If they do, add them to a dictionary, keyed on the
    subdirectory name, with a subdictionary containing the path to the results
    for each detector. It is assumed that the file names have the format:
    ``{fnamestr}_{det}_{dname}_result.[hdf5,json]``, where ``fnamestr``
    defaults to ``cwinpy_pe``, ``dname`` is the directory name (assumed to be
    the name of the source, e.g., a pulsar J-name), and ``det`` is the
    two-character detector alias (e.g., ``H1`` for the LIGO Hanford detector),
    or concatenation of multiple detector names if the results are from a joint
    analysis.

    Parameters
    ----------
    resdir: str, Path
        The directory containing the results sub-directories.
    fnamestr: str
        A prefix for the results file names.

    Returns
    -------
    rfiles: dict
        A dictionary containing subdictionaries with all the file paths.
    """

    if not isinstance(resdir, (str, Path)):
        raise TypeError(f"'{resdir}' must be a string or a Path object")

    respath = Path(resdir)
    if not respath.is_dir():
        raise ValueError(f"'{resdir}' is not a directory")

    # iterate through directories
    resfiles = {}

    for rd in respath.iterdir():
        if rd.is_dir():
            dname = rd.name

            # check directory contains results objects
            for ext in ["hdf5", "json"]:
                fnamematch = f"{fnamestr}_*_{dname}_result.{ext}"
                rfiles = list(rd.glob(fnamematch))
                if len(rfiles) > 0:
                    break

            if len(rfiles) > 0:
                resfiles[dname] = {}

                for rf in rfiles:
                    # extract detector name
                    detmatch = re.search(f"{fnamestr}_(.*?)_{dname}", str(rf))
                    if detmatch is None:
                        raise RuntimeError(
                            f"{rd} contains incorrectly named results file '{rf}'"
                        )

                    # set files
                    resfiles[dname][detmatch.group(1)] = rf.resolve()

    return resfiles


def find_heterodyned_files(hetdir, ext="hdf5"):
    """
    Given a directory, find the heterodyned data files and sort them into a
    dictionary keyed on the source name with each value being a sub-dictionary
    keyed by detector name and pointing to the heterodyned file. This assumes
    that files are in a format where they start with:
    ``heterodyne_{sourcename}_{det}_{freqfactor}``. If data for more than one
    frequency factor exists then the output will be a tuple of dictionaries for
    each frequency factor, starting with the lowest.

    Parameters
    ----------
    hetdir: str, Path
        The path to the directory within which heterodyned data files will be
        searched for.
    ext: str
        The expected file extension of the heterodyned data files. This
        defaults to ``hdf5``.
    """

    if not isinstance(hetdir, (str, Path)):
        raise TypeError("hetdir must be a string or Path object")

    hetpath = Path(hetdir)
    if not hetpath.is_dir():
        raise ValueError(f"{hetdir} is not a directory")

    # get all files with correct extension
    allfiles = list(hetpath.rglob(f"*.{ext}"))

    if len(allfiles) == 0:
        # no files found
        return {}

    # file name format to match
    retext = r"heterodyne_(?P<psrname>\S+)_(?P<det>\S+)_(?P<freqfactor>\S+)_"

    # loop through files
    filedict = {}
    for hetfile in allfiles:
        try:
            finfo = re.match(retext, hetfile.name).groupdict()
        except AttributeError:
            # did not match so skip file
            continue

        if finfo["freqfactor"] not in filedict:
            filedict[finfo["freqfactor"]] = {}

        if finfo["psrname"] not in filedict[finfo["freqfactor"]]:
            filedict[finfo["freqfactor"]][finfo["psrname"]] = {}

        filedict[finfo["freqfactor"]][finfo["psrname"]][
            finfo["det"]
        ] = hetfile.resolve()

    if len(filedict) == 0:
        # no files found
        return {}
    elif len(filedict) == 1:
        return list(filedict.values())[0]
    else:
        return (filedict[key] for key in sorted(filedict.keys()))


def plot_snr_vs_odds(S, R, **kwargs):
    """
    Plot the signal-to-noise ratio for a set of sources versus their Bayesian
    odds. The inputs can either be a dictionary of SNRs, keyed on source name,
    and a dictionary of odds values, also keyed on source name, or it can be a
    directory path containing a set of cwinpy_pe parameter estimation results
    and a directory containing heterodyned data. In the later case, the
    :func:`cwinpy.peutils.optimal_snr` and :func:`cwinpy.peutils.results_odds`
    functions will be used to calculate the respective values.

    Parameters
    ----------
    S: str, Path, dict
        A dictionary of signal-to-noise ratios, or a path to a directory
        containing heterodyned data files for a set of sources.
    R: str, Path, dict
        A dictionary of odds values, or a path to a directory containing
        parameter estimation results for a set of sources.
    oddstype: str
        The type of odds (``"svn"`` or ``"cvi"``, see
        :func:`~cwinpy.peutils.results_odds`) for calculating the odds and/or
        using on the figure axis label.
    scale: str
        The scale for the odds (``"log10"`` or ``"ln"``, see
        :func:`~cwinpy.peutils.results_odds`) for calculating the odds and/or
        using on the figure axis label.
    which: str
        Whether to calculate SNRs using the maximum a-posterior value or
        maximum likelihood value (see
        :func:`~cwinpy.peutils.optimal_snr`).
    remove_outliers: bool
        Whether to remove outliers before calculating SNRs (see
        :func:`~cwinpy.peutils.optimal_snr`).
    det: str
        The detector for which to calculate the SNRs (see
        :func:`~cwinpy.peutils.optimal_snr`).
    fig: Figure
        A :class:`~matplotlib.figure.Figure` object on which to overplot the
        results.
    xaxis: str
        Set whether to plot the ``"odds"`` or ``"snr"`` on the x-axis. Defaults
        to ``"snr"``.
    scatterc: str
        Set whether to use the ``"odds"`` or ``"snr"`` to set the plot marker
        colour. Default is None.
    """

    if isinstance(S, (str, Path)) and isinstance(R, (str, Path)):
        # calculate SNRs and odds values
        try:
            snrs = optimal_snr(
                R,
                S,
                which=kwargs.pop("which", "posterior"),
                remove_outliers=kwargs.pop("remove_outliers", False),
                det=kwargs.pop("det", None),
            )
        except (ValueError, TypeError):
            raise ValueError("Could not calculate the SNRs")

        try:
            scale = kwargs.pop("scale", "log10")
            oddstype = kwargs.pop("oddstype", "svn")

            logodds = results_odds(
                R,
                scale=scale,
                oddstype=oddstype,
            )
        except (ValueError, TypeError, RuntimeError):
            raise ValueError("Could not calculate the odds")
    else:
        scale = kwargs.pop("scale", "log10")
        oddstype = kwargs.pop("oddstype", "svn")

        snrs = S
        logodds = R

    if not isinstance(snrs, dict) or not isinstance(logodds, dict):
        raise TypeError("Inputs must be dictionaries or paths")

    # check dictionaries contain consistent sources
    sset = set(list(snrs.keys()))
    rset = set(list(logodds.keys()))

    if not sset.issubset(rset) and not rset.issubset(sset):
        raise ValueError("SNRs and odds must contain consistent sources")

    pset = sset if sset.issubset(rset) else rset
    snrsp = [snrs[p] for p in pset]
    lop = [logodds[p] for p in pset]

    fig = kwargs.pop("fig", None)

    if fig is None:
        figkwargs = {}
        if "figsize" in kwargs:
            figkwargs["figsize"] = kwargs.pop("figsize")
        if "dpi" in kwargs:
            figkwargs["dpi"] = kwargs.pop("dpi")
        fig, ax = plt.subplots(**figkwargs)
    else:
        ax = fig.gca()

    # create plot
    scatterc = kwargs.pop("scatterc", None)
    if scatterc == "odds":
        scattercvals = snrsp
    elif scatterc == "snr":
        scattercvals = lop
    else:
        scattercvals = None

    oddsscalelabel = r"\log{}_{10}" if scale == "log10" else r"\ln{}"

    # x and y axes scales
    xscale = kwargs.pop("xscale", "linear")
    yscale = kwargs.pop("yscale", "linear")

    if kwargs.pop("xaxis", "snr") == "odds":
        xvals = lop
        yvals = snrsp

        xlabel = rf"${oddsscalelabel}\mathcal{{O}}_{{\rm {oddstype}}}$"
        ylabel = r"$\rho$"
    else:
        xvals = snrsp
        yvals = lop

        ylabel = rf"${oddsscalelabel}\mathcal{{O}}_{{\rm {oddstype}}}$"
        xlabel = r"$\rho$"

    ax.scatter(xvals, yvals, c=scattercvals, **kwargs)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    return fig
