import re
from copy import deepcopy
from itertools import permutations
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable
from bilby.core.result import Result, read_in_result
from matplotlib import pyplot as plt

from ..data import HeterodynedData, MultiHeterodynedData
from ..parfile import PulsarParameters
from ..utils import get_psr_name


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
    det: str
        If passing a directory to ``results`` and wanting the ``"svn"`` odds
        for a particular detector within that directory, then that can be
        specified.

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
            if oddstype == "svn" and "det" in kwargs:
                key = kwargs.pop("det")
                if key not in resultd:
                    raise KeyError(f"{key} not in {list(resultd.keys())}")
            else:
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
                    post.log_likelihood.idxmax()
                    if which.lower() == "likelihood"
                    else (post.log_likelihood + post.log_prior).idxmax()
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
                post.log_likelihood.idxmax()
                if which.lower() == "likelihood"
                else (post.log_likelihood + post.log_prior).idxmax()
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
                    pn = dname.replace("+", r"\+")
                    detmatch = re.search(f"{fnamestr}_(.*?)_{pn}", str(rf))
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
    ax: Axes
        A :class:`~matplotlib.axes.Axes` object on which to overplot the
        results.
    xaxis: str
        Set whether to plot the ``"odds"`` or ``"snr"`` on the x-axis. Defaults
        to ``"snr"``.
    scatterc: str
        Set whether to use the ``"odds"`` or ``"snr"`` to set the plot marker
        colour. Default is None.
    plotsources: str, list
        A name, or list of names, of the sources to include on the plot. If not
        give then all sources will be plotted.

    Returns
    -------
    fig:
        The :class:`~matplotlib.figure.Figure` object containing the plot.
    ax:
        The :class:`~matplotlib.axes.Axes` object containing the plot.
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

    # get sources to include
    psources = kwargs.pop("plotsources", [])
    if isinstance(psources, str):
        psources = [psources]

    pset = sset if sset.issubset(rset) else rset
    snrsp = [snrs[p] for p in pset if p in psources or len(psources) == 0]
    lop = [logodds[p] for p in pset if p in psources or len(psources) == 0]

    ax = kwargs.pop("ax", None)

    if ax is None:
        figkwargs = {}
        if "figsize" in kwargs:
            figkwargs["figsize"] = kwargs.pop("figsize")
        if "dpi" in kwargs:
            figkwargs["dpi"] = kwargs.pop("dpi")
        fig, ax = plt.subplots(**figkwargs)
    else:
        fig = plt.gcf()

    # create plot
    scatterc = kwargs.pop("scatterc", None)
    if scatterc == "odds":
        scattercvals = lop
    elif scatterc == "snr":
        scattercvals = snrsp
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

    return fig, ax


class UpperLimitTable(QTable):
    AMPPARAMS = {
        "h0": u.dimensionless_unscaled,
        "c21": u.dimensionless_unscaled,
        "c22": u.dimensionless_unscaled,
        "q22": u.kg * u.m**2,
    }

    def __init__(self, resdir=None, **kwargs):
        """
        Generate a table (as an :class:`astropy.table.QTable`) of upper limits
        on gravitational-wave amplitude parameters for a set of pulsars. This
        requires the path to a results directory which will be globbed for
        files that are the output of the CWInPy parameter estimation pipeline.
        It is assumed that the results files are in the format
        ``cwinpy_pe_{det}_{dname}_result.[hdf5,json]`` where ``{det}`` is the
        detector, or detector combination, prefix and ``{dname}`` is the
        pulsar's PSR J-name.

        From each of these, upper limits on the amplitude parameters will be
        produced. If requested, upper limits on ellipticity, mass quadrupole
        :math:`Q_{22}` and the ratio of the limit compared to the spin-down
        limit can be calculated. For these, a dictionary of distances/frequency
        derivatives cen be supplied, otherwise these will be extracted from the
        ATNF pulsar catalogue.

        Parameters
        ----------
        resdir: str
            The path to a directory containing results. This will be
            recursively globbed to find valid .hdf5/json results files. It is
            assumed that the results files contain the pulsars J-name, which
            will be used in the table.
        pulsars: list
            By default all pulsar results found in the ``resdir`` will be used.
            If wanting a subset of those pulsars, then a list of required
            pulsars can be given.
        detector: str:
            By default upper limits from results for all detectors found in the
            ``resdir`` will be used. If just wanted results from one detector
            it can be specified.
        ampparam: str
            The amplitude parameter which to include in the upper limit table,
            e.g., ``"h0"``. If not given, upper limits on all amplitude
            parameters in the results file will be included.
        upperlimit: float
            A fraction between 0 and 1 giving the credibility value for the
            upper limit. This defaults to 0.95, i.e., the 95% credible upper
            limit.
        includeell: bool
            Include the inferred ellipticity upper limit. This requires the
            pulsar distances, which can be supplied by the user or otherwise
            obtained from the "best" distance estimate given in the ATNF pulsar
            catalogue. If no distance estimate is available the table column
            will be left blank.
        includeq22: bool
            Include the inferred mass quadrupole upper limit. This requires the
            pulsar distances, which can be supplied by the user or otherwise
            obtained from the "best" distance estimate given in the ATNF pulsar
            catalogue. If no distance estimate is available the table column
            will be left blank.
        includesdlim: bool
            Include the ratio of the upper limit to the inferred spin-down
            limit. This requires the pulsar distances and frequency derivative,
            which can be supplied by the user or otherwise obtained from the
            ATNF pulsar catalogue. The intrinsic frequency derivative (i.e.,
            the value corrected for proper motion effects) will be
            preferentially used if avaiable. If no distance estimate or
            frequency derivative is available the table column will be left
            blank.
        distances: dict
            A dictionary of distances (in kpc) to use when calculating derived
            values. If a pulsar is not included in the dictionary, then the
            ATNF value will be used.
        fdot: dict
            A dictionary of frequency derivatives to use when calculating
            spin-down limits. If a pulsar is not included in the dictionary,
            then the ATNF value will be used.
        izz: float, Quantity
            The principal moment of inertia to use when calculating derived
            quantities. This defaults to :math:`10^{38}` kg m:sup:`2`. If
            passing a float it assumes the units are kg m:sup:`2`. If
            requiring different units then an :class:`astropy.units.Quantity`
            can be used.
        querykwargs: dict
            Any keyword arguments to pass to the QueryATNF object.
        """

        # get keyword args
        pulsars = kwargs.pop("pulsars", None)
        detector = kwargs.pop("detector", None)
        ampparam = kwargs.pop("ampparam", None)
        upperlimit = kwargs.pop("upperlimit", 0.95)
        distances = kwargs.pop("distances", {})
        f1s = kwargs.pop("fdot", {})
        incell = kwargs.pop("includeell", False)
        incq22 = kwargs.pop("includeq22", False)
        incsdlim = kwargs.pop("includesdlim", False)
        izz = kwargs.pop("izz", 1e38 * u.kg * u.m**2)
        querykwargs = kwargs.pop("querykwargs", {})

        if resdir is None:
            super().__init__(**kwargs)
            return

        resfiles = find_results_files(resdir)

        if len(resfiles) == 0:
            print(f"No results files were found in {resdir}")
            super().__init__(**kwargs)
            return

        if isinstance(pulsars, str):
            pulsars = [pulsars]
        if isinstance(pulsars, list):
            # remove pulsar not in the list
            for psr in list(resfiles.keys()):
                if psr not in pulsars:
                    resfiles.pop(psr)

            if len(resfiles) == 0:
                print(f"No results for the request pulsars were found in {resdir}")
                super().__init__(**kwargs)
                return

        pulsars = sorted(list(resfiles.keys()))

        if isinstance(detector, str):
            # remove detector data that is not required
            for psr in pulsars:
                dets = list(resfiles[psr].keys())
                if detector not in dets:
                    print(
                        f"No results for {detector} and PSR {psr} were found in {resdir}"
                    )
                    return

                for det in dets:
                    if det != detector:
                        resfiles[psr].pop(det)

        if isinstance(ampparam, str):
            if ampparam.lower() not in self.AMPPARAMS:
                raise ValueError(f"ampparams must be one of {self.AMPPARAMS}")

            useampparams = [ampparam]
        else:
            useampparams = list(self.AMPPARAMS.keys())

        try:
            if upperlimit <= 0.0 or upperlimit > 1.0:
                raise ValueError("Upper limit must be between 0 and 1")
        except TypeError:
            raise TypeError("Upper limit must be a float")

        resdict = {"PSRJ": pulsars}

        if incell or incq22 or incsdlim:
            from psrqpy import QueryATNF

            try:
                from cweqgen import equations
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    "You need to install cweqgen to do equation conversion"
                )

            psrq = QueryATNF(
                psrs=pulsars,
                params=["PSRJ", "F0", "P0", "P1_I", "F1", "DIST"],
                **querykwargs,
            )

            f0s = {}  # store frequencies
            pdottofdot = equations("rotationfdot_to_period")

            for psr in pulsars:
                psrrow = psrq[psr]

                if psr not in distances:
                    distances[psr] = psrrow["DIST"][0] * u.kpc

                f0s[psr] = psrrow["F0"][0] * u.Hz

                if psr not in f1s:
                    if np.isfinite(psrrow["P1_I"][0]):
                        # use intrinsic Pdot
                        f1s[psr] = pdottofdot(
                            rotationperiod=psrrow["P0"][0],
                            rotationpdot=psrrow["P1_I"][0],
                        )
                    else:
                        f1s[psr] = psrrow["F1"][0] * u.Hz / u.s

            if incell:
                elleq = equations("h0").rearrange("ellipticity")

            if incq22:
                q22eq = (
                    equations("h0")
                    .substitute(equations("massquadrupole").rearrange("ellipticity"))
                    .rearrange("massquadrupole")
                )

            if incsdlim:
                sdeq = equations("h0spindown")

            # sort in pulsar order
            f0s = dict(sorted(f0s.items()))
            f1s = dict(sorted(f1s.items()))
            distances = dict(sorted(distances.items()))

            # add in pulsar parameters
            resdict["F0ROT"] = list(f0s.values())
            resdict["F1ROT"] = list(f1s.values())
            resdict["DIST"] = list(distances.values())

        ulstr = f"_{int(100 * upperlimit)}%UL"

        # get amplitude upper limits
        for psr in pulsars:
            psrfiles = resfiles[psr]

            for j, det in enumerate(psrfiles):
                resdat = read_in_result_wrapper(psrfiles[det])
                post = resdat.posterior

                h0colname = None
                hasq22 = False
                for amppar in useampparams:
                    if amppar in post.columns:
                        colname = (
                            amppar + f"{ulstr}"
                            if len(psrfiles) == 1
                            else amppar + f"_{det}{ulstr}"
                        )

                        ul = np.quantile(post[amppar], upperlimit)
                        if colname not in resdict:
                            resdict[colname] = [ul]
                        else:
                            resdict[colname].append(ul)

                        if amppar == "h0":
                            h0colname = colname
                        elif amppar == "q22":
                            hasq22 = True

                # get derived parameters
                if h0colname is not None:
                    if incell:
                        colname = (
                            f"ELL{ulstr}" if len(psrfiles) == 1 else f"ELL_{det}{ulstr}"
                        )

                        ellul = (
                            elleq(
                                h0=resdict[h0colname][-1],
                                frot=f0s[psr],
                                distance=distances[psr],
                                izz=izz,
                            )
                            if psr in distances
                            else np.nan
                        )

                        if colname not in resdict:
                            resdict[colname] = [ellul]
                        else:
                            resdict[colname].append(ellul)

                    if incq22 and not hasq22:
                        colname = (
                            f"Q22{ulstr}" if len(psrfiles) == 1 else f"Q22_{det}{ulstr}"
                        )

                        q22ul = (
                            q22eq(
                                h0=resdict[h0colname][-1],
                                frot=f0s[psr],
                                distance=distances[psr],
                            )
                            if psr in distances
                            else np.nan
                        )

                        if colname not in resdict:
                            resdict[colname] = [q22ul]
                        else:
                            resdict[colname].append(q22ul)

                    if incsdlim:
                        colname = (
                            f"SDRAT{ulstr}"
                            if len(psrfiles) == 1
                            else f"SDRAT_{det}{ulstr}"
                        )

                        h0sd = np.nan
                        if psr in f1s and psr in distances:
                            if f1s[psr] < 0.0:
                                h0sd = sdeq(
                                    rotationfdot=f1s[psr],
                                    rotationfrequency=f0s[psr],
                                    distance=distances[psr],
                                    izz=izz,
                                )

                        if colname not in resdict:
                            resdict[colname] = [resdict[h0colname][-1] / h0sd]
                        else:
                            resdict[colname].append(resdict[h0colname][-1] / h0sd)

                        if "SDLIM" not in resdict:
                            # add in spin-down limit
                            resdict["SDLIM"] = [h0sd]
                        elif j == 0:
                            resdict["SDLIM"].append(h0sd)

        super().__init__(resdict, **kwargs)
