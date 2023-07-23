from pathlib import Path

from bilby.core.result import Result

from ..data import HeterodynedData
from ..parfile import PulsarParameters
from ..plot import Plot
from ..utils import get_psr_name, is_par_file

# from .peutils import UpperLimitTable, optimal_snr, read_in_result_wrapper, results_odds


def pulsar_summary_plots(
    parfile,
    heterodyneddata=None,
    posteriordata=None,
    ulresultstable=None,
    oddstable=None,
    snrtable=None,
    outdir=None,
    outputsuffix=None,
    plotformat=".png",
    **plotkwargs,
):
    """
    Produce plots summarising the information from a pulsar analysis.

    Parameters
    ----------
    parfile: :class:`cwinpy.parfile.PulsarParameters`, str
        The pulsar parameters or a file containing pulsar the pulsar
        parameters.
    heterodyneddata: :class:`cwinpy.data.HeterodynedData`, str, dict
        A :class:`~cwinpy.data.HeterodynedData` object, path to a file
        containing heterodyned data, or a dictionary to multiple
        :class:`~cwinpy.data.HeterodynedData` objects (or file paths). If this
        is given, summary time series and spectrogram plots will be created.
        The dictionary keys will be treated as the data's names, e.g., detector
        names, and used as a suffix for the output plot file name if one is not
        given.
    posteriordata: :class:`bilby.core.results.Result`, str, dict
        A :class:`~bilby.core.result.Result` object, path to a results file, or
        dictionary containing multiple :class:`~bilby.core.result.Result`
        objects (or file paths). If this is given, posterior plots will be
        created. The dictionary keys will be treated as the data's names, e.g.,
        detector names, and used as a suffix for the output plot file name if
        one is not given.
    ulresultstable: UpperLimitTable
        A table of upper limits from which limits for the given pulsar can be
        extracted. If given, this will be included in a summary table for that
        pulsar.
    oddstable: dict
        A dictionary of odds results from which the value for the given pulsar
        can be extracted. If given, this will be included in a summary table for
        that pulsar.
    snrtable: dict
        A dictionary of SNR values from which the value for the given pulsar
        can be extracted. If given, this will be included in a summary table
        for that pulsar.
    outdir: str, Path
        The output directory into which to save the plots/summary table. If not
        given, the current working directory will be used.
    outputsuffix: str
        An suffix to append to the output files names. Default is None.
    plotformat: str
        The file format with which the save the figures. The default is ".png".

    Returns
    -------
    summaryfiles: dict
        A dictionary containing to paths to all the summary files.
    """

    if is_par_file(parfile):
        par = PulsarParameters(parfile)
    else:
        raise ValueError("Supplied pulsar .par file is invalid.")

    if outdir is None:
        outpath = Path.cwd()
    else:
        outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)

    pname = get_psr_name(par)

    summaryfiles = {}

    if heterodyneddata is not None:
        if isinstance(heterodyneddata, (str, Path, HeterodynedData)):
            if isinstance(heterodyneddata, HeterodynedData):
                het = heterodyneddata
            else:
                het = HeterodynedData.read(heterodyneddata)

            outsuf = "" if outputsuffix is None else f"_{outputsuffix}"

            # plot time series
            hetfig = het.plot(which="abs", remove_outliers=True)
            hetfig.tight_layout()
            filename = f"time_series_plot_{pname}_{outsuf}"
            hetfig.savefig(
                outpath / f"{filename}{plotformat}", dpi=plotkwargs.get("dpi", 150)
            )
            summaryfiles[filename] = outpath / f"{filename}{plotformat}"

            # plot spectrogram
            specfig = het.spectrogram(remove_outliers=True)
            specfig.tight_layout()
            filename = f"time_series_plot_{pname}_{outsuf}"
            specfig.savefig(
                outpath / f"{filename}{plotformat}", dpi=plotkwargs.get("dpi", 150)
            )
            summaryfiles[filename] = outpath / f"{filename}{plotformat}"
        elif isinstance(heterodyneddata, dict):
            for suf in heterodyneddata:
                if outputsuffix is None:
                    outsuf = suf
                else:
                    outsuf = f"{outputsuffix}_{suf}"

                sf = pulsar_summary_plots(
                    par,
                    heterodyneddata=heterodyneddata[suf],
                    outputsuffix=outsuf,
                    outdir=outdir,
                    plotformat=plotformat,
                    **plotkwargs,
                )

                summaryfiles.update(sf)
        else:
            raise TypeError("heterodyneddata is not the correct type.")

    if posteriordata is not None:
        if isinstance(posteriordata, (str, Path, Result)):
            postdata = posteriordata

            if outputsuffix is not None:
                postdata = {outputsuffix: postdata}

            # plot posteriors
            plot = Plot(postdata, plottype="corner")
            plot.plot()

            outsuf = "" if outputsuffix is None else f"_{outputsuffix}"
            filename = f"posteriors_{pname}_{outsuf}"
            plot.savefig(
                outpath / f"{filename}{plotformat}", dpi=plotkwargs.get("dpi", 150)
            )
            summaryfiles[filename] = outpath / f"{filename}{plotformat}"
        elif isinstance(posteriordata, dict):
            for suf in posteriordata:
                if outputsuffix is None:
                    outsuf = suf
                else:
                    outsuf = f"{outputsuffix}_{suf}"

                sf = pulsar_summary_plots(
                    par,
                    posteriordata=posteriordata[suf],
                    outputsuffix=outsuf,
                    outdir=outdir,
                    plotformat=plotformat,
                    **plotkwargs,
                )

                summaryfiles.update(sf)
        else:
            raise TypeError("posteriordata is not the correct type.")

    return summaryfiles
