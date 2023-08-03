from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import numpy as np
from bilby.core.result import Result
from matplotlib import pyplot as plt
from pesummary.core.webpage.webpage import BOOTSTRAP, OTHER_SCRIPTS, page
from scipy.stats import hmean

from ..data import HeterodynedData
from ..parfile import PulsarParameters
from ..plot import Plot
from ..utils import get_psr_name, is_par_file
from .pe import pe_pipeline
from .peutils import (  # , optimal_snr, read_in_result_wrapper, results_odds
    UpperLimitTable,
    set_formats,
)

# add MathJAX to HOME_SCRIPTS and OTHER_SCRIPTS
SCRIPTS_AND_CSS = f"""   <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js"></script>
{OTHER_SCRIPTS}"""


def make_html(
    web_dir: Union[str, Path],
    label: str,
    suffix: str = None,
    title: str = "Summary Pages",
):
    """
    Make the initial html page. Adapted from pesummary.

    Parameters
    ----------
    web_dir: str, Path
        Path to the location where you would like the html file to be saved.
    label: str
        Label used to create page name.
    suffix: str
        Suffix to page name
    title: str, optional
        Header title of html page.
    """

    if suffix is None:
        pagename = f"{label}.html"
    else:
        pagename = f"{label}_{suffix}.html"

    htmlfile = Path(web_dir) / "html" / pagename
    with open(htmlfile, "w") as f:
        bootstrap = BOOTSTRAP.split("\n")
        bootstrap[1] = "  <title>{}</title>".format(title)
        bootstrap = [j + "\n" for j in bootstrap]
        f.writelines(bootstrap)
        scripts = SCRIPTS_AND_CSS.split("\n")
        scripts = [j + "\n" for j in scripts]
        f.writelines(scripts)

    return htmlfile


def open_html(web_dir, base_url, html_page, label):
    """
    Open html page ready so you can manipulate the contents. Adapted from
    pesummary.

    Parameters
    ----------
    web_dir: str
        path to the location where you would like the html file to be saved
    base_url: str
        url to the location where you would like the html file to be saved
    page: str
        name of the html page that you would like to edit
    """
    try:
        if html_page[-5:] == ".html":
            html_page = html_page[:-5]
    except Exception:
        pass

    htmlfile = Path(web_dir) / f"{html_page}.html"
    f = open(htmlfile, "a")

    return page(f, web_dir, base_url, label)


PULSAR_HEADER_FORMATS = {
    "F0": {
        "html": r"\(f_{\rm rot}\) (Hz)",
        "ultablename": "F0ROT",
        "formatter": set_formats(name="F0ROT", type="html", dp=2),
    },
    "2F0": {
        "html": r"\(f_{\rm gw}\,[2f_{\rm rot}]\) (Hz)",
        "ultablename": "F0ROT",
        "formatter": lambda x: set_formats(name="F0ROT", type="html", dp=2)(2 * x),
    },
    "F1": {
        "html": r"\(\dot{f}_{\rm rot}\) (Hz/s)",
        "ultablename": "F1ROT",
        "formatter": set_formats(name="F1ROT", type="html", dp=2, scinot=True),
    },
    "DIST": {
        "html": "distance (kpc)",
        "ultablename": "DIST",
        "formatter": set_formats(name="DIST", type="html", dp=1),
    },
    "SDLIM": {
        "html": "\(h_0\) spin-down limit",
        "ultablename": "SDLIM",
        "formatter": set_formats(name="SDLIM", type="html", dp=1, scinot=True),
    },
}

RESULTS_HEADER_FORMATS = {
    "H0": {
        "html": r"\(h_0^{95\%}\) upper limit",
        "ultablename": "H0_{}_95%UL",
        "formatter": set_formats(name="H0", type="html", dp=1, scinot=True),
    },
    "ELL": {
        "html": r"\(\varepsilon^{95\%}\) upper limit",
        "ultablename": "ELL_{}_95%UL",
        "formatter": set_formats(name="ELL", type="html", dp=1, scinot=True),
    },
    "Q22": {
        "html": r"\(Q_{22}^{95\%}\) upper limit (kg m<sup>2</sup>)",
        "ultablename": "Q22_{}_95%UL",
        "formatter": set_formats(name="Q22", type="html", dp=1, scinot=True),
    },
    "SDRAT": {
        "html": r"\(h_0^{95\%}\,/\,h_0^{\rm spin-down}\)",
        "ultablename": "SDRAT_{}_95%UL",
        "formatter": set_formats(name="SDRAT", type="html"),
    },
    "C21": {
        "html": r"\(C_{21}^{95\%}\) upper limit",
        "ultablename": "C21_{}_95%UL",
        "formatter": set_formats(name="C21", type="html", dp=1, scinot=True),
    },
    "C22": {
        "html": r"\(C_{22}^{95\%}\) upper limit",
        "ultablename": "C22_{}_95%UL",
        "formatter": set_formats(name="C22", type="html", dp=1, scinot=True),
    },
}


def pulsar_summary_plots(
    parfile: Union[str, Path, PulsarParameters],
    heterodyneddata: Union[str, dict, HeterodynedData, Path] = None,
    posteriordata: Union[str, dict, Path, Result] = None,
    ulresultstable: UpperLimitTable = None,
    oddstable: dict = None,
    snrtable: dict = None,
    outdir: Union[str, Path] = None,
    outputsuffix: str = None,
    plotformat: str = ".png",
    showindividualparams: bool = False,
    webpage: page = None,
    **kwargs,
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
    showindividualparams: bool
        Set to true to produce posterior plots for all individual parameters as
        well as the joint posterior plot. Default is False.
    webpage: :class:`pesummary.core.webpage.webpage.page`, dict
        A :class:`~pesummary.core.webpage.webpage.page` onto which to add the
        plots/tables or a dictionary of pages, where the dictionary keys will
        be treated as the detector names.

    Returns
    -------
    summaryfiles: dict
        A dictionary containing to paths to all the summary files.
    """

    if is_par_file(parfile):
        par = PulsarParameters(parfile)
    elif isinstance(parfile, PulsarParameters):
        par = parfile
    else:
        raise ValueError(f"Supplied pulsar .par file '{parfile}' is invalid.")

    if outdir is None:
        outpath = Path.cwd()
    else:
        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)

    pname = get_psr_name(par)

    summaryfiles = {}

    if isinstance(ulresultstable, UpperLimitTable):
        if isinstance(webpage, page) and kwargs.get("det", None) is not None:
            tloc = ulresultstable.loc[pname]

            # create table of results for the pulsar
            det = kwargs["det"]  # get detector if passed as kwarg

            # construct table of pulsar information and results
            header = [f"PSR {pname}"]

            psrtable = []
            for key in PULSAR_HEADER_FORMATS:
                tname = PULSAR_HEADER_FORMATS[key]["ultablename"]
                if tname in ulresultstable.columns:
                    psrtable.append(
                        [
                            PULSAR_HEADER_FORMATS[key]["html"],
                            PULSAR_HEADER_FORMATS[key]["formatter"](tloc[tname].value),
                        ]
                    )

            webpage.make_div()  # div to contain tables
            webpage.make_div(_style="float:left;width:50%")  # div for first table
            webpage.make_table(
                headings=header,
                heading_span=2,
                accordian=False,
                contents=psrtable,
                colors=["#ffffff"],
            )
            webpage.end_div()

            resheader = ["Results"]
            restable = []
            for key in RESULTS_HEADER_FORMATS:
                tname = RESULTS_HEADER_FORMATS[key]["ultablename"].format(det)
                if tname in ulresultstable.columns:
                    restable.append(
                        [
                            RESULTS_HEADER_FORMATS[key]["html"],
                            RESULTS_HEADER_FORMATS[key]["formatter"](
                                tloc[tname].value
                                if hasattr(tloc[tname], "value")
                                else tloc[tname]
                            ),
                        ]
                    )

            webpage.make_div(_style="float:left;width:50%")  # div for second table
            webpage.make_table(
                headings=resheader,
                heading_span=2,
                accordian=False,
                contents=restable,
                colors=["#ffffff"],
            )
            webpage.end_div()
            webpage.end_div()
        elif isinstance(webpage, dict):
            for det in webpage:
                pulsar_summary_plots(
                    parfile,
                    ulresultstable=ulresultstable,
                    webpage=webpage[det],
                    det=det,
                )

    if heterodyneddata is not None:
        if isinstance(heterodyneddata, (str, Path, HeterodynedData)):
            if isinstance(heterodyneddata, HeterodynedData):
                het = heterodyneddata
            else:
                het = HeterodynedData.read(heterodyneddata)

            outsuf = "" if outputsuffix is None else f"{outputsuffix}"

            # plot time series
            hetfig = het.plot(which="abs", remove_outliers=True)
            hetfig.tight_layout()
            filename = f"time_series_plot_{pname}_{outsuf}"
            hetfig.savefig(
                outpath / f"{filename}{plotformat}", dpi=kwargs.get("dpi", 150)
            )
            summaryfiles[filename] = outpath / f"{filename}{plotformat}"
            plt.close()

            # plot spectrogram
            specfig = het.spectrogram(remove_outliers=True)
            filename = f"spectrogram_plot_{pname}_{outsuf}"
            specfig[-1].savefig(
                outpath / f"{filename}{plotformat}", dpi=kwargs.get("dpi", 150)
            )
            summaryfiles[filename] = outpath / f"{filename}{plotformat}"
            plt.close()

            # plot spectrum
            sfig = het.power_spectrum(remove_outliers=True, asd=True)
            filename = f"asd_plot_{pname}_{outsuf}"
            sfig[-1].savefig(
                outpath / f"{filename}{plotformat}", dpi=kwargs.get("dpi", 150)
            )
            summaryfiles[filename] = outpath / f"{filename}{plotformat}"
            plt.close()
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
                    **kwargs,
                )

                summaryfiles.update(sf)
        else:
            raise TypeError("heterodyneddata is not the correct type.")

    if posteriordata is not None:
        if isinstance(posteriordata, (str, Path, Result)):
            postdata = posteriordata

            if outputsuffix is not None:
                postdata = {outputsuffix: postdata}

            # copy of plotting kwargs
            tplotkwargs = kwargs.copy()

            # get output dpi
            dpi = tplotkwargs.pop("dpi", 150)

            # set default number of histogram bins for plots
            if "bins" not in tplotkwargs:
                tplotkwargs["bins"] = 30

            # plot posteriors for all parameters
            plot = Plot(postdata, plottype="corner")
            plot.plot(**tplotkwargs)

            outsuf = "" if outputsuffix is None else f"{outputsuffix}"
            filename = f"posteriors_{pname}_{outsuf}"
            plot.savefig(outpath / f"{filename}{plotformat}", dpi=dpi)
            summaryfiles[filename] = outpath / f"{filename}{plotformat}"
            plt.close()

            # plot individual parameter marginal posteriors if requested
            if showindividualparams:
                params = plot.parameters  # get all parameter names

                for param in params:
                    plot = Plot(postdata, parameters=param, plottype="hist", kde=True)
                    plot.plot(hist_kwargs={"bins": tplotkwargs["bins"]})

                    filename = f"posteriors_{pname}_{param}_{outsuf}"
                    plot.savefig(outpath / f"{filename}{plotformat}", dpi=dpi)
                    summaryfiles[filename] = outpath / f"{filename}{plotformat}"
                    plt.close()
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
                    showindividualparams=showindividualparams,
                    **kwargs,
                )

                summaryfiles.update(sf)
        else:
            raise TypeError("posteriordata is not the correct type.")

    return summaryfiles


def generate_summary_pages(**kwargs):
    """
    Generate summary webpages following a ``cwinpy_knope_pipeline`` analysis
    (see :func:`~cwinpy.knope.knope_pipeline`).

    Parameters
    ----------
    config: str, Path
        The configuration file used for the ``cwinpy_knope_pipeline`` analysis.
    outpath: str, Path
        The output path for the summary results webpages and plots.
    url: str
        The URL from which the summary results pages will be accessed.
    showposteriors: bool
        Set to enable/disable production of plots showing the joint posteriors
        for all parameters. The default is True.
    showindividualposteriors: bool
        Set to enable/disable production of plots showing the marginal
        posteriors for each individual parameter. The default is False.
    showtimeseries: bool
        Set to enable/disable production of plots showing the heterodyned time
        series data (and spectral representations). The default is True.
    pulsars: list, str
        A list of pulsars to show. By default all pulsars analysed will be
        shown.
    upperlimittable: bool
        Set to enable/disable production of a table of amplitude upper limits.
        The default is True.
    upperlimitplot: bool
        Set to enable/disable production of a plot of amplitude upper limits
        as a function of frequency. The default is True.
    """

    if "cli" not in kwargs:
        configfile = kwargs.pop("config")
        outpath = Path(kwargs.pop("outpath"))
        url = kwargs.pop("url")

        showposteriors = kwargs.pop("showposteriors", True)
        showindividualparams = kwargs.pop("showindividualposteriors", False)
        showtimeseries = kwargs.pop("showtimeseries", True)

        pulsars = kwargs.pop("pulsar", None)
        if isinstance(pulsars, str):
            pulsars = [pulsars]

        upperlimittable = kwargs.pop("upperlimittable", True)
        # upperlimitplot = kwargs.pop("upperlimitplot", True)
    else:  # pragma: no cover
        parser = ArgumentParser(
            description=(
                "A script to create results summary pages from a "
                "cwinpy_knope_pipeline analysis."
            )
        )
        parser.add_argument(
            "config",
            help=("The configuration file from the cwinpy_knope_pipeline analysis."),
        )
        parser.add_argument(
            "--outpath",
            "-o",
            help=("The output path for the summary results webpages and plots."),
            required=True,
        )
        parser.add_argument(
            "--url",
            "-u",
            help=("The URL from which the summary results pages will be accessed."),
            required=True,
        )
        parser.add_argument(
            "--pulsars",
            "-p",
            nargs="+",
            help=(
                "Provide the pulsars for which to produces summary results. "
                "By default, all pulsars from the analysis will be used."
            ),
        )
        parser.add_argument(
            "--disable-posteriors",
            action="store_true",
            default=False,
            help="Set this flag to disable production of posterior plots.",
        )
        parser.add_argument(
            "--enable-individual-posteriors",
            action="store_true",
            default=False,
            help=(
                "Set this flag to enable to produciton of marginal posterior "
                "plots for each individual parameters."
            ),
        )
        parser.add_argument(
            "--disable-timeseries",
            action="store_true",
            default=False,
            help="Set this flag to disable production of time series plots.",
        )
        parser.add_argument(
            "--disable-upper-limit-table",
            action="store_true",
            default=False,
            help=(
                "Set this flag to disable production of a table of amplitude "
                "upper limits."
            ),
        )
        parser.add_argument(
            "--disable-upper-limit-plot",
            action="store_true",
            default=False,
            help=(
                "Set this flag to disable production of a plot of amplitude "
                "upper limits as a function of frequency."
            ),
        )

        args = parser.parse_args()
        configfile = args.config
        outpath = Path(args.outpath)
        url = args.url

        showposteriors = not args.disable_posteriors
        showtimeseries = not args.disable_timeseries
        showindividualparams = args.enable_individual_posteriors

        pulsars = args.pulsars

        upperlimittable = not args.disable_upper_limit_table
        # upperlimitplot = not args.disable_upper_limit_plot

    # make the output directory
    outpath.mkdir(parents=True, exist_ok=True)

    # extract run information from configuration file
    pipeline_data = pe_pipeline(config=configfile, build=False)

    if upperlimittable:
        # try and get base directory for results:
        ultable = UpperLimitTable(
            resdir=pipeline_data.resultsbase,
            includesdlim=True,
            includeell=True,
            includeq22=True,
        )

        # if upperlimitplot:
        # get power spectral densities
        # psds = generate_power_spectrum(pipeline_data.datadict)
    else:
        ultable = None
        # psds = None

    # plot posteriors
    if showposteriors:
        posteriorplots = {}

        if not pipeline_data.resultsfiles:
            raise ValueError("No results files given in pipeline configuration!")

        posteriorplotdir = outpath / "posterior_plots"

        for psr in pipeline_data.resultsfiles:
            if pulsars is not None and psr not in pulsars:
                continue

            posteriorplots[psr] = pulsar_summary_plots(
                pipeline_data.pulsardict[psr],
                posteriordata=pipeline_data.resultsfiles[psr],
                outdir=posteriorplotdir / psr,
                ulresultstable=ultable,
                showindividualparams=showindividualparams,
            )

        if not posteriorplots:
            raise ValueError(
                "None of the specified pulsars were found in the analysis."
            )

    if showtimeseries:
        timeseriesplots = {}

        if not pipeline_data.datadict:
            raise ValueError(
                "No heterodyned data files given in pipeline configuration!"
            )

        timeseriesplotdir = outpath / "timeseries_plots"

        for psr in pipeline_data.datadict:
            if pulsars is not None and psr not in pulsars:
                continue

            timeseriesplots[psr] = {}
            for freqfactor in pipeline_data.datadict[psr]:
                timeseriesplots[psr][freqfactor] = pulsar_summary_plots(
                    pipeline_data.pulsardict[psr],
                    heterodyneddata=pipeline_data.datadict[psr][freqfactor],
                    outdir=timeseriesplotdir / psr / freqfactor,
                )

        if not timeseriesplots:
            raise ValueError(
                "None of the specified pulsars were found in the analysis."
            )

    # html table showing all results
    allresultstable = []
    htmldir = outpath / "html"
    htmldir.mkdir(parents=True, exist_ok=True)

    # generate pages for each pulsar
    for psr in pipeline_data.resultsfiles:
        if pulsars is not None and psr not in pulsars:
            continue

        # row containing this pulsar's results
        thispulsarresults = []

        # create webpage
        dets = list(pipeline_data.resultsfiles[psr].keys())

        pages = {}
        links = ["Detectors", [{det: psr.replace("+", "%2B") for det in dets}]]
        for det in dets:
            # make the initial page
            htmlpage = make_html(
                outpath, psr.replace("+", "%2B"), det, title=f"PSR {psr} ({det})"
            )
            purl = f"{url}/html/{psr.replace('+', '%2B')}_{det}.html"
            pages[det] = open_html(
                det,
                purl,
                htmldir / htmlpage.stem,
                label=f"{psr.replace('+', '%2B')}_{det}",
            )
            pages[det].make_navbar(links)

        # add results tables to each page
        _ = pulsar_summary_plots(
            pipeline_data.pulsardict[psr],
            ulresultstable=ultable,
            webpage=pages,
        )

        # pulsar name with link (to final detector)
        thispulsarresults.append(
            f'<a href="../html/{psr.replace("+", "%2B")}_{det}">{psr}</a>'
        )

        allresultstable.append(thispulsarresults)

    return ultable


def generate_power_spectrum(
    heterodyneddata: dict,
    time_average: str = "median",
    freq_average: str = "median",
    asd: bool = False,
) -> dict:
    """
    Extract the power spectral density at the frequencies of a set of analysed
    pulsars using their heterodyned data products.

    Parameters
    ----------
    heterodyneddata: dict
        The input should be a dictionary as provided by the
        :attr:`~cwinpy.pe.pe.PEDAGRunner.datadict`, i.e., a dictionary key by
        pulsar name, then heterodyned data frequency factor ("1f" or "2f"),
        then a detector name, pointing to a path to some
        :class:`~cwinpy.data.HeterodynedData`.
    time_average: str
        The method of averaging each pulsar's spectrogram over time. The
        default is "median", although "harmonic_mean" may be useful. Allowed
        values are: "median", "mean", "harmonic_mean", "max" or "min".
    freq_average: str
        The method of averaging each pulsar's spectrogram over frequency. The
        default is "median", although "harmonic_mean" may be useful. Allowed
        values are: "median", "mean", "harmonic_mean", "max" or "min".
    asd: bool
        If set to True, output the amplitude spectral density values rather
        than power spectral density values.

    Returns
    -------
    specs: dict
        A dictionary keyed on the detector names, followed by frequency factors
        ("1f" or "2f") with values being 2D arrays containing the frequency and
        power spectral density (from using a median over the frequency band of
        the heterodyned data).
    """

    if not isinstance(heterodyneddata, dict):
        raise TypeError("heterodyneddata must be a dictionary")

    if freq_average.lower() not in [
        "median",
        "mean",
        "harmonic_mean",
        "hmean",
        "max",
        "min",
    ]:
        raise ValueError(
            'freq_average must be one of "median", "mean", "harmonic_mean", "max" or "min"'
        )

    # frequency averaging functions
    favfunc = {
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
        "median": np.median,
        "hmean": hmean,
        "harmonic_mean": hmean,
    }

    spec = {}
    freqs = {}
    for psr in heterodyneddata:
        if not isinstance(heterodyneddata[psr], dict):
            raise ValueError("heterodyneddata values must contain dictionaries")

        for ff in heterodyneddata[psr]:
            # check frequency factors of 1f or 2f
            if ff not in ["1f", "2f"]:
                raise KeyError("key must be either 1f or 2f")

            for det in heterodyneddata[psr][ff]:
                if isinstance(
                    heterodyneddata[psr][ff][det], (str, Path, HeterodynedData)
                ):
                    if isinstance(heterodyneddata[psr][ff][det], HeterodynedData):
                        het = heterodyneddata
                    else:
                        het = HeterodynedData.read(heterodyneddata[psr][ff][det])
                else:
                    raise ValueError("data is not a HeterodynedData object/path")

                if det not in freqs:
                    freqs[det] = {}
                    spec[det] = {}

                if ff not in freqs[det]:
                    freqs[det][ff] = []
                    spec[det][ff] = []

                # get power spectral densities
                _, power = het.power_spectrum(
                    remove_outliers=True,
                    plot=False,
                    average=time_average,
                    asd=asd,
                )
                spec[det][ff].append(favfunc[freq_average.lower()](power))

                # get the frequency
                freqs[det][ff].append(het.par["F0"] * int(ff[0]))

    # convert to arrays and sort by frequency
    specs = {}
    for det in freqs:
        specs[det] = {}
        for ff in freqs[det]:
            specs[det][ff] = np.array(sorted(zip(freqs[det][ff], spec[det][ff])))

    return specs


def generate_summary_pages_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_generate_summary_pages`` script. This just calls
    :func:`~cwinpy.pe.summary.generate_summary_pages`.
    """

    kwargs["cli"] = True
    generate_summary_pages(**kwargs)
