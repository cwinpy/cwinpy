import ast
import configparser
import os
import pathlib
from argparse import ArgumentParser

import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt

from ..heterodyne.heterodyne import HeterodyneDAGRunner
from ..info import (
    ANALYSIS_SEGMENTS,
    CVMFS_GWOSC_DATA_SERVER,
    CVMFS_GWOSC_DATA_TYPES,
    CVMFS_GWOSC_FRAME_CHANNELS,
    HW_INJ_RUNTIMES,
    HW_INJ_SEGMENTS,
    RUNTIMES,
    is_hwinj,
)
from ..parfile import PulsarParameters
from ..pe.pe import PEDAGRunner
from ..pe.peutils import results_odds
from ..utils import get_psr_name, is_par_file


def skyshift_pipeline(**kwargs):
    """
    Run skyshift pipeline within Python. This will create a
    `HTCondor <https://research.cs.wisc.edu/htcondor/>`_ DAG for consecutively
    running a ``cwinpy_heterodyne`` and multiple ``cwinpy_pe`` instances on a
    computer cluster. Optional parameters that can be used instead of a
    configuration file (for "quick setup") are given in the "Other parameters"
    section.

    Parameters
    ----------
    config: str
        A configuration file, or :class:`configparser:ConfigParser` object,
        for the analysis.
    nshifts: int
        The number of random sky-shifts to perform. The default will be 1000.
        These will be drawn from the same ecliptic hemisphere as the source.
    exclusion: float
        The exclusion region around the source's actual sky location and any
        sky shift locations.
    overlap: float
        Provide a maximum allowed fractional overlap (e.g., 0.01 for a maximum
        1% overlap) between the signal model at the true position and any of
        the sky-shifted position. If not given, this check will not be
        performed. If given, this check will be used in addition to the
        exclusion region check. Note: this does not check the overlap between
        each sky-shifted position, so some correlated sky-shifts may be
        present.

    Other parameters
    ----------------
    run: str
        The name of an observing run for which open data exists, which will be
        heterodyned, e.g., "O1".
    detector: str, list
        The detector, or list of detectors, for which the data will be
        heterodyned. If not set then all detectors available for a given run
        will be used.
    samplerate: str:
        Select the sample rate of the data to use. This can either be 4k or
        16k for data sampled at 4096 or 16384 Hz, respectively. The default
        is 4k, except if running on hardware injections for O1 or later, for
        which 16k will be used due to being required for the highest frequency
        source. For the S5 and S6 runs only 4k data is available from GWOSC,
        so if 16k is chosen it will be ignored.
    pulsar: str
        The path to a TEMPO(2)-style pulsar parameter file to heterodyne. If a
        pulsar name is given instead of a parameter file then an attempt will
        be made to find the pulsar's ephemeris from the ATNF pulsar catalogue,
        which will then be used.
    osg: bool
        Set this to True to run on the Open Science Grid rather than a local
        computer cluster.
    output: str
        The base location for outputting the heterodyned data and parameter
        estimation results. By default the current directory will be used.
        Within this directory, a subdirectory called "skyshift" will be created
        with subdirectories for each detector and for the result also created.
    joblength: int
        The length of data (in seconds) into which to split the individual
        heterodyne jobs. By default this is set to 86400, i.e., one day. If
        this is set to 0, then the whole dataset is treated as a single job.
    accounting_group_tag: str
        For LVK users this sets the computing accounting group tag.
    usetempo2: bool
        Set this flag to use Tempo2 (if installed) for calculating the signal
        phase evolution for the heterodyne rather than the default LALSuite
        functions.
    incoherent: bool
        If running with multiple detectors, set this flag to analyse each of
        them independently rather than coherently combining the data from all
        detectors. A coherent analysis and an incoherent analysis is run by
        default.

    Returns
    -------
    dag:
        An object containing a pycondor :class:`pycondor.Dagman` object.
    """

    if "config" in kwargs:
        hetconfigfile = kwargs.pop("config")
        peconfigfile = hetconfigfile
        nshifts = kwargs.pop("nshifts", 1000)
        exclusion = kwargs.pop("exclusion", 0.1)
        pulsar = kwargs.pop("pulsar", None)
        overlapfrac = kwargs.pop("overlap", None)
    else:  # pragma: no cover
        parser = ArgumentParser(
            description=(
                "A script to create a HTCondor DAG to process GW strain data "
                "by heterodyning it based on the expected phase evolution for "
                "a pulsar, and also a number of sky shifted locations, and "
                "then perform parameter estimation for the unknown signal for "
                "each of those locations."
            )
        )
        parser.add_argument(
            "config",
            nargs="?",
            help=("The configuration file for the analysis"),
            default=None,
        )

        skyshift = parser.add_argument_group("Sky shift arguments")
        skyshift.add_argument(
            "--nshifts",
            type=int,
            default=1000,
            required=True,
            help=(
                "The number of random sky-shifts to perform. The default will "
                "be %(default)s."
            ),
        )
        skyshift.add_argument(
            "--exclusion",
            help=(
                "The exclusion region around the source's actual sky location "
                "and any sky shift locations. The default is %(default)s "
                "radians."
            ),
            default=0.01,
            type=float,
        )
        skyshift.add_argument(
            "--check-overlap",
            help=(
                "Provide a maximum allowed fractional overlap (e.g., 0.01 for "
                "a maximum 1%% overlap) between the signal model at the true "
                "position and any of the sky-shifted position. If not given, "
                "this check will not be performed. If given, this check will "
                "be used in addition to the exclusion region check. Note: "
                "this does not check the overlap between each sky-shifted "
                "position, so some correlated sky-shifts may be present."
            ),
            type=float,
            dest="overlap",
        )

        optional = parser.add_argument_group(
            "Quick setup arguments (this assumes CVMFS open data access)."
        )
        optional.add_argument(
            "--run",
            help=(
                "Set an observing run name for which to heterodyne the data. "
                "This can be one of {} for which open data exists".format(
                    ", ".join(list(RUNTIMES.keys()))
                )
            ),
        )
        optional.add_argument(
            "--detector",
            action="append",
            help=(
                "The detector for which the data will be heterodyned. This "
                "can be used multiple times to specify multiple detectors. If "
                "not set then all detectors available for a given run will be "
                "used."
            ),
        )
        optional.add_argument(
            "--samplerate",
            help=(
                "Select the sample rate of the data to use. This can either "
                "be 4k or 16k for data sampled at 4096 or 16384 Hz, "
                "respectively. The default is %(default)s. For the S5 and S6 "
                "runs only 4k data is available from GWOSC, so if 16k is "
                "chosen it will be ignored."
            ),
            default="4k",
        )
        optional.add_argument(
            "--pulsar",
            help=("The path to a TEMPO(2)-style pulsar parameter file to heterodyne."),
            default=None,
        )
        optional.add_argument(
            "--osg",
            action="store_true",
            help=(
                "Set this flag to run on the Open Science Grid rather than a "
                "local computer cluster."
            ),
        )
        optional.add_argument(
            "--output",
            help=(
                "The base location for outputting the heterodyned data and "
                "parameter estimation results. By default the current "
                "directory will be used. Within this directory, a "
                'subdirectory called "skyshift" will be created with '
                "subdirectories for each detector and for the results also "
                "created."
            ),
            default=os.getcwd(),
        )
        optional.add_argument(
            "--joblength",
            type=int,
            help=(
                "The length of data (in seconds) into which to split the "
                "individual analysis jobs. By default this is set to 86400, "
                "i.e., one day. If this is set to 0, then the whole dataset "
                "is treated as a single job."
            ),
        )
        optional.add_argument(
            "--incoherent",
            action="store_true",
            help=(
                "If running with multiple detectors, set this flag to analyse "
                "each of them independently rather than coherently combining "
                "the data from all detectors. A coherent analysis and an "
                "incoherent analysis is run by default."
            ),
        )
        optional.add_argument(
            "--accounting-group-tag",
            dest="accgroup",
            help=("For LVK users this sets the computing accounting group tag"),
        )
        optional.add_argument(
            "--usetempo2",
            action="store_true",
            help=(
                "Set this flag to use Tempo2 (if installed) for calculating "
                "the signal phase evolution for the heterodyne rather than "
                "the default LALSuite functions."
            ),
        )

        args = parser.parse_args()
        if args.config is not None:
            pulsar = kwargs.get("pulsar", args.pulsar)

            hetconfigfile = args.config
            peconfigfile = args.config
        else:
            # use the "Quick setup" arguments
            hetconfigfile = configparser.ConfigParser()
            peconfigfile = configparser.ConfigParser()

            run = kwargs.get("run", args.run)
            if run not in RUNTIMES:
                raise ValueError(f"Requested run '{run}' is not available")

            # use pulsars provided
            runtimes = RUNTIMES
            segments = ANALYSIS_SEGMENTS

            # get sample rate
            srate = "16k" if (args.samplerate[0:2] == "16" and run[0] == "O") else "4k"

            pulsar = kwargs.get("pulsar", args.pulsar)
            hwinj = False
            if pulsar is None:
                raise ValueError("No pulsar parameter file has be provided")
            elif is_hwinj(pulsar):
                # using hardware injection
                runtimes = HW_INJ_RUNTIMES
                segments = HW_INJ_SEGMENTS
                srate = "16k" if run[0] == "O" else "4k"
                hwinj = True  # this is a hardware injection
                pulsar = is_hwinj(pulsar, return_file=True)

            detector = kwargs.get("detector", args.detector)
            if detector is None:
                detectors = list(runtimes[run].keys())
            else:
                detector = detector if isinstance(detector, list) else [detector]
                detectors = [det for det in args.detector if det in runtimes[run]]
                if len(detectors) == 0:
                    raise ValueError(
                        f"Provided detectors '{detector}' are not valid for the given run"
                    )

            # get run directory
            output = os.path.abspath(
                os.path.join(kwargs.get("output", args.output), "skyshift")
            )
            hetconfigfile["run"] = {}
            hetconfigfile["run"]["basedir"] = output

            # create required settings
            hetconfigfile["heterodyne_dag"] = {}
            peconfigfile["pe_dag"] = {}
            peconfigfile["pe_dag"]["submitdag"] = "True"  # submit automatically
            if kwargs.get("osg", args.osg):
                hetconfigfile["heterodyne_dag"]["osg"] = "True"
                hetconfigfile["pe_dag"]["osg"] = "True"

            hetconfigfile["heterodyne_job"] = {}
            hetconfigfile["heterodyne_job"]["getenv"] = "True"
            peconfigfile["pe_job"] = {}
            peconfigfile["pe_job"]["getenv"] = "True"
            if args.accgroup is not None:
                hetconfigfile["heterodyne_job"]["accounting_group"] = kwargs.get(
                    "accounting_group_tag", args.accgroup
                )
                peconfigfile["pe_job"]["accounting_group"] = kwargs.get(
                    "accounting_group_tag", args.accgroup
                )

            # add heterodyne settings
            hetconfigfile["heterodyne"] = {}
            hetconfigfile["heterodyne"]["detectors"] = str(detectors)
            hetconfigfile["heterodyne"]["starttimes"] = str(
                {det: runtimes[run][det][0] for det in detectors}
            )
            hetconfigfile["heterodyne"]["endtimes"] = str(
                {det: runtimes[run][det][1] for det in detectors}
            )

            hetconfigfile["heterodyne"]["frametypes"] = str(
                {det: CVMFS_GWOSC_DATA_TYPES[run][srate][det] for det in detectors}
            )
            hetconfigfile["heterodyne"]["host"] = CVMFS_GWOSC_DATA_SERVER
            hetconfigfile["heterodyne"]["channels"] = str(
                {det: CVMFS_GWOSC_FRAME_CHANNELS[run][srate][det] for det in detectors}
            )
            hetconfigfile["heterodyne"]["includeflags"] = str(
                {det: segments[run][det] for det in detectors}
            )
            if hwinj:
                hetconfigfile["heterodyne"]["includeflags"] = str(
                    {det: segments[run][det]["includesegments"] for det in detectors}
                )
                hetconfigfile["heterodyne"]["excludeflags"] = str(
                    {det: segments[run][det]["excludesegments"] for det in detectors}
                )
            else:
                hetconfigfile["heterodyne"]["includeflags"] = str(
                    {det: segments[run][det] for det in detectors}
                )
            hetconfigfile["heterodyne"]["outputdir"] = str(
                {det: os.path.join(output, det) for det in detectors}
            )
            hetconfigfile["heterodyne"]["overwrite"] = "False"

            # set whether to use Tempo2 for phase evolution
            if kwargs.get("usetempo2", args.usetempo2):
                hetconfigfile["heterodyne"]["usetempo2"] = "True"

            # split the analysis into on average day long chunks
            if kwargs.get("joblength", args.joblength) is None:
                hetconfigfile["heterodyne"]["joblength"] = "86400"
            else:
                hetconfigfile["heterodyne"]["joblength"] = str(
                    kwargs.get("joblength", args.joblength)
                )

            # set whether running a purely incoherent analysis or not
            peconfigfile["pe"] = {}
            peconfigfile["pe"]["incoherent"] = "True"
            peconfigfile["pe"]["coherent"] = str(
                not kwargs.get("incoherent", args.incoherent)
            )

            # merge the resulting files and remove individual files
            hetconfigfile["merge"] = {}
            hetconfigfile["merge"]["remove"] = "True"
            hetconfigfile["merge"]["overwrite"] = "True"

        nshifts = kwargs.get("nshifts", args.nshifts)
        exclusion = kwargs.get("exclusion", args.exclusion)
        overlapfrac = kwargs.pop("overlap", args.overlap)

    if isinstance(hetconfigfile, configparser.ConfigParser) and isinstance(
        peconfigfile, configparser.ConfigParser
    ):
        hetconfig = hetconfigfile
        peconfig = peconfigfile
    else:
        hetconfig = configparser.ConfigParser()
        peconfig = configparser.ConfigParser()

        try:
            hetconfig.read_file(open(hetconfigfile, "r"))
        except Exception as e:
            raise IOError(
                f"Problem reading configuration file '{hetconfigfile}'\n: {e}"
            )

        try:
            peconfig.read_file(open(peconfigfile, "r"))
        except Exception as e:
            raise IOError(f"Problem reading configuration file '{peconfigfile}'\n: {e}")

    # check that heterodyne_job section exists
    if not hetconfig.has_section("heterodyne_job"):
        hetconfig["heterodyne_job"] = {}

    # check for single pulsar
    if pulsar is None:
        pulsar = hetconfig.get("ephemerides", "pulsarfiles", fallback=None)
    if not isinstance(pulsar, str):
        raise ValueError("A pulsar parameter file must be given")

    # set sky-shift output directory, e.g., append "skyshift" onto the base directory
    if hetconfig.get("run", "basedir", fallback=None) is None:
        basedir = os.path.join(os.getcwd(), "skyshift")
    else:
        basedir = hetconfig.get("run", "basedir")
    for cf in [hetconfig, peconfig]:
        if not cf.has_section("run"):
            cf["run"] = {}
        cf["run"]["basedir"] = basedir

    # read in the parameter file
    psr = PulsarParameters(pulsar)

    # get the ecliptic hemisphere of the source
    ra = psr["RAJ"] if psr["RAJ"] is not None else psr["RA"]
    dec = psr["DECJ"] if psr["DECJ"] is not None else psr["DEC"]
    pos = SkyCoord(ra, dec, unit="rad")  # current position
    hemisphere = "north" if pos.barycentrictrueecliptic.lat >= 0 else "south"

    # set skyshift section in hetconfig
    hetconfig["skyshift"] = {}
    hetconfig["skyshift"]["nshifts"] = str(nshifts)
    hetconfig["skyshift"]["exclusion"] = str(exclusion)
    hetconfig["skyshift"]["hemisphere"] = hemisphere

    # check whether calculating overlap
    if overlapfrac is not None:
        hetconfig["skyshift"]["overlap"] = str(overlapfrac)
        if overlapfrac <= 0.0 or overlapfrac >= 1.0:
            raise ValueError("Overlap fraction must be between 0 and 1")

        # get observing time
        starttimes = ast.literal_eval(hetconfig.get("heterodyne", "starttimes"))
        endtimes = ast.literal_eval(hetconfig.get("heterodyne", "endtimes"))

        # get earliest start time
        if isinstance(starttimes, (int, float)):
            starttime = starttimes
        elif isinstance(starttimes, dict):
            starttime = min(starttimes.values())
        else:
            raise TypeError("Supplied start times must be number or dictionary")

        # get latest end time
        if isinstance(endtimes, (int, float)):
            endtime = endtimes
        elif isinstance(endtimes, dict):
            endtime = max(endtimes.values())
        else:
            raise TypeError("Supplied end times must be number or dictionary")

        tobs = endtime - starttime

        hetconfig["skyshift"]["tobs"] = str(tobs)

    # add ephemeris settings
    if not hetconfig.has_section("ephemerides"):
        hetconfig["ephemerides"] = {}

    # set up two-stage heterodyne
    hetconfig["ephemerides"]["pulsarfiles"] = pulsar
    hetconfig["heterodyne"]["stages"] = "2"

    filterknee = hetconfig.get("heterodyne", "filterknee", fallback=None)
    dfmax = None
    if filterknee is None:
        # get estimate of required bandwidth
        ve = (2.0 * np.pi * 150e9) / (86400 * 365.25)
        dfmax = (ve / 3e8) * psr["F0"]  # maximum Doppler shift from Earth's orbit

        if psr["BINARY"] is not None:
            vsini = 2.0 * np.pi * psr["A1"] / psr["PB"]
            dfmax += (vsini / 3e8) * psr["F0"]

        dfmax *= 2  # multiply by two for some extra room
        dfmax = max((0.1, dfmax))  # get max of 0.1 or df

        hetconfig["heterodyne"]["filterknee"] = f"{dfmax:.2f}"

    resamplerate = hetconfig.get("heterodyne", "resamplerate", fallback=None)
    if resamplerate is None:
        hetconfig["heterodyne"]["resamplerate"] = "[{}, 1/60]".format(
            1 if dfmax is None else int(np.ceil(2 * dfmax))
        )

    # don't include barycentring for initial heterodyne
    hetconfig["heterodyne"]["includessb"] = "[False, True]"
    hetconfig["heterodyne"]["includebsb"] = "[False, True]"
    hetconfig["heterodyne"]["includeglitch"] = "[False, True]"
    hetconfig["heterodyne"]["includefitwaves"] = "[False, True]"

    # output location for heterodyne (ignore config file location)
    detectors = ast.literal_eval(hetconfig.get("heterodyne", "detectors"))
    if not isinstance(detectors, list):
        detectors = [detectors]

    hetconfig["heterodyne"]["outputdir"] = str(
        [
            {
                det: os.path.join(hetconfig["run"]["basedir"], f"stage{stage}", det)
                for det in detectors
            }
            for stage in [1, 2]
        ]
    )

    if not hetconfig.has_section("heterodyne_dag"):
        hetconfig["heterodyne_dag"] = {}

    if not peconfig.has_section("pe_dag"):
        peconfig["pe_dag"] = {}

    # make sure "file transfer" is consistent with heterodyne value
    if hetconfig.getboolean(
        "heterodyne_dag", "transfer_files", fallback=True
    ) != hetconfig.getboolean("pe_dag", "transfer_files", fallback=True):
        peconfig["pe_dag"]["transfer_files"] = hetconfig.get(
            "heterodyne_dag", "transfer_files", fallback="True"
        )

    # make sure accounting group information is set for all sections
    sections = ["knope_job", "heterodyne_job", "pe_job"]
    for section in sections:
        accgroup = hetconfig.get(section, "accounting_group", fallback=None)
        if accgroup is not None:
            hetconfig["heterodyne_job"]["accounting_group"] = accgroup
            peconfig["pe_job"]["accounting_group"] = accgroup
            break

    for section in sections:
        accuser = hetconfig.get(section, "accounting_group_user", fallback=None)
        if accuser is not None:
            hetconfig["heterodyne_job"]["accounting_group_user"] = accuser
            peconfig["pe_job"]["accounting_group_user"] = accuser
            break

    # set dag name
    peconfig["pe_dag"]["name"] = "cwinpy_skyshift"

    # set the configuration file location
    configloc = hetconfig.get("heterodyne", "config", fallback="configs")
    hetconfig["heterodyne"]["config"] = configloc

    # set use of OSG
    osg = hetconfig.get("knope_dag", "osg", fallback=None)
    if osg is not None:
        hetconfig["heterodyne_dag"]["osg"] = osg
        peconfig["pe_dag"]["osg"] = osg

        desiredsites = hetconfig.get("knope_dag", "desired_sites", fallback=None)
        if desiredsites is not None:
            hetconfig["heterodyne_dag"]["desired_sites"] = desiredsites
            peconfig["pe_dag"]["desired_sites"] = desiredsites

        undesiredsites = hetconfig.get("knope_dag", "undesired_sites", fallback=None)
        if desiredsites is not None:
            hetconfig["heterodyne_dag"]["undesired_sites"] = undesiredsites
            peconfig["pe_dag"]["undesired_sites"] = undesiredsites

        singularity = hetconfig.get("knope_dag", "singularity", fallback=None)
        if singularity is not None:
            hetconfig["heterodyne_dag"]["singularity"] = singularity
            peconfig["pe_dag"]["singularity"] = singularity

    # set whether to submit or not (via the PE DAG generator)
    submit = hetconfig.get("knope_dag", "submitdag", fallback=None)
    if submit is not None:
        hetconfig["heterodyne_dag"]["submitdag"] = "False"
        peconfig["pe_dag"]["submitdag"] = submit

    # create heterodyne DAG
    build = hetconfig.getboolean("knope_dag", "build", fallback=True)
    hetconfig["heterodyne_dag"]["build"] = "False"  # don't build the DAG yet
    peconfig["pe_dag"]["build"] = str(build)

    # merge files
    hetconfig["merge"]["merge"] = "False"

    hetdag = HeterodyneDAGRunner(hetconfig, **kwargs)

    # add heterodyned files into PE configuration
    datadict = {1.0: {}, 2.0: {}}
    for det in hetdag.mergeoutputs:
        for ff in hetdag.mergeoutputs[det]:
            datadict[ff][det] = {
                psr: hetfile for psr, hetfile in hetdag.mergeoutputs[det][ff].items()
            }

    if len(datadict[1.0]) == 0 and len(datadict[2.0]) == 0:
        raise ValueError("No heterodyned data files are set to exist!")

    # make sure PE section is present
    if not peconfig.has_section("pe"):
        peconfig["pe"] = {}

    if (
        len(datadict[1.0]) > 0
        and peconfig.get("pe", "data-file-1f", fallback=None) is None
    ):
        peconfig["pe"]["data-file-1f"] = str(datadict[1.0])
    if (
        len(datadict[2.0]) > 0
        and peconfig.get("pe", "data-file-2f", fallback=None) is None
        and peconfig.get("pe", "data-file", fallback=None) is None
    ):
        peconfig["pe"]["data-file-2f"] = str(datadict[2.0])

    if (
        peconfig.get("pe", "data-file", fallback=None) is not None
        and peconfig.get("pe", "data-file-2f", fallback=None) is not None
    ):
        # make sure only "data-file-2f" is set rather than "data-file" in case of conflict
        peconfig.remove_option("pe", "data-file")

    # default to doing incoherent (single detector) and coherent (multi-detector) analysis
    peconfig["pe"]["incoherent"] = peconfig.get("pe", "incoherent", fallback="True")
    peconfig["pe"]["coherent"] = peconfig.get("pe", "coherent", fallback="True")

    # create PE DAG
    kwargs["dag"] = hetdag.dag  # add heterodyne DAG
    pedag = PEDAGRunner(peconfig, **kwargs)

    # return the full DAG
    return pedag


def skyshift_pipeline_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_skyshift_pipeline`` script. This just calls
    :func:`cwinpy.knope.skyshift_pipeline`, but does not return any objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = skyshift_pipeline(**kwargs)


def skyshift_results(
    shift,
    orig,
    resdir=None,
    oddstype="cvi",
    scale="log10",
    plot=False,
    kde=False,
    dist=None,
    yscale="log",
    **kwargs,
):
    """
    Using the output of the ``cwinpy_skyshift_pipeline``, generate the Bayesian
    odds statistic for each shifted location and the original location.
    These can also be plotted if requested, which will additionally return a
    :class:`~matplotlib.figure.Figure` object containing the plot.

    If you already have the output of this function and just want to
    (re)produce a plot, then you can instead pass in the results from the
    previous call and use those.

    Parameters
    ----------
    shift: str, array
        If generating the results, this should be the directory containing the
        pulsar parameter (``.par``) files for all the sky-shifted locations. If
        using pre-generated results, i.e., to (re)produce a plot, then this
        should be the Nx3 array containing the right ascension, declination and
        odds for all the sky-shifted sources.
    orig: str, tuple
        If generating the results, this should be the pulsar parameter
        (``.par``) file containing the original un-shifted location. If using
        pre-generated results, then this should be a length 3 array containing
        the right ascension, declination and odds to the true source.
    resdir: str
        The directory containing the parameter estimation outputs for each
        sky-shifted location.
    oddstype: str
        The odds type used by :func:`~cwinpy.pe.peutils.results_odds`. Defaults
        to ``"cvi"``.
    scale: str
        The odds scale used by :func:`~cwinpy.pe.peutils.results_odds`.
        Defaults to ``"log10"``.
    plot: str, bool
        If ``plot`` is given as ``"hist"`` a histogram of the distribution of
        sky-shifted odds values will be produced; if "invcdf", then it will
        plot the distribution (1 - CDF); if ``"hexbin"``, then a
        :func:`matplotlib.pyplot.hexbin` plot will be produced using the sky
        positions converted into a cartesian coordinate system and the maximum
        odds in each bin; if healpy is installed then the
        :func:`healpy.newvisufunc.projview` function will be used if any of
        the following arguments are given ``"hammer"``, ``"lambert"``,
        ``"mollweide"``, ``"cart"``, or ``"aitoff"``, which will plot the
        maximum odds in each HEALPix pixel. This defaults to ``False``, i.e.,
        no plot will be produced.
    plotkwargs: dict
        If making a plot, and further keyword arguments required for the
        plotting function can be passed using this dictionary.
    kde: bool
        If plotting a histogram plot and this is ``True``, a KDE of the
        distribution will also be added using the
        :class:`scipy.stats.gaussian_kde` function. The probability of getting a
        value greater than the true sky position's odds based on the KDE will
        be added to the plot. If the ``dist`` argument is given then the KDE
        will be ignored and the gamma distribution plotted instead.
    kdekwargs: dict
        If plotting a KDE, any keyword arguments can be passed using this
        dictionary.
    dist: str
        If plotting the histogram (and not plotting a KDE) then a fit to that
        histogram can also be added based on the name of the distribution given
        by this argument. Currently, the value here can either be ``"gamma"``
        (using :class:`scipy.stats.gamma`) or ``"gumbel"`` (using
        :class:`scipy.stats.gumbel_r`), for whi a best fit of the given
        distribution  will be plotted. The probability of getting a value from
        that distribution greater than the true sky position's odds will be
        added to the plot.
    yscale: str
        The scaling on the y-axis of a histogram or inverse CDF plot will
        default to a log scale. To set it to a linear scale set this argument
        to ``"linear"``.

    Returns
    -------
    shiftodds: array
        A :func:`numpy.ndarray` containing the right ascension, declination and
        odds value for all sky-shifted locations.
    trueodds: tuple
        A tuple containing the right ascension, declination and odds for the
        original un-shifted location.
    """

    if all([isinstance(item, (str, pathlib.Path)) for item in [shift, orig, resdir]]):
        # produce odds values from parameter estimation output
        shiftra = []
        shiftdec = []
        shiftnames = []
        shiftodds = []

        shiftpaths = list(pathlib.Path(shift).glob("*.par"))

        # get sky-shift locations (add original on to the end)
        for i, p in enumerate(shiftpaths + [orig]):
            # if original par file is prsent in shiftpaths ignore it
            if i < len(shiftpaths):
                if pathlib.Path(orig).name == p.name:
                    continue

            if is_par_file(p):
                psr = PulsarParameters(p)

                shiftra.append(psr["RAJ"] if psr["RAJ"] is not None else psr["RA"])
                shiftdec.append(psr["DECJ"] if psr["DECJ"] is not None else psr["DEC"])
                shiftnames.append(get_psr_name(psr))

        if len(shiftra) == 1:
            raise IOError(
                f"No valid sky-shifted pulsar parameter files were found in {shift}"
            )

        # get odds for each sky-shifted source
        logodds = results_odds(resdir, oddstype=oddstype, scale=scale)

        if set(logodds.keys()) != set(shiftnames):
            raise RuntimeError(
                "Inconsistent parameter file and results file directories"
            )

        # get odds
        shiftodds = [logodds[key] for key in logodds if key != shiftnames[-1]]
        shiftodds.append(
            logodds[shiftnames[-1]]
        )  # make sure true position is added last

        shiftout = np.array([shiftra[:-1], shiftdec[:-1], shiftodds[:-1]]).T
        trueodds = (shiftra[-1], shiftdec[-1], shiftodds[-1])
    else:
        # use pre-generated odds
        shiftout = np.atleast_2d(shift)

        if shiftout.shape[1] != 3 or shiftout.dtype != float:
            raise TypeError(
                "Array of sky-shifted values must be an Nx3 array of floats"
            )

        trueodds = np.atleast_2d(orig).reshape((3, 1))

        if trueodds.shape != (3, 1) or trueodds.dtype != float:
            raise TypeError("True values must include RA, DEC and odds")

        fullarr = np.concatenate((shiftout, trueodds.T))
        shiftra = fullarr[:, 0]
        shiftdec = fullarr[:, 1]
        shiftodds = fullarr[:, 2]
        trueodds = trueodds.flatten().tolist()

    if plot:
        scale_label = r"\log{}_{10}" if scale == "log10" else r"\ln{}"

        if plot.lower() in ["hist", "histogram", "invcdf", "1-cdf"]:
            fig, ax = plt.subplots()
            plotkwargs = kwargs.get("plotkwargs", {})

            plotkwargs.setdefault("bins", 25)
            plotkwargs.setdefault("density", True)
            plotkwargs.setdefault("histtype", "step")

            if plot.lower() in ["invcdf", "1-cdf"]:
                plotkwargs["cumulative"] = -1

            ax.hist(shiftout[:, 2], **plotkwargs)
            ax.axvline(trueodds[2], ls="--", color="k")

            if plot.lower() in ["hist", "histogram"]:
                ax.set_ylabel(rf"$p({scale_label}\mathcal{{O}}_{{\rm {oddstype}}})$")
            else:
                ax.set_ylabel("1 - CDF")

            ax.set_xlabel(rf"${scale_label}\mathcal{{O}}_{{\rm {oddstype}}}$")

            if kde or dist:
                # get range for kde/gamma evaluation for plotting
                frange = shiftout[:, 2].max() - shiftout[:, 2].min()
                xmin = shiftout[:, 2].min() - 0.25 * frange
                xmax = shiftout[:, 2].max() + 0.25 * frange
                xrange = np.linspace(xmin, xmax, 250)

            if kde and dist is None:
                from scipy.stats import gaussian_kde

                kdekwargs = kwargs.get("kdekwargs", {})

                # generation kde
                kdefunc = gaussian_kde(shiftout[:, 2], **kdekwargs)

                if plot.lower() in ["hist", "histogram"]:
                    ax.plot(xrange, kdefunc(xrange), "k-")
                else:
                    # get inverse cdf
                    icdf = 1 - np.array(
                        [kdefunc.integrate_box_1d(-np.inf, x) for x in xrange]
                    )
                    ax.plot(xrange, icdf)

                    prob = kdefunc.integrate_box_1d(trueodds[2], np.inf)

            elif dist is not None:
                from scipy.stats import gamma, gumbel_r

                dfuncdict = {"gamma": gamma, "gumbel": gumbel_r}

                try:
                    dfunc = dfuncdict[dist.lower()]
                except (AttributeError, KeyError):
                    raise ValueError(
                        f"Distribution '{dist}' must be 'gamma' or 'gumbel'"
                    )

                # fit gamma distribution (need to shift to be sure it is positive)
                gammashift = (
                    3 * np.abs(shiftout[:, 2].min()) if dist.lower() == "gamma" else 0.0
                )
                fg = dfunc.fit(shiftout[:, 2] + gammashift)

                if plot.lower() in ["hist", "histogram"]:
                    ax.plot(xrange, dfunc.pdf(xrange + gammashift, *fg))
                else:
                    # inverse CDF
                    ax.plot(xrange, 1 - dfunc.cdf(xrange + gammashift, *fg))

                    prob = 1 - dfunc.cdf(trueodds[2] + gammashift, *fg)

            if (kde or dist) and plot.lower() in ["invcdf", "1-cdf"]:
                if prob != 0.0:
                    # convert to scientific notation
                    b10e = int(np.floor(np.log10(prob)))
                    b10m = prob / 10**b10e
                    prob = rf"${b10m:.1f}\!\times\!10^{{{b10e}}}$"

                # add text with probability
                ax.text(
                    0.962,
                    0.5,
                    (
                        rf"$p({scale_label}\mathcal{{O}}_{{\rm {oddstype}}}) \geq {scale_label}\mathcal{{O}}"
                        rf"_{{\rm {oddstype}}}^{{\rm source}}$ = {prob}"
                    ),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                )
                ax.set_xbound(lower=shiftout[:, 2].min())

            ax.set_yscale(yscale)

            fig.tight_layout()

        elif plot.lower() == "hexbin":
            from astropy.coordinates import spherical_to_cartesian

            fig, ax = plt.subplots()

            plotkwargs = kwargs.get("plotkwargs", {})

            # convert RA, DEC to cartesian coords
            x, y, _ = spherical_to_cartesian(np.ones_like(shiftra), shiftdec, shiftra)

            # use the largest value in a bin
            plotkwargs.setdefault("reduce_C_function", np.amax)
            plotkwargs.setdefault("gridsize", 20)

            ax.hexbin(x, y, C=shiftodds, **plotkwargs)
            cbar = ax.colorbar()
            cbar.ax.set_ylabel(
                rf"${scale_label}\mathcal{{O}}_{{\rm {oddstype}}}$",
                rotation=270,
                labelpad=15,
            )

            # draw circle around True location
            ax.plot(x[-1], y[-1], marker="o", ls="none", mfc="none", ms=20, c="m")

            ax.set_aspect("equal", "box")
        elif plot.lower() in ["hammer", "lambert", "mollweide", "cart", "aitoff"]:
            try:
                import healpy as hp
                from healpy.newvisufunc import projview, newprojplot
            except (ModuleNotFoundError, ImportError):
                raise ImportError(
                    "You can only use 'hammer', 'lambert', 'mollweide', "
                    "'cart', or 'aitoff' if you have healpy installed"
                )

            plotkwargs = kwargs.get("plotkwargs", {})

            # set number of healpix pixels
            nside = plotkwargs.pop("nside", 8)

            pixel_indices = hp.ang2pix(
                nside, np.abs(np.array(shiftdec) - (np.pi / 2)), shiftra
            )

            # create map using the maximim odds in a pixel
            minodds = np.min(shiftodds)
            m = np.full(hp.nside2npix(nside), minodds - 1)

            for i in range(len(m)):
                if i in pixel_indices:
                    idxs = i == pixel_indices
                    m[i] = np.max(shiftodds[idxs])

            # set colormap
            cmap = matplotlib.cm.get_cmap(plotkwargs.get("cmap", "viridis")).copy()
            cmap.set_under((0.95, 0.95, 0.95, 1))

            im = projview(
                m,
                min=minodds,
                coord=plotkwargs.get("coord", ["G"]),
                graticule=plotkwargs.get("graticule", True),
                graticule_labels=plotkwargs.get("graticule_labels", True),
                projection_type=plot.lower(),
                cmap=cmap,
                cbar=False,  # don't add color bar
            )

            fig = plt.gcf()
            ax = plt.gca()

            # add new color
            fig.colorbar(
                im,
                ax=ax,
                extend="neither",
                label=rf"${scale_label}\mathcal{{O}}_{{\rm {oddstype}}}$",
                shrink=0.67,
                pad=0.04,
            )

            # get pixel containing the true position
            truepix = hp.ang2pix(nside, np.abs(trueodds[1] - np.pi / 2), trueodds[0])
            pixpos = hp.pix2ang(nside, truepix)

            # add circle around the pixel containing the true sky position
            newprojplot(
                theta=pixpos[0],
                phi=-(2 * np.pi - pixpos[1]),
                marker="o",
                mfc="none",
                ms=20,
                c="m",
            )
        else:
            raise ValueError(f"Unknown plot type {plot}")

        return shiftout, trueodds, fig
    else:
        return shiftout, trueodds
