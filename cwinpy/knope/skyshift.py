import ast
import configparser
import os
from argparse import ArgumentParser

import numpy as np
from astropy.coordinates import SkyCoord

from ..heterodyne.heterodyne import HeterodyneDAGRunner
from ..info import (
    ANALYSIS_SEGMENTS,
    CVMFS_GWOSC_DATA_SERVER,
    CVMFS_GWOSC_DATA_TYPES,
    CVMFS_GWOSC_FRAME_CHANNELS,
    RUNTIMES,
)
from ..parfile import PulsarParameters
from ..pe.pe import PEDAGRunner

# from pathlib import Path


def skyshift_pipeline(**kwargs):
    """
    Run skyshift pipeline within Python. This will create a `HTCondor <https://research.cs.wisc.edu/htcondor/>`_
    DAG for consecutively running a ``cwinpy_heterodyne`` and multiple ``cwinpy_pe`` instances on a
    computer cluster. Optional parameters that can be used instead of a configuration file (for
    "quick setup") are given in the "Other parameters" section.

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

            if pulsar is None:
                raise ValueError("No pulsar parameter file has be provided")

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

            pulsar = kwargs.get("pulsar", args.pulsar)
            if pulsar is None:
                raise ValueError("No pulsar parameter file has be provided")

            # get sample rate
            srate = "16k" if (args.samplerate[0:2] == "16" and run[0] == "O") else "4k"

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
            hetconfigfile["heterodyne"]["outputdir"] = str(
                {det: os.path.join(args.output, det) for det in detectors}
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
    basedir = os.path.join(
        hetconfig.get("run", "basedir", fallback=os.getcwd()), "skyshift"
    )
    for cf in [hetconfig, peconfig]:
        if not cf.has_section("run"):
            cf["run"] = {}
        cf["run"]["basedir"] = basedir

    # read in the parameter file
    psr = PulsarParameters(pulsar)

    # generate new positions
    ra = psr["RAJ"]
    dec = psr["DECJ"]
    pos = SkyCoord(ra, dec, unit="rad")  # current position
    hemisphere = "north" if pos.barycentrictrueecliptic.lat >= 0 else "south"

    # set skyshift section in hetconfig
    hetconfig["skyshift"] = {}
    hetconfig["skyshift"]["nshifts"] = str(nshifts)
    hetconfig["skyshift"]["exclusion"] = str(exclusion)
    hetconfig["skyshift"]["overlap"] = str(overlapfrac)
    hetconfig["skyshift"]["hemisphere"] = hemisphere

    # check whether calculating overlap
    if overlapfrac is not None:
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
    hetconfig["heterodyne"]["stages"] = 2

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
        {
            det: [
                os.path.join(hetconfig["run"]["basedir"], f"stage{stage}", det)
                for stage in [1, 2]
            ]
            for det in detectors
        }
    )

    # make sure "file transfer" is consistent with heterodyne value
    if hetconfig.getboolean(
        "heterodyne_dag", "transfer_files", fallback=True
    ) != hetconfig.getboolean("pe_dag", "transfer_files", fallback=True):
        if not peconfig.has_section("pe_dag"):
            peconfig["pe_dag"] = {}
        peconfig["pe_dag"]["transfer_files"] = hetconfig.get(
            "heterodyne_dag", "transfer_files", fallback="True"
        )

    # DAG name is taken from the "knope_dag" section, but falls-back to
    # "cwinpy_knope" if not given
    hetconfig["heterodyne_dag"]["name"] = hetconfig.get(
        "knope_dag", "name", fallback="cwinpy_knope"
    )

    # set accounting group information
    accgroup = hetconfig.get("knope_job", "accounting_group", fallback=None)
    accuser = hetconfig.get("knope_job", "accounting_group_user", fallback=None)
    if accgroup is not None:
        hetconfig["heterodyne_job"]["accounting_group"] = accgroup
        peconfig["pe_job"]["accounting_group"] = accgroup
    if accuser is not None:
        hetconfig["heterodyne_job"]["accounting_group_user"] = accuser
        peconfig["pe_job"]["accounting_group_user"] = accuser

    # set job names (for sub files) to be different for the 2 stages
    jobname = hetconfig.get("heterodyne_job", "name", fallback="cwinpy_heterodyne")
    if not hetconfig.has_option("heterodyne_job", "name"):
        hetconfig["heterodyne_job"]["name"] = jobname + "_skyshift"

    # set the configuration file location
    configloc = hetconfig.get("heterodyne", "config", fallback="config")
    hetconfig["heterodyne"]["config"] = configloc

    # set use of OSG
    osg = hetconfig.get("knope_dag", "osg", fallback=None)
    if osg is not None:
        hetconfig["heterodyne_dag"]["osg"] = osg
        peconfig["pe_dag"]["osg"] = osg

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

    # create PE DAG
    kwargs["dag"] = hetdag.dag  # add heterodyne DAG
    kwargs["generation_nodes"] = hetdag.pulsar_nodes  # add Heterodyne nodes
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
