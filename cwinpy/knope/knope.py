import configparser
import os
from argparse import ArgumentParser

import cwinpy

from ..heterodyne.heterodyne import HeterodyneDAGRunner, heterodyne
from ..info import (
    ANALYSIS_SEGMENTS,
    CVMFS_GWOSC_DATA_SERVER,
    CVMFS_GWOSC_DATA_TYPES,
    CVMFS_GWOSC_FRAME_CHANNELS,
    HW_INJ,
    HW_INJ_RUNTIMES,
    HW_INJ_SEGMENTS,
    RUNTIMES,
)
from ..pe.pe import PEDAGRunner, pe


def create_knope_parser():
    """
    Create the argument parser for ``cwinpy_knope``.
    """

    description = """\
A script to run the CWInPy known pulsar analysis pipeline; gravitational-wave \
data will be preprocessed based on the phase evolution of a pulsar which will \
then be used to infer the unknown signal parameters.
"""

    parser = ArgumentParser(
        description=description,
        allow_abbrev=False,
    )
    parser.add(
        "--heterodyne-config",
        action="append",
        help=(
            "A configuration file for the heterodyne pre-processing using "
            "cwinpy_heterodyne. If requiring multiple detectors then this "
            "option can be used multiple times to pass configuration files "
            "for each detector."
        ),
        required=True,
    )
    parser.add(
        "--pe-config",
        type=str,
        help=(
            "A configuration file for the Bayesian inference stage using " "cwinpy_pe."
        ),
        required=True,
    )
    parser.add(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=cwinpy.__version__),
    )

    return parser


def knope(**kwargs):
    """
    Run the known pulsar pipeline within Python. It is highly recommended to
    run the pipeline script ``cwinpy_knope_dag`` to create a HTCondor DAG,
    particular if using data from a long observing run and/or for multiple
    pulsars. The main use of this function is for quick testing.

    This interface can be used for analysing data from multiple detectors or
    multiple harmonics by passing multiple heterodyne configuration settings
    for each case, but this is **not** recommended.

    Parameters
    ----------
    heterodyne_config: str, list
        The path to a configuration file, or list of paths to multiple
        configuration files if using more than one detector, of the type
        required by ``cwinpy_heterodyne``.
    hetkwargs: dict, list
        If not using a configuration file, arguments for
        :func:`cwinpy.heterodyne.heterodyne` can be provided as a dictionary,
        or list of dictionaries if requiring more than one detector.
    pe_config: str
        The path to a configuration file of the type required by ``cwinpy_pe``.
        The input data arguments and pulsars used will be assumed to be the
        same as for the heterodyne configuration file, so any values given in
        the configuration file will be ignored.
    pekwargs: dict
        If not using a configuration file, arguments for :func:`cwinpy.pe.pe`
        can be provided as a dictionary.

    Returns
    -------
    tuple:
        The returned values are a dictionary containing lists of the
        :class:`cwinpy.heterodyne.Heterodyne` objects produced during
        heterodyning (keyed by frequency scale factor) and a dictionary
        (keyed by pulsar name) containing :class:`bilby.core.result.Result`
        objects for each pulsar.
    """

    hetkwargs = []
    pekwargs = {}

    if "cli" in kwargs:  # pragma: no cover
        parser = create_knope_parser()
        args = parser.parse_args()

        # set configuration files
        hetconfig = args.heterodyne_config
        peconfig = args.pe_config

        for conf in hetconfig:
            hetkwargs.append({"config": conf})
        pekwargs["config"] = peconfig
    else:
        if "heterodyne_config" in kwargs:
            if isinstance(kwargs["heterodyne_config"], str):
                hetkwargs.append({"config": kwargs["heterodyne_config"]})
            elif isinstance(kwargs["heterodyne_config"], list):
                for conf in kwargs["heterodyne_config"]:
                    hetkwargs.append({"config": conf})
            else:
                raise TypeError(
                    "heterodyne_config argument must be a string or list of configuration files."
                )

        if "pe_config" in kwargs:
            pekwargs["config"] = kwargs["pe_config"]

        if "hetkwargs" in kwargs:
            if isinstance(kwargs["hetkwargs"], dict):
                if len(hetkwargs) == 0:
                    hetkwargs.append(kwargs["hetkwargs"])
                elif len(hetkwargs) == 1:
                    hetkwargs[0].update(kwargs["hetkwargs"])
                else:
                    raise TypeError("Inconsistent heterodyne_config and hetkwargs")

        try:
            pekwargs.update(**kwargs["pekwargs"])
        except (KeyError, TypeError):
            pass

    if len(hetkwargs) == 0:
        raise ValueError(
            "No heterodyned configuration file or keyword arguments have been given"
        )

    if len(pekwargs) == 0:
        raise ValueError(
            "No parameter estimation configuration file or keyword arguments have been given"
        )

    # run heterodyne
    hetrun = {}
    freqfactordet = {}
    for i, hkw in enumerate(hetkwargs):
        het = heterodyne(**hkw)
        if het.freqfactor in freqfactordet:
            freqfactordet[het.freqfactor].append(het.detector)
            hetrun[het.freqfactor].append(het)
        else:
            freqfactordet[het.freqfactor] = [het.detector]
            hetrun[het.freqfactor] = [het]

        # check for consistent pulsars for each detector
        if i == 0:
            pulsars = sorted(list(het.pulsars))
        else:
            if pulsars != sorted(het.pulsars):
                raise ValueError("Inconsistent pulsars between heterodynes.")

    # run parameter estimation
    perun = {}  # store results as a dictionary
    for pulsar in pulsars:
        for ff in freqfactordet.keys():
            pekwargs["data_file_{0:d}f".format(int(ff))] = [
                "{}:{}".format(det, het.outputfiles[pulsar])
                for het, det in zip(hetrun[ff], freqfactordet[ff])
            ]

        # remove "detector" if already given in pekwargs as this will be
        # given by the data file lists above
        if "detector" in pekwargs:
            pekwargs.pop("detector")

        # remove "par_file" if given in pekwargs as this will be set from the
        # heterodyned data
        if "par_file" in pekwargs:
            pekwargs.pop("par_file")

        perun[pulsar] = pe(**pekwargs)

    return hetrun, perun


def knope_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_knope`` script. This just calls
    :func:`cwinpy.knope.knope`, but does not return any objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = knope(**kwargs)


def knope_dag(**kwargs):
    """
    Run knope within Python. This will create a `HTCondor <https://research.cs.wisc.edu/htcondor/>`_
    DAG for consecutively running multiple ``cwinpy_heterodyne`` and ``cwinpy_pe`` instances on a
    computer cluster. Optional parameters that can be used instead of a configuration file (for
    "quick setup") are given in the "Other parameters" section.

    Parameters
    ----------
    config: str
        A configuration file, or :class:`configparser:ConfigParser` object,
        for the analysis.

    Other parameters
    ----------------
    run: str
        The name of an observing run for which open data exists, which will be
        heterodyned, e.g., "O1".
    detector: str, list
        The detector, or list of detectors, for which the data will be
        heterodyned. If not set then all detectors available for a given run
        will be used.
    hwinj: bool
        Set this to True to analyse the continuous hardware injections for a
        given run. No ``pulsar`` argument is required in this case.
    samplerate: str:
        Select the sample rate of the data to use. This can either be 4k or
        16k for data sampled at 4096 or 16384 Hz, respectively. The default
        is 4k, except if running on hardware injections for O1 or later, for
        which 16k will be used due to being requred for the highest frequency
        source. For the S5 and S6 runs only 4k data is avaialble from GWOSC,
        so if 16k is chosen it will be ignored.
    pulsar: str, list
        The path to, or list of paths to, a TEMPO(2)-style pulsar parameter
        file(s), or directory containing multiple parameter files, to
        heterodyne. If a pulsar name is given instead of a parameter file
        then an attempt will be made to find the pulsar's ephemeris from the
        ATNF pulsar catalogue, which will then be used.
    osg: bool
        Set this to True to run on the Open Science Grid rather than a local
        computer cluster.
    output: str,
        The location for outputting the heterodyned data. By default the
        current directory will be used. Within this directory, subdirectories
        for each detector will be created.
    joblength: int
        The length of data (in seconds) into which to split the individual
        analysis jobs. By default this is set to 86400, i.e., one day. If this
        is set to 0, then the whole dataset is treated as a single job.
    accounting_group_tag: str
        For LVK users this sets the computing accounting group tag.
    usetempo2: bool
        Set this flag to use Tempo2 (if installed) for calculating the signal
        phase evolution for the heterodyne rather than the default LALSuite
        functions.

    Returns
    -------
    dag:
        An object containing a pycondor :class:`pycondor.Dagman` object.
    """

    if "config" in kwargs:
        hetconfigfile = kwargs.pop("config")
        peconfigfile = hetconfigfile
    else:  # pragma: no cover
        parser = ArgumentParser(
            description=(
                "A script to create a HTCondor DAG to process GW strain data "
                "by heterodyning it based on the expected phase evolution for "
                "a selection of pulsars, and then perform parameter "
                "estimation for the unknown signal parameters of those "
                "sources."
            )
        )
        parser.add_argument(
            "config",
            nargs="?",
            help=("The configuration file for the analysis"),
            default=None,
        )

        optional = parser.add_argument_group(
            "Quick setup arguments (this assumes CVMFS open data access)."
        )
        optional.add_argument(
            "--run",
            help=(
                "Set an observing run name for which to heterodyne the data. "
                "This can be one of {} for which open data exists".format(
                    list(RUNTIMES.keys())
                )
            ),
        )
        optional.add_argument(
            "--detector",
            action="append",
            help=(
                "The detector for which the data will be heterodyned. This can "
                "be used multiple times to specify multiple detectors. If not "
                "set then all detectors available for a given run will be "
                "used."
            ),
        )
        optional.add_argument(
            "--hwinj",
            action="store_true",
            help=(
                "Set this flag to analyse the continuous hardware injections "
                "for a given run. No '--pulsar' arguments are required in "
                "this case."
            ),
        )
        optional.add_argument(
            "--samplerate",
            help=(
                "Select the sample rate of the data to use. This can either "
                "be 4k or 16k for data sampled at 4096 or 16384 Hz, "
                "respectively. The default is 4k, except if running on "
                "hardware injections for O1 or later, for which 16k will be "
                "used due to being required for the highest frequency source. "
                "For the S5 and S6 runs only 4k data is available from GWOSC, "
                "so if 16k is chosen it will be ignored."
            ),
            default="4k",
        )
        optional.add_argument(
            "--pulsar",
            action="append",
            help=(
                "The path to a TEMPO(2)-style pulsar parameter file, or "
                "directory containing multiple parameter files, to "
                "heterodyne. This can be used multiple times to specify "
                "multiple pulsar inputs. If a pulsar name is given instead "
                "of a parameter file then an attempt will be made to find the "
                "pulsar's ephemeris from the ATNF pulsar catalogue, which "
                "will then be used."
            ),
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
                "The location for outputting the heterodyned data. By default "
                "the current directory will be used. Within this directory, "
                "subdirectories for each detector will be created."
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
                "In running with multiple detectors, set this flag to analyse "
                "each of them independently rather than coherently combining "
                "the data from all detectors. The coherent analysis is the "
                "default."
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
            hetconfigfile = args.config
            peconfigfile = args.config
        else:
            # use the "Quick setup" arguments
            hetconfigfile = configparser.ConfigParser()
            peconfigfile = configparser.ConfigParser()

            run = kwargs.get("run", args.run)
            if run not in RUNTIMES:
                raise ValueError(f"Requested run '{run}' is not available")

            pulsars = []
            if args.hwinj:
                # use hardware injections for the run
                runtimes = HW_INJ_RUNTIMES
                segments = HW_INJ_SEGMENTS
                pulsars.extend(HW_INJ[run]["hw_inj_files"])

                # set sample rate to 16k, expect for S runs
                srate = "16k" if run[0] == "O" else "4k"
            else:
                # use pulsars provided
                runtimes = RUNTIMES
                segments = ANALYSIS_SEGMENTS

                pulsar = kwargs.get("pulsar", args.pulsar)
                if pulsar is None:
                    raise ValueError("No pulsar parameter files have be provided")

                pulsars.extend(pulsar if isinstance(list) else [pulsar])

                # get sample rate
                srate = (
                    "16k" if (args.samplerate[0:2] == "16" and run[0] == "O") else "4k"
                )

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
            hetconfigfile["run"] = {}
            hetconfigfile["run"]["basedir"] = kwargs.get("output", args.output)
            peconfigfile["run"] = {}
            peconfigfile["run"]["basedir"] = kwargs.get("output", args.output)

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

            # add ephemeris settings
            hetconfigfile["ephemerides"] = {}
            hetconfigfile["ephemerides"]["pulsarfiles"] = str(pulsars)

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
            if args.hwinj:
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

            # set whether running a coherent or incoherent analysis
            peconfigfile["pe"] = {}
            peconfigfile["pe"]["incoherent"] = str(
                kwargs.get("incoherent", args.incoherent)
            )
            peconfigfile["pe"]["coherent"] = str(
                not kwargs.get("incoherent", args.incoherent)
            )

            # merge the resulting files and remove individual files
            hetconfigfile["merge"] = {}
            hetconfigfile["merge"]["remove"] = "True"
            hetconfigfile["merge"]["overwrite"] = "True"

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
        hetconfigfile["heterodyne_job"]["accounting_group"] = accgroup
        peconfigfile["pe_job"]["accounting_group"] = accgroup
    if accuser is not None:
        hetconfigfile["heterodyne_job"]["accounting_group_user"] = accuser
        peconfigfile["pe_job"]["accounting_group_user"] = accuser

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
    if not hetconfig.has_section("merge"):
        hetconfig["merge"] = {}
    hetconfig["merge"]["merge"] = "True"  # always merge files
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


def knope_dag_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_knope_dag`` script. This just calls
    :func:`cwinpy.knope.knope_dag`, but does not return any objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = knope_dag(**kwargs)
