import configparser
import os
from argparse import ArgumentParser

import cwinpy

from ..heterodyne.heterodyne import HeterodyneDAGRunner, heterodyne
from ..info import (
    ANALYSIS_SEGMENTS,
    CVMFS_GWOSC_FRAME_CHANNELS,
    CVMFS_GWOSC_FRAME_DATA_LOCATIONS,
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
        type=str,
        help=(
            "A configuration file for the heterodyne pre-processing using "
            "cwinpy_heterodyne."
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


def knope(**kwargs):
    """
    Run the known pulsar pipeline within Python. It is highly recommended to
    run the pipeline using a HTCondor DAG particular if using data from a long
    observing run and/or for multiple pulsar. The main use of this function is
    for quick testing for individual sources and single detectors. This
    interface cannot be used for analysing data from multiple detectors or
    mulitple harmonics for a particular pulsar. In those cases the HTCondor DAG
    pipeline script ``cwinpy_knope_dag`` **must** be used.

    Parameters
    ----------
    heterodyne_config: str
        The path to a configuration file of the type required by
        ``cwinpy_heterodyne``.
    hetkwargs: dict
        If not using a configuration file, arguments for
        :func:`cwinpy.heterodyne.heterodyne` can be provided as a dictionary.
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
        The returned value is a tuple containing the
        :class:`cwinpy.heterodyne.Heterodyne` object produced and a dictionary
        (keyed by pulsar name) containing :class:`bilby.core.result.Result`
        objects for each pulsar.
    """

    hetkwargs = {}
    pekwargs = {}

    if "cli" in kwargs:
        parser = create_knope_parser()
        args = parser.parse_args()

        # set configuration files
        hetconfig = args.heterodyne_config
        peconfig = args.pe_config

        hetkwargs["config"] = hetconfig
        pekwargs["config"] = peconfig
    else:
        if "heterodyne_config" in kwargs:
            hetkwargs["config"] = kwargs["heterodyne_config"]

        if "pe_config" in kwargs:
            pekwargs["config"] = kwargs["pe_config"]

        try:
            hetkwargs.update(**kwargs["hetkwargs"])
        except (KeyError, TypeError):
            pass

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
    het = heterodyne(**hetkwargs)

    # run parameter estimation
    perun = {}
    for pulsar in het.pulsars:
        pekwargs["data_file_{0:d}".format(int(het.freqfactor))] = het.outputfiles[
            pulsar
        ]
        pekwargs["detector"] = het.detector
        perun[pulsar] = pe(**pekwargs)

    return het, perun


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
    computer cluster.

    Parameters
    ----------
    config: str
        A configuration file, or :class:`configparser:ConfigParser` object,
        for the analysis.

    Optional parameters
    -------------------
    run: str
        The name of an observing run for which open data exists, which will be
        heterodyned, e.g., "O1".

    Returns
    -------
    dag:
        The pycondor :class:`pycondor.Dagman` object.
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
            "--accounting-group-tag",
            dest="accgroup",
            help=("For LVK users this sets the computing accounting group tag"),
        )

        args = parser.parse_args()
        if args.config is not None:
            hetconfigfile = args.config
            peconfigfile = args.config
        else:
            # use the "Quick setup" arguments
            hetconfigfile = configparser.ConfigParser()
            peconfigfile = configparser.ConfigParser()

            run = args.run
            if run not in RUNTIMES:
                raise ValueError("Requested run '{}' is not available".format(args.run))

            pulsars = []
            if args.hwinj:
                # use hardware injections for the run
                runtimes = HW_INJ_RUNTIMES
                segments = HW_INJ_SEGMENTS
                pulsars.extend(HW_INJ[run]["hw_inj_files"])
            else:
                # use pulsars provided
                runtimes = RUNTIMES
                segments = ANALYSIS_SEGMENTS

                if args.pulsar is None:
                    raise ValueError("No pulsar parameter files have be provided")

                pulsars.extend(args.pulsar)

            # check pulsar files/directories exist
            pulsars = [
                pulsar
                for pulsar in pulsars
                if (os.path.isfile(pulsar) or os.path.idir(pulsar))
            ]
            if len(pulsars) == 0:
                raise ValueError("No valid pulsar parameter files have be provided")

            if args.detector is None:
                detectors = list(runtimes[run].keys())
            else:
                detectors = [det for det in args.detector if det in runtimes[run]]
                if len(detectors) == 0:
                    raise ValueError(
                        "Provided detectors '{}' are not valid for the given run".format(
                            args.detector
                        )
                    )

            # create required settings
            hetconfigfile["run"] = {}
            hetconfigfile["run"]["basedir"] = args.output

            hetconfigfile["heterodyne_dag"] = {}
            peconfigfile["pe_dag"] = {}
            if args.osg:
                hetconfigfile["heterodyne_dag"]["osg"] = "True"
                hetconfigfile["pe_dag"]["osg"] = "True"

            hetconfigfile["job"] = {}
            hetconfigfile["job"]["getenv"] = "True"
            if args.accgroup is not None:
                hetconfigfile["job"]["accounting_group"] = args.accgroup

            # add heterodyne settings
            hetconfigfile["heterodyne"] = {}
            hetconfigfile["heterodyne"]["detectors"] = str(detectors)
            hetconfigfile["heterodyne"]["pulsarfiles"] = str(pulsars)
            hetconfigfile["heterodyne"]["starttimes"] = str(
                {det: runtimes[run][det][0] for det in detectors}
            )
            hetconfigfile["heterodyne"]["endtimes"] = str(
                {det: runtimes[run][det][1] for det in detectors}
            )

            hetconfigfile["heterodyne"]["framecaches"] = str(
                {
                    det: CVMFS_GWOSC_FRAME_DATA_LOCATIONS[run]["4k"][det]
                    for det in detectors
                }
            )
            hetconfigfile["heterodyne"]["channels"] = str(
                {det: CVMFS_GWOSC_FRAME_CHANNELS[run]["4k"][det] for det in detectors}
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

            # split the analysis into on average day long chunks
            if args.joblength is None:
                hetconfigfile["heterodyne"]["joblength"] = "86400"
            else:
                hetconfigfile["heterodyne"]["joblength"] = str(args.joblength)

            # merge the resulting files and remove individual files
            hetconfigfile["merge"] = {}
            hetconfigfile["merge"]["remove"] = "True"
            hetconfigfile["merge"]["overwrite"] = "True"

            # add PE settings
            peconfigfile["pe"] = {}

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
                "Problem reading configuration file '{}'\n: {}".format(hetconfigfile, e)
            )

        try:
            peconfig.read_file(open(peconfigfile, "r"))
        except Exception as e:
            raise IOError(
                "Problem reading configuration file '{}'\n: {}".format(peconfigfile, e)
            )

    # create heterodyne DAG
    hetconfig["dag"]["build"] = "False"  # don't build the DAG yet
    hetconfigfile["merge"]["merge"] = "True"  # always merge files
    hetdag = HeterodyneDAGRunner(hetconfig, **kwargs)

    # add heterodyned files into PE configuration
    datadict = {"1f": {}, "2f": {}}
    for det in hetdag.heterodyned_files:
        for ff in hetdag.heterodyned_files[det]:
            datadict[ff][det] = hetdag.heterodyned_files[det][ff]

    if len(datadict["1f"]) == 0 and len(datadict["2f"]) == 0:
        raise ValueError("No heterodyned data files are set to exist!")

    # make sure PE section is present
    if not peconfig.has_section("pe"):
        peconfigfile["pe"] = {}

    if (
        len(datadict["1f"]) > 0
        and peconfig.get("pe", "data-file-1f", fallback=None) is None
    ):
        peconfig["pe"]["data-file-1f"] = str(datadict["1f"])
    if (
        len(datadict["2f"]) > 0
        and peconfig.get("pe", "data-file-2f", fallback=None) is None
        and peconfig.get("pe", "data-file", fallback=None) is None
    ):
        peconfig["pe"]["data-file-2f"] = str(datadict["2f"])

    if (
        peconfig.get("pe", "data-file", fallback=None) is not None
        and peconfig.get("pe", "data-file-2f", fallback=None) is not None
    ):
        # make sure only "data-file-2f" is set rather than "data-file" in case of conflict
        peconfig.remove_option("pe", "data-file")

    # set pulsar files
    if peconfig.get("pe", "pulsars", fallback=None) is None:
        peconfig["pe"]["pulsars"] = str(hetdag.pulsar_files)

    # create PE DAG
    kwargs["dag"] = hetdag  # add heterodyne DAG
    kwargs["generation_nodes"] = hetdag.pulsar_nodes  # add Heterodyne nodes
    pedag = PEDAGRunner(peconfig, **kwargs)

    # return the full DAG
    return pedag


def knope_dag_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_knope`` script. This just calls
    :func:`cwinpy.knope.knope`, but does not return any objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = knope_dag(**kwargs)
