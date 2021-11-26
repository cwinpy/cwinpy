import ast
import configparser
import os
from argparse import ArgumentParser
from copy import deepcopy

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
from ..utils import draw_ra_dec, get_psr_name, int_to_alpha


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
    exclusion: float
        The exclusion region around the source's actual sky location and any
        sky shift locations (including about a "antipode" point on the sky).

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

        skyshift = parser.add_argument_group("Sky shift arguments")
        skyshift.add_argument(
            "--nshifts",
            type=int,
            default=1000,
            required=True,
            help=(
                "The number of random sky-shifts to perform. The default will be "
                "1000."
            ),
        )
        skyshift.add_argument(
            "--exclusion",
            help=(
                "The exclusion region around the source's actual sky location "
                "and any sky shift locations (including about a 'antipode' "
                "point on the sky)."
            ),
            default=0.01,
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
            "--samplerate",
            help=(
                "Select the sample rate of the data to use. This can either be 4k "
                "or 16k for data sampled at 4096 or 16384 Hz, respectively. The "
                "default is 4k. For the S5 and S6 runs only 4k data is available "
                "from GWOSC, so if 16k is chosen it will be ignored."
            ),
            default="4k",
        )
        optional.add_argument(
            "--pulsar",
            help=(
                "The path to a TEMPO(2)-style pulsar parameter file to " "heterodyne."
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

            nshifts = args.nshifts
            exclusion = args.exclusion

    if isinstance(hetconfigfile, configparser.ConfigParser) and isinstance(
        peconfigfile, configparser.ConfigParser
    ):
        hetconfig1 = hetconfigfile  # initial heterodyne
        hetconfig2 = hetconfig1.copy()  # sky-shifted heterodyne
        peconfig = peconfigfile
    else:
        hetconfig1 = configparser.ConfigParser()  # initial heterodyne
        peconfig = configparser.ConfigParser()

        try:
            hetconfig1.read_file(open(hetconfigfile, "r"))
        except Exception as e:
            raise IOError(
                f"Problem reading configuration file '{hetconfigfile}'\n: {e}"
            )

        hetconfig2 = hetconfig1.copy()  # sky-shifted heterodyne

        try:
            peconfig.read_file(open(peconfigfile, "r"))
        except Exception as e:
            raise IOError(f"Problem reading configuration file '{peconfigfile}'\n: {e}")

    # check for single pulsar
    if pulsar is None:
        pulsar = hetconfig1.get("ephemerides", "pulsarfiles", fallback=None)
    if not isinstance(pulsar, str):
        raise ValueError("A pulsar parameter file must be given")

    # set sky-shift output directory, e.g., append "skyshift" onto the base directory
    basedir = os.path.join(
        hetconfig1.get("run", "basedir", fallback=os.getcwd()), "skyshift"
    )
    for cf in [hetconfig1, hetconfig2, peconfig]:
        if not cf.has_section("run"):
            cf["run"] = {}
        cf["run"]["basedir"] = basedir

    # read in the parameter file
    psr = PulsarParameters(pulsar)

    # generate new positions
    ra = psr["RA"]
    dec = psr["DEC"]
    pos = SkyCoord(ra, dec, unit="rad")  # current position
    antipode = SkyCoord(np.fmod(ra + np.pi, 2.0 * np.pi), -1.0 * dec, unit="rad")

    pulsardir = os.path.join(hetconfig1["run"]["basedir"], "pulsars")
    if not os.path.exists(pulsardir):
        os.makedirs(pulsardir)

    # add ephemeris settings
    if not hetconfig1.has_section("ephemerides"):
        hetconfig1["ephemerides"] = {}
        hetconfig2["ephemerides"] = {}

    psrnames = []
    parfiles = []

    for i in range(nshifts):
        # generate points while checking angular distance from true position
        while True:
            ranew, decnew = draw_ra_dec()
            newpos = SkyCoord(ranew, decnew, unit="rad")

            if (
                np.abs(pos.separation(newpos).rad) > exclusion
                and np.abs(antipode.separation(newpos).rad) > exclusion
            ):
                break

        # output par files for all new positions
        newpsr = deepcopy(psr)
        newpsr["RAJ"] = newpos.ra.rad
        newpsr["DECJ"] = newpos.dec.rad

        # make name unique with additional alphabetical values
        anum = int_to_alpha(i)
        newpsr["PSRJ"] = get_psr_name(newpsr) + anum
        psrnames.append(newpsr["PSRJ"])

        parfiles.append(os.path.join(pulsardir, newpsr["PSRJ"]))

        # output parameter file
        newpsr.pp_to_par(parfiles[-1])

    parfiles.append(pulsar)  # add on original parameter file

    # set up first and second stage heterodynes
    hetconfig1["ephemerides"]["pulsarfiles"] = pulsar

    # include all sky-shifted pulsars
    hetconfig2["ephemerides"]["pulsarfiles"] = str(parfiles)

    filterknee = hetconfig1.get("heterodyne", "filterknee", fallback=None)
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

        hetconfig1["heterodyne"]["filterknee"] = f"{dfmax:.2f}"
    hetconfig2.remove_option(
        "heterodyne", "filterknee"
    )  # no filter for second heterodyne

    resamplerate = hetconfig1.get("heterodyne", "resamplerate", fallback=None)
    if resamplerate is None:
        if dfmax is None:
            hetconfig1["heterodyne"]["resamplerate"] = "1"  # 1 Hz default
        else:
            rsrate = int(np.ceil(2 * dfmax))
            hetconfig1["heterodyne"]["resamplerate"] = str(rsrate)
    hetconfig2["heterodyne"]["resamplerate"] = "1/60"

    # don't include barycentring for initial heterodyne
    hetconfig1["heterodyne"]["includessb"] = "False"
    hetconfig1["heterodyne"]["includebsb"] = "False"
    hetconfig1["heterodyne"]["includeglitch"] = "False"
    hetconfig1["heterodyne"]["includefitwaves"] = "False"

    # include barycentring for second heterodyne
    hetconfig2["heterodyne"]["includessb"] = "True"
    hetconfig2["heterodyne"]["includebsb"] = "True"
    hetconfig2["heterodyne"]["includeglitch"] = "True"
    hetconfig2["heterodyne"]["includefitwaves"] = "True"

    # output location for heterodyne (ignore config file location)
    detectors = ast.literal_eval(hetconfig1.get("heterodyne", "detectors"))
    hetconfig1["heterodyne"]["outputdir"] = str(
        {
            det: os.path.join(hetconfig1["run"]["basedir"], "stage1", det)
            for det in detectors
        }
    )
    hetconfig2["heterodyne"]["outputdir"] = str(
        {
            det: os.path.join(hetconfig1["run"]["basedir"], "stage2", det)
            for det in detectors
        }
    )

    # create heterodyned data dictionary pointing to directories
    hetdata = {}
    for name in psrnames:
        hetdata[name] = os.path.join(hetconfig1["run"]["basedir"], "stage1")

    hetconfig2["heterodyne"]["heterodyneddata"] = str(hetdata)
    hetconfig2["crop"] = "0"  # no further cropping required

    # make sure "file transfer" is consistent with heterodyne value
    if hetconfig1.getboolean(
        "heterodyne_dag", "transfer_files", fallback=True
    ) != hetconfig1.getboolean("pe_dag", "transfer_files", fallback=True):
        if not peconfig.has_section("pe_dag"):
            peconfig["pe_dag"] = {}
        peconfig["pe_dag"]["transfer_files"] = hetconfig1.get(
            "heterodyne_dag", "transfer_files", fallback="True"
        )

    # DAG name is taken from the "knope_dag" section, but falls-back to
    # "cwinpy_knope" if not given
    hetconfig1["heterodyne_dag"]["name"] = hetconfig1.get(
        "knope_dag", "name", fallback="cwinpy_knope"
    )

    # set accounting group information
    accgroup = hetconfig1.get("knope_job", "accounting_group", fallback=None)
    accuser = hetconfig1.get("knope_job", "accounting_group_user", fallback=None)
    if accgroup is not None:
        hetconfig1["heterodyne_job"]["accounting_group"] = accgroup
        hetconfig2["heterodyne_job"]["accounting_group"] = accgroup
        peconfig["pe_job"]["accounting_group"] = accgroup
    if accuser is not None:
        hetconfig1["heterodyne_job"]["accounting_group_user"] = accuser
        hetconfig2["heterodyne_job"]["accounting_group_user"] = accuser
        peconfig["pe_job"]["accounting_group_user"] = accuser

    # set use of OSG
    osg = hetconfig1.get("knope_dag", "osg", fallback=None)
    if osg is not None:
        hetconfig1["heterodyne_dag"]["osg"] = osg
        hetconfig2["heterodyne_dag"]["osg"] = osg
        peconfig["pe_dag"]["osg"] = osg

    # set whether to submit or not (via the PE DAG generator)
    submit = hetconfig1.get("knope_dag", "submitdag", fallback=None)
    if submit is not None:
        hetconfig1["heterodyne_dag"]["submitdag"] = "False"
        hetconfig2["heterodyne_dag"]["submitdag"] = "False"
        peconfig["pe_dag"]["submitdag"] = submit

    # create heterodyne DAG
    build = hetconfig1.getboolean("knope_dag", "build", fallback=True)
    hetconfig1["heterodyne_dag"]["build"] = "False"  # don't build the DAG yet
    hetconfig2["heterodyne_dag"]["build"] = "False"
    peconfig["pe_dag"]["build"] = str(build)

    # don't merge in first heterodyne
    if hetconfig1.has_section("merge"):
        hetconfig1.remove_section("merge")

    # do merge for second heterodyne
    hetconfig2["merge"] = {}
    hetconfig2["merge"]["merge"] = "True"  # always merge files

    hetdag1 = HeterodyneDAGRunner(hetconfig1, **kwargs)

    kwargs["dag"] = hetdag1.dag  # add heterodyne DAG
    kwargs["generation_nodes"] = hetdag1.pulsar_nodes
    hetdag2 = HeterodyneDAGRunner(hetconfig2, **kwargs)

    # add heterodyned files into PE configuration
    datadict = {1.0: {}, 2.0: {}}
    for det in hetdag2.mergeoutputs:
        for ff in hetdag2.mergeoutputs[det]:
            datadict[ff][det] = {
                psr: hetfile for psr, hetfile in hetdag2.mergeoutputs[det][ff].items()
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
    kwargs["dag"] = hetdag2.dag  # add heterodyne DAG
    kwargs["generation_nodes"] = hetdag2.pulsar_nodes  # add Heterodyne nodes
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
