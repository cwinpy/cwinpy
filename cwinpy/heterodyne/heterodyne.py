"""
Run heterodyne pre-processing of gravitational-wave data.
"""

import ast
import configparser
import os
import signal
import sys

import cwinpy
import numpy as np
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.input import Input
from bilby_pipe.job_creation.dag import Dag
from bilby_pipe.job_creation.node import Node
from bilby_pipe.utils import (
    BilbyPipeError,
    check_directory_exists_and_if_not_mkdir,
    parse_args,
)
from configargparse import DefaultConfigFileParser

from ..utils import sighandler
from .base import Heterodyne


def create_heterodyne_parser():
    """
    Create the argument parser.
    """

    description = """\
A script to heterodyne raw gravitational-wave strain data based on the \
expected evolution of the gravitational-wave signal from a set of pulsars."""

    parser = BilbyArgParser(
        prog=sys.argv[0],
        description=description,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
    )
    parser.add("--config", type=str, is_config_file=True, help="Configuration ini file")
    parser.add(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=cwinpy.__version__),
    )
    parser.add(
        "--periodic-restart-time",
        default=43200,
        type=int,
        help=(
            "Time after which the job will be self-evicted with code 130. "
            "After this, condor will restart the job. Default is 43200s. "
            "This is used to decrease the chance of HTCondor hard evictions."
        ),
    )
    parser.add(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Set this flag to resume heterodyning in case not all pulsars "
            "completed. This checks whether output files (as set using "
            '"--output" and "--label" arguments) already exist and does '
            "not repeat the analysis if that is the case. If wanting to "
            "overwrite existing files make sure this flag is not given."
        ),
    )

    dataparser = parser.add_argument_group("Data inputs")
    dataparser.add(
        "--starttime",
        required=True,
        type=int,
        help=("The start time of the data to be heterodyned in GPS seconds."),
    )
    dataparser.add(
        "--endtime",
        required=True,
        type=int,
        help=("The end time of the data to be heterodyned in GPS seconds."),
    )
    dataparser.add(
        "--stride",
        default=3600,
        type=int,
        help=(
            "The number of seconds to stride through the data (i.e., this "
            "number of seconds of data will be read in in one go), Defaults "
            "to 3600."
        ),
    )
    dataparser.add(
        "--detector",
        required=True,
        type=str,
        help=("The name of the detectors for which the data is to be " "heterodyned."),
    )
    dataparser.add(
        "--frametype",
        type=str,
        help=(
            'The "frame type" name of the data to be heterodyned. If this '
            "is not given the correct data set will be attempted to be found "
            "using the channel name."
        ),
    )
    dataparser.add(
        "--channel",
        required=True,
        type=str,
        help=(
            'The "channel" within the gravitational-wave data file(s) '
            '(either a GW frame ".gwf", or HDF5 file) containing the strain '
            "data to be heterodyned. The channel name should contain the "
            "detector name prefix as the first two characters followed by a "
            'colon, e.g., "L1:GWOSC-4KHZ_R1_STRAIN"'
        ),
    )
    dataparser.add(
        "--host",
        type=str,
        help=(
            "The server name for finding the gravitational-wave data files. "
            'Use "datafind.ligo.org:443" for open data available via CVMFS. '
            "To use open data available from the GWOSC use "
            '"https://www.gw-openscience.org".'
        ),
    )
    dataparser.add(
        "--outputframecache",
        type=str,
        help=(
            "If given this should give a file path to which a list of "
            "gravitational-wave data file paths, as found by the code, will "
            "be written. If not given then the file list will not be output."
        ),
    )
    dataparser.add(
        "--appendframecache",
        action="store_true",
        default=False,
        help=(
            "If writing out the frame cache to a file, set this to True to "
            "append to the file rather than overwriting. Default is False."
        ),
    )
    dataparser.add(
        "--framecache",
        help=(
            "Provide a pregenerate cache of gravitational-wave files, either "
            "as a single file, or a list of files. If a list, this should be "
            "in the form of a Python list, surrounded by quotation marks, "
            "e.g., \"['file1.lcf','file2.lcf']\"."
        ),
    )
    dataparser.add(
        "--heterodyneddata",
        help=(
            "A string, or dictionary of strings, containing the full file "
            "path, or directory path, pointing the the location of "
            "pre-heterodyned data. For a single pulsar a file path can be "
            "given. For multiple pulsars a directory containing heterodyned "
            "files (in HDF5 or txt format) can be given provided that within "
            "it the file names contain the pulsar names as supplied in the "
            'file input with "--pulsarfiles". Alternatively, a dictionary '
            "can be supplied, keyed on the pulsar name, containing a single "
            "file path or a directory path as above. If supplying a "
            "directory, it can contain multiple heterodyned files for a each "
            "pulsar and all will be used. If giving a dictionary it should be "
            "surrounded by quotation marks."
        ),
    )

    segmentparser = parser.add_argument_group("Analysis segment inputs")
    segmentparser.add(
        "--segmentlist",
        help=(
            "Provide a list of data segment start and end times, as tuple "
            "pairs in the list, or an ASCII text file containing the "
            "segment start and end times in two columns. If a list, this "
            "should be in the form of a Python list, surrounded by quotation "
            'marks, e.g., "[(900000000,900086400),(900100000,900186400)]".'
        ),
    )
    segmentparser.add(
        "--includeflags",
        help=(
            "If not providing a segment list then give a string, or list of "
            "strings, giving the data DQ flags that will be used to generate "
            "a segment list. Lists should be surrounded by quotation marks, "
            "e.g., \"['L1:DMT-ANALYSIS_READY:1']\"."
        ),
    )
    segmentparser.add(
        "--excludeflags",
        help=(
            "A string, or list of strings, giving the data DQ flags to "
            "when generating a segment list. Lists should be surrounded by "
            "quotation marks."
        ),
    )
    segmentparser.add(
        "--outputsegmentlist",
        type=str,
        help=(
            "If generating a segment list it will be output to the file "
            "specified by this argument."
        ),
    )
    segmentparser.add(
        "--appendsegmentlist",
        action="store_true",
        default=False,
        help=(
            "If generating a segment list set this to True to append to the "
            'file specified by "--outputsegmentlist" rather than '
            "overwriting. Default is False."
        ),
    )
    segmentparser.add(
        "--segmentserver",
        type=str,
        help=("The segment database URL."),
    )

    pulsarparser = parser.add_argument_group("Pulsar inputs")
    pulsarparser.add(
        "--pulsarfiles",
        help=(
            "This specifies the pulsars for which to heterodyne the data. It "
            "can be either i) a string giving the path to an individual "
            "pulsar TEMPO(2)-style parameter file, ii) a string giving the "
            "path to a directory containing multiple TEMPO(2)-style parameter "
            "files (the path will be recursively searched for any file with "
            'the extension ".par"), iii) a list of paths to individual '
            "pulsar parameter files, iv) a dictionary containing paths to "
            "individual pulsars parameter files keyed to their names. If "
            "providing a list or dictionary it should be surrounded by "
            "quotation marks."
        ),
    )
    pulsarparser.add(
        "--pulsars",
        help=(
            "You can analyse only particular pulsars from those specified by "
            'parameter files found through the "--pulsarfiles" argument by '
            "passing a string, or list of strings, with particular pulsars "
            "names to use."
        ),
    )

    outputparser = parser.add_argument_group("Data output inputs")
    outputparser.add(
        "--output",
        help=(
            "The base directory into which the heterodyned results will be "
            "output. To specify explicit directory paths for individual "
            "pulsars this can be a dictionary of directory paths keyed to the "
            'pulsar name (in which case the "--label" argument will be used '
            "to set the file name), or full file paths, which will be used in "
            'place of the "--label" argument.'
        ),
    )
    outputparser.add(
        "--label",
        help=(
            "The output format for the heterdyned data files. These can be "
            'format strings containing the keywords "psr" for the pulsar '
            'name, "det" for the detector, "freqfactor" for the rotation '
            'frequency scale factor used, "gpsstart" for the GPS start '
            'time, and "gpsend" for the GPS end time. The extension should '
            'be given as ".hdf", ".h5", or ".hdf5". E.g., the default '
            'is "heterodyne_{psr}_{det}_{freqfactor}_{gpsstart}-{gpsend}.hdf".'
        ),
    )

    heterodyneparser = parser.add_argument_group("Heterodyne inputs")
    heterodyneparser.add(
        "--filterknee",
        type=float,
        help=(
            "The knee frequency (Hz) of the low-pass filter applied after "
            "heterodyning the data. This should only be given when "
            "heterodying raw strain data and not if re-heterodyning processed "
            "data. Default is 0.5 Hz."
        ),
    )
    heterodyneparser.add(
        "--resamplerate",
        type=float,
        required=True,
        help=(
            "The rate in Hz at which to resample the data (via averaging) "
            "after application of the heterodyne (and filter if applied)."
        ),
    )
    heterodyneparser.add(
        "--freqfactor",
        type=float,
        help=(
            "The factor applied to the pulsars rotational parameters when "
            "defining the gravitational-wave phase evolution. For example, "
            "the default value of 2 multiplies the phase evolution by 2 under "
            "the assumption of a signal emitted from the l=m=2 quadrupole "
            "mode of a rigidly rotating triaxial neutron star."
        ),
    )
    heterodyneparser.add(
        "--crop",
        type=int,
        help=(
            "The number of seconds to crop from the start and end of data "
            "segments to remove filter impulse effects and issues prior to "
            "lock-loss. Default is 60 seconds."
        ),
    )
    heterodyneparser.add(
        "--includessb",
        action="store_true",
        default=False,
        help=(
            "Set this flag to include removing the modulation of the signal due to "
            "Solar System motion and relativistic effects (e.g., Roemer, "
            "Einstein, and Shapiro delay) during the heterodyne."
        ),
    )
    heterodyneparser.add(
        "--includebsb",
        action="store_true",
        default=False,
        help=(
            "Set this flag to include removing the modulation of the signal "
            "due to binary system motion and relativistic effects during the "
            'heterodyne. To use this "--includessb" must also be set.'
        ),
    )
    heterodyneparser.add(
        "--includeglitch",
        action="store_true",
        default=False,
        help=(
            "Set this flag to include removing the effects of the phase "
            "evolution of any modelled pulsar glitches during the heterodyne."
        ),
    )
    heterodyneparser.add(
        "--includefitwaves",
        action="store_true",
        default=False,
        help=(
            "Set this to True to include removing the phase evolution of a "
            "series of sinusoids designed to model low-frequency timing noise "
            "in the pulsar signal during the heterodyne."
        ),
    )

    ephemerisparser = parser.add_argument_group("Solar system ephemeris inputs")
    ephemerisparser.add(
        "--earthephemeris",
        help=(
            'A dictionary, keyed to ephemeris names, e.g., "DE405", pointing '
            "to the location of a file containing that ephemeris for the "
            "Earth. The dictionary must be supplied within quotation marks, "
            "e.g., \"{'DE436':'earth_DE436.txt'}\". If a pulsar requires a "
            "specific ephemeris that is not provided in this dictionary, then "
            "the code will automatically attempt to find or download the "
            "required file if available."
        ),
    )
    ephemerisparser.add(
        "--sunephemeris",
        help=(
            'A dictionary, keyed to ephemeris names, e.g., "DE405", pointing '
            "to the location of a file containing that ephemeris for the "
            "Sun. If a pulsar requires a specific ephemeris that is not "
            "provided in this dictionary, then the code will automatically "
            "attempt to find or download the required file if available."
        ),
    )
    ephemerisparser.add(
        "--timeephemeris",
        help=(
            "A dictionary, keyed to time system name, which can be either "
            '"TCB" or "TDB", pointing to the location of a file containing '
            "that ephemeris for that time system. If a pulsar requires a "
            "specific ephemeris that is not provided in this dictionary, then "
            "the code will automatically attempt to find or download the "
            "required file if available."
        ),
    )

    return parser


def heterodyne(**kwargs):
    """
    Run heterodyne within Python. See the
    `class::~cwinpy.heterodyne.Heterodyne` class for the required arguments.

    Returns
    -------
    het: `class::~cwinpy.heterodyne.Heterodyne`
        The heterodyning class object.
    """

    if "cli" in kwargs or "config" in kwargs:
        if "cli" in kwargs:
            kwargs.pop("cli")

        # get command line arguments
        parser = create_heterodyne_parser()

        # parse config file or command line arguments
        if "config" in kwargs:
            cliargs = ["--config", kwargs["config"]]
        else:
            cliargs = sys.argv[1:]

        try:
            args, _ = parse_args(cliargs, parser)
        except BilbyPipeError as e:
            raise IOError("{}".format(e))

        # convert args to a dictionary
        hetkwargs = vars(args)

        if "config" in kwargs:
            # update with other keyword arguments
            hetkwargs.update(kwargs)
            hetkwargs.pop("config")
    else:
        hetkwargs = kwargs

    # check non-standard arguments that could be Python objects
    nsattrs = [
        "framecache",
        "heterodyneddata",
        "segmentlist",
        "includeflags",
        "excludeflags",
        "pulsarfiles",
        "pulsars",
        "output",
        "earthephemeris",
        "sunephemeris",
        "timeephemeris",
    ]
    for attr in nsattrs:
        value = hetkwargs.pop(attr, None)

        if isinstance(value, str):
            # check whether the value can be evaluated as a Python object
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass

            hetkwargs[attr] = value
        elif value is not None:
            hetkwargs[attr] = value

    # set up the run
    het = Heterodyne(**hetkwargs)

    # heterodyne the data
    signal.signal(signal.SIGALRM, handler=sighandler)
    signal.alarm(hetkwargs.pop("periodic_restart_time"))
    het.heterodyne()

    return het


def heterodyne_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_heterodyne`` script. This just calls
    :func:`cwinpy.heterodyne.heterodyne`, but does not return any objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = heterodyne(**kwargs)


class HeterodyneDAGRunner(object):
    """
    Set up and run the heterodyne DAG.

    Parameters
    ----------
    config: :class:`configparser.ConfigParser`
          A :class:`configparser.ConfigParser` object with the analysis setup
          parameters.
    """

    def __init__(self, config, **kwargs):
        # create and build the dag
        self.create_dag(config, **kwargs)

        if self.submitdag:
            if self.build:
                self.dag.build_submit(submit_options=self.submit_options)
            else:
                self.dag.submit_dag(self.submit_options)

    def create_dag(self, config, **kwargs):
        """
        Create the HTCondor DAG from the configuration parameters.

        Parameters
        ----------
        config: :class:`configparser.ConfigParser`
            A :class:`configparser.ConfigParser` object with the analysis setup
            parameters.
        """

        if not isinstance(config, configparser.ConfigParser):
            raise TypeError("'config' must be a ConfigParser object")

        inputs = HeterodyneInput(config)

        if "dag" in kwargs:
            # get a previously created DAG if given (for example for a full
            # analysis pipeline)
            self.dag = kwargs["dag"]
        else:
            self.dag = Dag(inputs)

        # get whether to build the dag
        self.build = config.getboolean("dag", "build", fallback=True)

        # get whether to automatically submit the dag
        self.submitdag = config.getboolean("dag", "submitdag", fallback=False)

        # get any additional submission options
        self.submit_options = config.get("dag", "submit_options", fallback=None)

        # create configurations for each cwinpy_heterodyne job
        if not config.has_section("heterodyne"):
            raise IOError("Configuration file must have a [heterodyne] section.")

        # create Heterodyne object to get pulsar parameter file information
        pulsarfiles = config.get("heterodyne", "pulsarfiles", fallback=None)
        pulsars = config.get("heterodyne", "pulsars", fallback=None)
        if pulsarfiles is None:
            raise ValueError("A set of pulsar parameter files must be supplied")
        het = Heterodyne(pulsarfiles=self.eval(pulsarfiles), pulsars=self.eval(pulsars))

        detectors = self.eval(config.get("heterodyne", "detectors", fallback=None))
        if isinstance(detectors, str):
            detectors = [detectors]  # make into a list
        elif detectors is None:
            raise ValueError("At least one detector must be supplied")

        freqfactors = self.eval(
            config.get("heterodyne", "freqfactors", fallback="[2.0]")
        )
        if isinstance(freqfactors, (int, float)):
            freqfactors = [freqfactors]  # make into a list

        # get times of data to analyse
        fullstarttimes = self.eval(
            config.get("heterodyne", "starttimes", fallback=None)
        )
        if isinstance(fullstarttimes, dict):
            if sorted(detectors) != sorted(fullstarttimes.keys()):
                raise ValueError("Start times must be specified for all detectors")
            for key, value in fullstarttimes.copy().items():
                if isinstance(value, int):
                    fullstarttimes[key] = [value]  # convert values to lists
                elif not isinstance(value, list):
                    raise TypeError("Must have a list of start times for a detector")
        elif isinstance(fullstarttimes, int):
            fullstarttimes = {
                det: [fullstarttimes] for det in detectors
            }  # convert to dict
        else:
            raise ValueError("Start times must be given")

        fullendtimes = self.eval(config.get("heterodyne", "endtimes", fallback=None))
        if isinstance(fullendtimes, dict):
            if sorted(detectors) != sorted(fullendtimes.keys()):
                raise ValueError("End times must be specified for all detectors")
            for key, value in fullendtimes.copy().items():
                if isinstance(value, int):
                    fullendtimes[key] = [value]  # convert values to lists
                elif not isinstance(value, list):
                    raise TypeError("Must have a list of end times for a detector")
        elif isinstance(fullendtimes, int):
            fullendtimes = {det: [fullendtimes] for det in detectors}  # convert to dict
        else:
            raise ValueError("End times must be given")

        for det in detectors:
            if len(fullendtimes[det]) != len(fullstarttimes[det]):
                raise ValueError("Inconsistent numbers of start and end times")

        stride = config.getint("heterodyne", "stride", fallback=None)
        ntimejobs = config.getint("heterodyne", "ntimejobs", fallback=1)

        # get frame data information
        frametypes = self.eval(config.get("heterodyne", "frametypes", fallback=None))
        framecaches = self.eval(config.get("heterodyne", "framecaches", fallback=None))
        channels = self.eval(config.get("heterodyne", "channels", fallback=None))
        host = config.get("heterodyne", "host", fallback=None)
        framedata = {det: [] for det in detectors}
        if frametypes is None and framecaches is None:
            raise ValueError("Frame types of frame cache files must be supplied")
        for finfo, fname in zip(
            [frametypes, framecaches, channels],
            ["frametypes", "framecaches", "channels"],
        ):
            if finfo is not None:
                # set frame types/caches
                if isinstance(finfo, str) and len(detectors) == 1:
                    finfo = {det: [finfo] for det in detectors}
                elif isinstance(finfo, dict):
                    for key, value in finfo.copy.items():
                        if isinstance(value, str):
                            finfo[key] = [value] * len(fullstarttimes[key])
                        elif not isinstance(value, list):
                            raise TypeError("Must have a list of {}".format(fname))
                else:
                    raise TypeError("{} should be a dictionary".format(fname))

        # get segment information
        segmentserver = config.get("heterodyne", "segmentserver", fallback=None)
        segmentlists = self.eval(
            config.get("heterodyne", "segmentlists", fallback=None)
        )
        includeflags = self.eval(
            config.get("heterodyne", "includeflags", fallback=None)
        )
        # excludeflags = self.eval(
        #    config.get("heterodyne", "excludeflags", fallback=None)
        # )
        if segmentlists is None and includeflags is None:
            raise ValueError(
                "Segment lists of segment data quality flags must be supplied"
            )

        # get all the split times
        if ntimejobs == 1:
            starttimes = fullstarttimes
            endtimes = fullendtimes

            for det in detectors:
                for i in range(len(fullstarttimes[det])):
                    frinfo = {}
                    if frametypes is not None:
                        frinfo["frametype"] = frametypes[det][i]
                    else:
                        frinfo["framecache"] = framecaches[det][i]
                    frinfo["channel"] = channels[det][i]
                    framedata[det].append(frinfo.copy())
        elif ntimejobs > 1:
            starttimes = {det: [] for det in detectors}
            endtimes = {det: [] for det in detectors}

            totaltimes = {
                det: sum(
                    fullendtimes[det][i] - fullstarttimes[det][i]
                    for i in range(len(fullendtimes[det]))
                )
                for det in detectors
            }
            for det in detectors:
                tstep = int(np.ceil(totaltimes / ntimejobs))

                idx = 0
                for starttime, endtime in zip(fullstarttimes[det], fullendtimes[det]):
                    curstart = starttime
                    while curstart < endtime:
                        curend = curstart + tstep
                        starttimes[det].append(curstart)
                        endtimes[det].append(min([curend, endtime]))
                        curstart = curend
                        frinfo = {}
                        if frametypes is not None:
                            frinfo["frametype"] = frametypes[det][idx]
                        else:
                            frinfo["framecache"] = framecaches[det][idx]
                        frinfo["channel"] = channels[det][idx]
                        framedata[det].append(frinfo.copy())
                    idx += 1
        else:
            raise ValueError("Number of jobs must be a positive integer")

        # get number over which to split up pulsars
        npulsarjobs = config.getint("heterodyne", "npulsarjobs", fallback=1)
        pulsargroups = []
        if npulsarjobs == 1 or len(het.pulsars) == 1:
            pulsargroups.append(het.pulsars)
        else:
            pstep = int(np.ceil(len(het.pulsars) / npulsarjobs))
            for i in range(npulsarjobs):
                pulsargroups.append(het.pulsars[pstep * i : pstep * (i + 1)])

        # set whether to perform the heterodyne in 1 or two stages
        stages = config.getint("heterodyne", "stages", fallback=1)
        if stages not in [1, 2]:
            raise ValueError("Stages must either be 1 or 2")

        # get the resample rate(s)
        if stages == 1:
            resamplerate = [
                self.eval(
                    config.get("heterodyne", "resamplerate", fallback="1.0 / 60.0")
                )
            ]
        else:
            resamplerate = self.eval(
                config.get("heterodyne", "resamplerate", fallback="[1.0, 1.0 / 60.0]")
            )

        # set the components of the signal modulation, i.e., solar system,
        # binary system, to include in the heterodyne stages. By default a
        # single stage heterodyne will include all components and a two stage
        # heteodyne will include no components in the first stage, but all
        # components in the second stage. If suppling different values for a
        # two stage process use lists
        if stages == 1:
            includessb = [config.getboolean("heterodyne", "includessb", fallback=True)]
            includebsb = [config.getboolean("heterodyne", "includebsb", fallback=True)]
            includeglitch = [
                config.getboolean("heterodyne", "includeglitch", fallback=True)
            ]
            includefitwaves = [
                config.getboolean("heterodyne", "includefitwaves", fallback=True)
            ]
        else:
            includessb = self.eval(
                config.getboolean("heterodyne", "includessb", fallback="[False, True]")
            )
            includebsb = self.eval(
                config.getboolean("heterodyne", "includebsb", fallback="[False, True]")
            )
            includeglitch = self.eval(
                config.getboolean(
                    "heterodyne", "includeglitch", fallback="[False, True]"
                )
            )
            includefitwaves = self.eval(
                config.getboolean(
                    "heterodyne", "includefitwaves", fallback="[False, True]"
                )
            )

        # create jobs
        self.hetnodes = []
        self.pulsar_nodes = {
            psr: [] for psr in het.pulsars
        }  # dictionary to contain all nodes for a given pulsar

        # loop over sets of pulsars
        for pgroup in pulsargroups:
            self.hetnodes.append([])
            # loop over frequency factors
            for ff in freqfactors:
                # loop over each detector
                for det in detectors:
                    # loop over times
                    idx = 0
                    for starttime, endtime in zip(starttimes[det], endtimes[det]):
                        configdict = {}

                        configdict["starttime"] = starttime
                        configdict["endtime"] = endtimes
                        configdict["detector"] = det
                        configdict["freqfactor"] = ff
                        configdict["resamplerate"] = resamplerate[0]

                        # set frame data info
                        configdict.update(framedata[det][idx])
                        if host is not None:
                            configdict["host"] = host

                        if stride is not None:
                            configdict["stride"] = stride

                        # set segment data info
                        configdict["segmentserver"] = segmentserver

                        configdict["pulsarfiles"] = pulsarfiles
                        configdict["pulsars"] = pgroup

                        # set whether to include modulations
                        configdict["includessb"] = includessb[0]
                        configdict["includebsb"] = includebsb[0]
                        configdict["includeglitch"] = includeglitch[0]
                        configdict["includefitwaves"] = includefitwaves[0]

                        self.hetnodes[-1].append(
                            HeterodyneNode(inputs, configdict.copy(), self.dag)
                        )

                        # put nodes into dictionary for each pulsar
                        if stages == 1:
                            for psr in pgroup:
                                self.pulsar_nodes[psr].append(self.hetnodes[-1][-1])

                        idx += 1

        # need to check whether doing fine heterodyne - in this case need to create new jobs on a per pulsar basis
        if stages == 2:
            for i, pgroup in enumerate(pulsargroups):
                for psr in pgroup:
                    for ff in freqfactors:
                        for det in detectors:
                            configdict = {}
                            configdict["starttime"] = starttimes[det][0]
                            configdict["endtime"] = endtimes[det][-1]
                            configdict["detector"] = det
                            configdict["freqfactor"] = ff
                            configdict["pulsars"] = psr
                            configdict["resamplerate"] = resamplerate[-1]

                            # include all modulations
                            configdict["includessb"] = includessb[-1]
                            configdict["includebsb"] = includebsb[-1]
                            configdict["includeglitch"] = includeglitch[-1]
                            configdict["includefitwaves"] = includefitwaves[-1]

                            self.pulsar_nodes[psr].append(
                                HeterodyneNode(
                                    inputs,
                                    configdict.copy(),
                                    self.dag,
                                    generation_node=self.hetnodes[i],
                                )
                            )

        if self.build:
            self.dag.build()

    def eval(self, arg):
        """
        Try and evaluate a string using :func:`ast.literal_eval`.

        Parameters
        ----------
        arg: str
            A string to be evaluated.

        Returns
        -------
        object:
            The evaluated object, or original string, if not able to be evaluated.
        """

        # copy of string
        newobj = str(arg)

        try:
            newobj = ast.literal_eval(newobj)
        except (ValueError, SyntaxError):
            pass

        return newobj


class HeterodyneInput(Input):
    def __init__(self, cf):
        """
        Class that sets inputs for the DAG and analysis node generation.

        Parameters
        ----------
        cf: :class:`configparser.ConfigParser`
            The configuration file for the DAG set up.
        """

        self.config = cf
        self.submit = cf.getboolean("dag", "submitdag", fallback=False)
        self.transfer_files = cf.getboolean("dag", "transfer-files", fallback=True)
        self.osg = cf.getboolean("dag", "osg", fallback=False)
        self.label = cf.get("dag", "name", fallback="cwinpy_pe")

        # see bilby_pipe MainInput class
        self.scheduler = cf.get("dag", "scheduler", fallback="condor")
        self.scheduler_args = cf.get("dag", "scheduler_args", fallback=None)
        self.scheduler_module = cf.get("dag", "scheduler_module", fallback=None)
        self.scheduler_env = cf.get("dag", "scheduler_env", fallback=None)
        self.scheduler_analysis_time = cf.get(
            "dag", "scheduler_analysis_time", fallback="7-00:00:00"
        )

        self.outdir = cf.get("run", "basedir", fallback=os.getcwd())

        self.universe = cf.get("job", "universe", fallback="vanilla")
        self.getenv = cf.getboolean("job", "getenv", fallback=False)
        self.pe_log_directory = cf.get(
            "job", "log", fallback=os.path.join(os.path.abspath(self._outdir), "log")
        )
        self.request_memory = cf.get("job", "request_memory", fallback="4 GB")
        self.request_cpus = cf.getint("job", "request_cpus", fallback=1)
        self.accounting = cf.get(
            "job", "accounting_group", fallback="cwinpy"
        )  # cwinpy is a dummy tag
        self.accounting_user = cf.get("job", "accounting_group_user", fallback=None)
        requirements = cf.get("job", "requirements", fallback=None)
        self.requirements = [requirements] if requirements else []
        self.retry = cf.getint("job", "retry", fallback=0)
        self.notification = cf.get("job", "notification", fallback="Never")
        self.email = cf.get("job", "email", fallback=None)
        self.condor_job_priority = cf.getint("job", "condor_job_priority", fallback=0)

        # needs to be set for the bilby_pipe Node initialisation, but is not a
        # requirement for cwinpy_heterodyne
        self.online_pe = False
        self.extra_lines = []
        self.run_local = False

    @property
    def submit_directory(self):
        subdir = self.config.get(
            "dag", "submit", fallback=os.path.join(self._outdir, "submit")
        )
        check_directory_exists_and_if_not_mkdir(subdir)
        return subdir

    @property
    def initialdir(self):
        if hasattr(self, "_initialdir"):
            if self._initialdir is not None:
                return self._initialdir
            else:
                return os.getcwd()
        else:
            return os.getcwd()

    @initialdir.setter
    def initialdir(self, initialdir):
        if isinstance(initialdir, str):
            self._initialdir = initialdir
        else:
            self._initialdir = None


class HeterodyneNode(Node):
    def __init__(self, inputs, configdict, dag, generation_node=None):
        super().__init__(inputs)
        self.dag = dag

        self.request_cpus = inputs.request_cpus
        self.retry = inputs.retry
        self.getenv = inputs.getenv
        self._universe = inputs.universe

        starttime = configdict["starttime"]
        endtime = configdict["endtime"]
        detector = configdict["detector"]
        freqfactor = configdict["freqfactor"]
        pulsar = configdict.get("pulsar", None)

        psrstring = "" if pulsar is None else "{}_".format(pulsar.replace("+", "plus"))

        # resdir = inputs.config.get("pe", "results", fallback="results")
        # self.psrbase = os.path.join(inputs.outdir, resdir, psrname)
        # if self.inputs.n_parallel > 1:
        #    self.resdir = os.path.join(self.psrbase, "par{}".format(parallel_idx))
        # else:
        #    self.resdir = self.psrbase

        check_directory_exists_and_if_not_mkdir(self.resdir)
        configdict["outdir"] = self.resdir

        # job name prefix
        jobname = inputs.config.get("job", "name", fallback="cwinpy_heterodyne")
        self.base_job_name = "{}_{}{}_{}_{}-{}".format(
            jobname, psrstring, detector, freqfactor, starttime, endtime
        )
        self.job_name = self.base_job_name

        # output the configuration file
        configdir = inputs.config.get("heterodyne", "config", fallback="configs")
        configlocation = os.path.join(inputs.outdir, configdir)
        check_directory_exists_and_if_not_mkdir(configlocation)
        configfile = os.path.join(
            configlocation,
            "{}{}_{}_{}-{}.ini".format(
                psrstring, detector, freqfactor, starttime, endtime
            ),
        )

        self.setup_arguments(
            add_ini=False, add_unknown_args=False, add_command_line_args=False
        )

        # add files for transfer
        if self.inputs.transfer_files or self.inputs.osg:
            tmpinitialdir = self.inputs.initialdir

            self.inputs.initialdir = self.resdir

            input_files_to_transfer = [
                self._relative_topdir(configfile, self.inputs.initialdir)
            ]

            # make paths relative
            for key in ["par_file", "inj_par", "data_file_1f", "data_file_2f", "prior"]:
                if key in list(configdict.keys()):
                    if key in ["data_file_1f", "data_file_2f"]:
                        for detkey in configdict[key]:
                            input_files_to_transfer.append(
                                self._relative_topdir(
                                    configdict[key][detkey], self.inputs.initialdir
                                )
                            )

                            # set to use only file as the transfer directory is flat
                            configdict[key][detkey] = os.path.basename(
                                configdict[key][detkey]
                            )
                    else:
                        input_files_to_transfer.append(
                            self._relative_topdir(
                                configdict[key], self.inputs.initialdir
                            )
                        )

                        # set to use only file as the transfer directory is flat
                        configdict[key] = os.path.basename(configdict[key])

            configdict["outdir"] = "results/"

            # add output directory to inputs in case resume file exists
            input_files_to_transfer.append(".")

            self.extra_lines.extend(
                self._condor_file_transfer_lines(
                    list(set(input_files_to_transfer)), [configdict["outdir"]]
                )
            )

            self.arguments.add("config", os.path.basename(configfile))
        else:
            tmpinitialdir = None
            self.arguments.add("config", configfile)

        self.extra_lines.extend(self._checkpoint_submit_lines())

        # add accounting user
        if self.inputs.accounting_user is not None:
            self.extra_lines.append(
                "accounting_group_user = {}".format(self.inputs.accounting_user)
            )

        parseobj = DefaultConfigFileParser()
        with open(configfile, "w") as fp:
            fp.write(parseobj.serialize(configdict))

        self.process_node()

        # reset initial directory
        if tmpinitialdir is not None:
            self.inputs.initialdir = tmpinitialdir

        if generation_node is not None:
            # for fine heterodyne, add previous jobs as parent
            if isinstance(generation_node, Node):
                self.job.add_parent(generation_node.job)
            elif isinstance(generation_node, list):
                self.job.add_parents(
                    [gnode.job for gnode in generation_node if isinstance(gnode, Node)]
                )

    @property
    def executable(self):
        jobexec = self.inputs.config.get(
            "job", "executable", fallback="cwinpy_heterodyne"
        )
        return self._get_executable_path(jobexec)

    @property
    def request_memory(self):
        return self.inputs.request_memory

    @property
    def log_directory(self):
        check_directory_exists_and_if_not_mkdir(self.inputs.pe_log_directory)
        return self.inputs.pe_log_directory

    @property
    def result_directory(self):
        """ The path to the directory where result output will be stored """
        check_directory_exists_and_if_not_mkdir(self.resdir)
        return self.resdir

    @staticmethod
    def _relative_topdir(path, reference):
        """Returns the top-level directory name of a path relative
        to a reference
        """
        try:
            return os.path.relpath(path, reference)
        except ValueError as exc:
            exc.args = ("cannot format {} relative to {}".format(path, reference),)
            raise
