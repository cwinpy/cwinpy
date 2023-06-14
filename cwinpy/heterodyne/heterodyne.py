"""
Run heterodyne pre-processing of gravitational-wave data.
"""

import ast
import configparser
import copy
import os
import shutil
import signal
import sys
import tempfile
from argparse import ArgumentParser

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from configargparse import ArgumentError
from htcondor.dags import DAG, write_dag
from simpleeval import EvalWithCompoundTypes, NameNotDefined
from solar_system_ephemerides.paths import JPLDE

import cwinpy

from ..condor import submit_dag
from ..condor.hetnodes import HeterodyneLayer, MergeLayer
from ..cwinpyargparser import CWInPyArgParser
from ..data import HeterodynedData
from ..info import (
    ANALYSIS_SEGMENTS,
    CVMFS_GWOSC_DATA_SERVER,
    CVMFS_GWOSC_DATA_TYPES,
    CVMFS_GWOSC_FRAME_CHANNELS,
    HW_INJ,
    HW_INJ_RUNTIMES,
    HW_INJ_SEGMENTS,
    RUNTIMES,
    is_hwinj,
)
from ..parfile import PulsarParameters
from ..utils import (
    LAL_BINARY_MODELS,
    check_for_tempo2,
    draw_ra_dec,
    get_psr_name,
    int_to_alpha,
    overlap,
    parse_args,
    sighandler,
)
from .base import Heterodyne, generate_segments, remote_frame_cache


def create_heterodyne_parser():
    """
    Create the argument parser.
    """

    description = """\
A script to heterodyne raw gravitational-wave strain data based on the \
expected evolution of the gravitational-wave signal from a set of pulsars."""

    parser = CWInPyArgParser(
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
        default=14400,
        type=int,
        help=(
            "Time after which the job will be self-evicted with code 130. "
            "After this, condor will restart the job. Default is 14400s. "
            "This is used to decrease the chance of HTCondor hard evictions."
        ),
    )
    parser.add(
        "--overwrite",
        action="store_true",
        default=False,
        help=(
            "Set this flag to make sure any previously generated heterodyned "
            'files are overwritten. By default the analysis will "resume" '
            "from where it left off (by checking whether output files, as set "
            'using "--output" and "--label" arguments, already exist), such '
            "as after forced Condor eviction for checkpointing purposes. "
            "Therefore, this flag is needs to be explicitly given (the "
            "default is False) if not wanting to use resume and overwrite "
            "existing files."
        ),
    )

    dataparser = parser.add_argument_group("Data inputs")
    dataparser.add(
        "--starttime",
        required=True,
        help=(
            "The start time of the data to be heterodyned in GPS seconds or "
            "as an ISO format date string."
        ),
    )
    dataparser.add(
        "--endtime",
        required=True,
        help=(
            "The end time of the data to be heterodyned in GPS seconds or as "
            "an ISO format date string."
        ),
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
        help=("The name of the detectors for which the data is to be heterodyned."),
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
            '"https://gwosc.org".'
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
            "Provide a pregenerated cache of gravitational-wave files, either "
            "as a single file, or a list of files. Alternatively, you can "
            "supply a directory containing the files (which will be "
            "searched recursively for gwf and then hdf5 files), which should "
            'be used in conjunction with the "frametype" argument. If giving '
            "a list, this should be in the form of a Python list, surrounded "
            "by quotation marks, e.g., \"['file1.lcf','file2.lcf']\"."
        ),
    )
    dataparser.add(
        "--strictdatarequirement",
        help=(
            "Set this to True to strictly require that all frame data files "
            "that are requested can be read in and used. In the case that a "
            "frame cannot be read (i.e., it is corrupted or unavailable) then "
            "the code will raise an exception and exit. If this is False, "
            "which is the default, the code will just ignore that frame and "
            "move on the next, while printing out a message about the ignored "
            "data."
        ),
        action="store_true",
        default=False,
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
            "Provide an ASCII text file containing a list of science segment "
            "start and end times in two columns."
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
    segmentparser.add("--segmentserver", type=str, help=("The segment database URL."))

    pulsarparser = parser.add_argument_group("Pulsar inputs")
    pulsarparser.add(
        "--pulsarfiles",
        action="append",
        help=(
            "This specifies the pulsars for which to heterodyne the data. It "
            "can be either i) a string giving the path to an individual "
            "pulsar Tempo(2)-style parameter file, ii) a string giving the "
            "path to a directory containing multiple Tempo(2)-style parameter "
            "files (the path will be recursively searched for any file with "
            'the extension ".par"), iii) a list of paths to individual '
            "pulsar parameter files, iv) a dictionary containing paths to "
            "individual pulsars parameter files keyed to their names. If "
            "instead, pulsar names are given rather than parameter files it "
            "will attempt to extract an ephemeris for those pulsars from the "
            "ATNF pulsar catalogue. If such ephemerides are available then "
            "they will be used (notification will be given when this is "
            "these cases). If providing a list or dictionary it should be "
            "surrounded by quotation marks."
        ),
    )
    pulsarparser.add(
        "--pulsars",
        action="append",
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
            'place of the "--label" argument. If not given then the current'
            "working directory will be used."
        ),
    )
    outputparser.add(
        "--label",
        help=(
            "The output format for the heterodyned data files. These can be "
            'format strings containing the keywords "psr" for the pulsar '
            'name, "det" for the detector, "freqfactor" for the rotation '
            'frequency scale factor used, "gpsstart" for the GPS start '
            'time, and "gpsend" for the GPS end time. The extension should '
            'be given as ".hdf", ".h5", or ".hdf5". E.g., the default '
            'is "heterodyne_{psr}_{det}_{freqfactor}_{gpsstart}-{gpsend}.hdf5".'
        ),
    )

    heterodyneparser = parser.add_argument_group("Heterodyne inputs")
    heterodyneparser.add(
        "--filterknee",
        type=float,
        help=(
            "The knee frequency (Hz) of the low-pass filter applied after "
            "heterodyning the data. This should only be given when "
            "heterodyning raw strain data and not if re-heterodyning processed "
            "data. Default is 0.5 Hz."
        ),
    )
    heterodyneparser.add(
        "--filterorder",
        type=int,
        help=(
            "The order of the low-pass Butterworth filter applied after "
            "heterodyning the data. Default is 9."
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
    heterodyneparser.add(
        "--usetempo2",
        action="store_true",
        default=False,
        help=(
            "Set this to True to use Tempo2 (via libstempo) to calculate the "
            "signal phase evolution. For this to be used v2.4.2 or greater of "
            "libstempo must be installed. When using Tempo2 the "
            '"--earthephemeris" and "--sunephemeris" arguments do not need to '
            "be supplied. This can only be used when running the full "
            "heterodyne in one stage, but not for re-heterodyning previous "
            'data, as such all the "--include..." arguments will be assumed '
            "to be True."
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

    cfparser = parser.add_argument_group("Configuration inputs")
    cfparser.add(
        "--cwinpy-heterodyne-pipeline-config-file",
        help=(
            "A path to the cwinpy_heterodyne_pipeline configuration file can be "
            "supplied if this was has been used to setup the heterodyne job."
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

        args, _ = parse_args(cliargs, parser)

        # convert args to a dictionary
        hetkwargs = vars(args)

        if "config" in kwargs:
            # update with other keyword arguments
            hetkwargs.update(kwargs)
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
    ]
    for attr in nsattrs:
        value = str(hetkwargs.pop(attr, None))

        # check whether the value can be evaluated as a Python object
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        # if the value was a string within a string, e.g., '"[2.3]"',
        # evaluate again just in case it contains a Python object!
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass

        if value is not None:
            hetkwargs[attr] = value

    # check if pulsarfiles is a single entry list containing a dictionary
    if isinstance(hetkwargs["pulsarfiles"], list):
        if len(hetkwargs["pulsarfiles"]) == 1:
            try:
                value = ast.literal_eval(hetkwargs["pulsarfiles"][0])

                if isinstance(value, dict):
                    # switch to passing the dictionary
                    hetkwargs["pulsarfiles"] = value
            except (SyntaxError, ValueError):
                pass

    signal.signal(signal.SIGALRM, handler=sighandler)
    signal.alarm(hetkwargs.pop("periodic_restart_time", 14400))

    # remove any None values
    for key in hetkwargs.copy():
        if hetkwargs[key] is None:
            hetkwargs.pop(key)

    # convert "overwrite" to "resume"
    hetkwargs["resume"] = not hetkwargs.pop("overwrite", False)

    # remove "config" from hetkwargs
    if "config" in hetkwargs:
        hetkwargs.pop("config")

    # set up the run
    het = Heterodyne(**hetkwargs)

    # heterodyne the data
    het.heterodyne()

    return het


def heterodyne_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_heterodyne`` script. This just calls
    :func:`cwinpy.heterodyne.heterodyne`, but does not return any objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = heterodyne(**kwargs)


def create_heterodyne_merge_parser():
    """
    Create the argument parser for merging script.
    """

    description = "A script to merge multiple heterodyned data files."

    parser = CWInPyArgParser(
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
        "--heterodynedfiles",
        action="append",
        type=str,
        help=("A path, or list of paths, to heterodyned data files to merge together."),
    )
    parser.add(
        "--output", type=str, help=("The output file for the merged heterodyned data.")
    )
    parser.add(
        "--overwrite",
        action="store_true",
        help=("Set if wanting to overwrite an existing merged file."),
    )
    parser.add(
        "--remove",
        action="store_true",
        help=("Set if wanting to delete individual files being merged."),
    )

    return parser


def heterodyne_merge(**kwargs):
    """
    Merge the output of multiple heterodynes for a specific pulsar.

    Parameters
    ----------
    heterodynedfiles: str, list
        A string, or list of strings, giving the paths to heterodyned data
        files to be read in and merged
    output: str
        The output file name to write the data to. If not given then the data
        will not be output.
    overwrite: bool
        Set whether to overwrite an existing file. Defaults to False.
    remove: bool
        Set whether to remove the individual files that form the merged file.
        Defaults to False.

    Returns
    -------
    het: `class::~cwinpy.heterodyne.Heterodyne`
        The merged heterodyning class object.
    """

    if "cli" in kwargs:
        # get command line arguments
        parser = create_heterodyne_merge_parser()
        cliargs = sys.argv[1:]

        args, _ = parse_args(cliargs, parser)

        # convert args to a dictionary
        mergekwargs = vars(args)
    else:
        mergekwargs = kwargs

    if "heterodynedfiles" not in mergekwargs:
        raise ArgumentError("'heterodynedfiles' is a required argument")

    heterodynedfiles = mergekwargs["heterodynedfiles"]
    filelist = (
        heterodynedfiles if isinstance(heterodynedfiles, list) else [heterodynedfiles]
    )
    filelist = [hf for hf in filelist if os.path.isfile(hf)]

    if len(filelist) == 0:
        raise ValueError("None of the heterodyned files given exists!")

    # read in and merge all the files
    het = HeterodynedData.read(filelist)

    # write out the merged data file
    if "output" in mergekwargs:
        het.write(mergekwargs["output"], overwrite=mergekwargs.get("overwrite", False))

    if mergekwargs.get("remove", False):
        # remove the individual files
        for hf in filelist:
            os.remove(hf)

    return het


def heterodyne_merge_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_heterodyne_merge`` script. This just calls
    :func:`cwinpy.heterodyne.heterodyne_merge`, but does not return any
    objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = heterodyne_merge(**kwargs)


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

        dagsection = "heterodyne_dag" if config.has_section("heterodyne_dag") else "dag"

        if "dag" in kwargs:
            # get a previously created DAG if given (for example for a full
            # analysis pipeline)
            self.dag = kwargs["dag"]
        else:
            self.dag = DAG()

        # get whether to build the dag
        self.build = config.getboolean(dagsection, "build", fallback=True)

        # get the base directory
        self.basedir = config.get("run", "basedir", fallback=os.getcwd())

        # create configurations for each cwinpy_heterodyne job
        if not config.has_section("heterodyne"):
            raise IOError("Configuration file must have a [heterodyne] section.")

        # detectors to use
        detectors = self.eval(config.get("heterodyne", "detectors", fallback=None))
        if isinstance(detectors, str):
            detectors = [detectors]  # make into a list
        elif detectors is None:
            raise ValueError("At least one detector must be supplied")

        # get pulsar information
        pulsarfiles = self.eval(config.get("ephemerides", "pulsarfiles", fallback=None))
        pulsars = self.eval(config.get("ephemerides", "pulsars", fallback=None))
        if pulsarfiles is None:
            raise ValueError("A set of pulsar parameter files must be supplied")

        # output information
        outputdirs = self.eval(config.get("heterodyne", "outputdir", fallback=None))
        if not isinstance(outputdirs, list):
            outputdirs = [outputdirs]

        for i, outputdir in enumerate(copy.deepcopy(outputdirs)):
            if isinstance(outputdir, str):
                outputdirs[i] = {det: outputdir for det in detectors}
            elif isinstance(outputdir, dict):
                if sorted(outputdir.keys()) != sorted(detectors):
                    raise KeyError(
                        "outputdirs dictionary must have same keys as the given "
                        "detectors"
                    )
                for det in detectors:
                    if not isinstance(outputdir[det], str):
                        raise TypeError("outputdirs must be a string")
            else:
                raise TypeError("outputdirs must be a string or dictionary")

        label = self.eval(config.get("heterodyne", "label", fallback=None))
        if label is not None:
            if isinstance(label, str):
                label = [label]
            elif not isinstance(label, list):
                raise TypeError("label must be a string or a list")

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
                if isinstance(value, (int, float, str)):
                    fullstarttimes[key] = [value]  # convert values to lists
                elif not isinstance(value, list):
                    raise TypeError("Must have a list of start times for a detector")
        elif isinstance(fullstarttimes, (int, float, str)):
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
                if isinstance(value, (int, float, str)):
                    fullendtimes[key] = [value]  # convert values to lists
                elif not isinstance(value, list):
                    raise TypeError("Must have a list of end times for a detector")
        elif isinstance(fullendtimes, (int, float, str)):
            fullendtimes = {det: [fullendtimes] for det in detectors}  # convert to dict
        else:
            raise ValueError("End times must be given")

        # check if any start/end times are strings that can be converted to GPS
        for timedict in [fullstarttimes, fullendtimes]:
            for key in list(timedict.keys()):
                for i in range(len(timedict[key])):
                    if isinstance(timedict[key][i], (int, float)):
                        timedict[key][i] = int(timedict[key][i])
                    else:
                        try:
                            strtime = int(timedict[key][i])
                        except ValueError:
                            try:
                                strtime = int(Time(timedict[key][i], scale="utc").gps)
                            except ValueError:
                                raise ValueError(
                                    f"Time {timedict[key][i]} cannot be converted to GPS time"
                                )

                        # get GPS seconds
                        timedict[key][i] = strtime

        for det in detectors:
            if len(fullendtimes[det]) != len(fullstarttimes[det]):
                raise ValueError("Inconsistent numbers of start and end times")

        stride = config.getint("heterodyne", "stride", fallback=None)
        joblength = config.getint("heterodyne", "joblength", fallback=86400)

        # get frame data information
        frametypes = self.eval(config.get("heterodyne", "frametypes", fallback=None))
        if isinstance(frametypes, str) and len(detectors) == 1:
            frametypes = {det: frametypes for det in detectors}
        framecaches = self.eval(config.get("heterodyne", "framecaches", fallback=None))
        if isinstance(framecaches, str) and len(detectors) == 1:
            framecaches = {det: framecaches for det in detectors}
        channels = self.eval(config.get("heterodyne", "channels", fallback=None))
        if isinstance(channels, str) and len(detectors) == 1:
            channels = {det: channels for det in detectors}
        host = config.get("heterodyne", "host", fallback=None)
        heterodyneddata = self.eval(
            config.get("heterodyne", "heterodyneddata", fallback=None)
        )
        framedata = {det: [] for det in detectors}
        if frametypes is None and framecaches is None and heterodyneddata is None:
            raise ValueError(
                "Frame types, frame cache files, or heterodyned data information must "
                "be supplied"
            )
        require_gwosc = False

        if heterodyneddata is None:
            for fname, finfo in dict(
                frametypes=frametypes, framecaches=framecaches, channels=channels
            ).items():
                if finfo is not None:
                    # set frame types/caches
                    if isinstance(finfo, dict):
                        for key, value in finfo.copy().items():
                            if isinstance(value, str):
                                finfo[key] = [
                                    value for _ in range(len(fullstarttimes[key]))
                                ]
                            elif isinstance(value, list):
                                if len(value) != len(fullstarttimes[key]):
                                    raise ValueError(
                                        "{} lists must be consistent with the number of start and end times".format(
                                            fname
                                        )
                                    )
                            else:
                                raise TypeError("Must have a list of {}".format(fname))
                    else:
                        raise TypeError("{} should be a dictionary".format(fname))

        # set whether to allow unreadable frames to be ignored
        strictdatarequirement = config.get(
            "heterodyne", "strictdatarequirement", fallback=False
        )

        # get segment information
        segmentserver = config.get("heterodyne", "segmentserver", fallback=None)
        segmentlists = self.eval(
            config.get("heterodyne", "segmentlists", fallback=None)
        )
        if isinstance(segmentlists, str) and len(detectors) == 1:
            segmentlists = {det: segmentlists for det in detectors}
        includeflags = self.eval(
            config.get("heterodyne", "includeflags", fallback=None)
        )
        if isinstance(includeflags, str) and len(detectors) == 1:
            includeflags = {det: includeflags for det in detectors}
        excludeflags = self.eval(
            config.get("heterodyne", "excludeflags", fallback=None)
        )
        if isinstance(excludeflags, str) and len(detectors) == 1:
            excludeflags = {det: excludeflags for det in detectors}
        segmentdata = {det: [] for det in detectors}
        if segmentlists is None and includeflags is None and heterodyneddata is None:
            raise ValueError(
                "Segment lists of segment data quality flags must be supplied"
            )

        for sname, sinfo in dict(
            includeflags=includeflags,
            excludeflags=excludeflags,
            segmentlists=segmentlists,
        ).items():
            if sinfo is not None:
                if isinstance(sinfo, dict):
                    for key, value in sinfo.copy().items():
                        if isinstance(value, str):
                            sinfo[key] = [
                                value for _ in range(len(fullstarttimes[key]))
                            ]
                        elif isinstance(value, list):
                            if len(value) != len(fullstarttimes[key]):
                                raise ValueError(
                                    f"{sname} lists must be consistent with the number of start and end times"
                                )
                        else:
                            raise TypeError(f"Must have a list of {sname}")
                else:
                    raise TypeError(f"{sname} should be a dictionary")

        # get ephemeris information
        earthephemeris = self.eval(config.get("ephemerides", "earth", fallback=None))
        sunephemeris = self.eval(config.get("ephemerides", "sun", fallback=None))

        # get all the split segment times and frame caches
        if joblength == 0:
            starttimes = fullstarttimes
            endtimes = fullendtimes

            for det in detectors:
                for i in range(len(fullstarttimes[det])):
                    frinfo = {}
                    if heterodyneddata is None:
                        if frametypes is not None:
                            # generate the frame caches now rather than relying on
                            # each job doing it
                            frcachedir = os.path.join(self.basedir, "cache")
                            if not os.path.exists(frcachedir):
                                os.makedirs(frcachedir)
                            frinfo["framecache"] = os.path.join(
                                frcachedir,
                                "frcache_{0:d}-{1:d}_{2}.txt".format(
                                    starttimes[det][i],
                                    endtimes[det][i],
                                    frametypes[det][i],
                                ),
                            )
                            _ = remote_frame_cache(
                                starttimes[det][i],
                                endtimes[det][i],
                                channels[det][i],
                                frametype=frametypes[det][i],
                                host=config.get("heterodyne", "host", fallback=None),
                                write=frinfo["framecache"],
                            )
                        else:
                            frinfo["framecache"] = framecaches[det][i]
                    frinfo["channel"] = channels[det][i]
                    framedata[det].append(copy.deepcopy(frinfo))

                    seginfo = {}
                    if segmentlists is not None:
                        seginfo["segmentlist"] = segmentlists[det][i]
                    else:
                        # GWOSC segments look like DET_DATA, DET_CW* or DET_*_CAT*
                        usegwosc = False
                        if (
                            "{}_DATA".format(det) == includeflags[det][i]
                            or "{}_CW".format(self.detector) in self.includeflags[0]
                            or "CBC_CAT" in includeflags[det][i]
                            or "BURST_CAT" in includeflags[det][i]
                        ):
                            usegwosc = True
                            require_gwosc = True

                        # if segment list files are not provided create the lists
                        # now rather than relying on each job doing it
                        segdir = os.path.join(self.basedir, "segments")
                        if not os.path.exists(segdir):
                            os.makedirs(segdir)
                        seginfo["segmentlist"] = os.path.join(
                            segdir,
                            "segments_{0:d}-{1:d}_{2}.txt".format(
                                starttimes[det][i],
                                endtimes[det][i],
                                includeflags[det][i].replace(":", "_"),
                            ),
                        )
                        _ = generate_segments(
                            starttime=starttimes[det][i],
                            endtime=endtimes[det][i],
                            includeflags=includeflags[det][i],
                            excludeflags=(
                                None
                                if excludeflags is None
                                else excludeflags[det][i].split(",")
                            ),
                            writesegments=seginfo["segmentlist"],
                            usegwosc=usegwosc,
                            server=segmentserver,
                        )

                    segmentdata[det].append(copy.deepcopy(seginfo))
        elif joblength > 0:
            starttimes = {det: [] for det in detectors}
            endtimes = {det: [] for det in detectors}

            for det in detectors:
                idx = 0
                for starttime, endtime in zip(fullstarttimes[det], fullendtimes[det]):
                    # if segment list files are not provided create the lists
                    # now rather than relying on each job doing it
                    seginfo = {}
                    if segmentlists is not None:
                        seginfo["segmentlist"] = segmentlists[det][idx]

                        segmentlist = generate_segments(
                            starttime=starttime,
                            endtime=endtime,
                            segmentfile=seginfo["segmentlist"],
                        )
                    else:
                        # GWOSC segments look like DET_DATA or DET_*_CAT*
                        usegwosc = False
                        if (
                            "{}_DATA".format(det) == includeflags[det][idx]
                            or "CBC_CAT" in includeflags[det][idx]
                            or "BURST_CAT" in includeflags[det][idx]
                        ):
                            usegwosc = True
                            require_gwosc = True

                        # if segment list files are not provided create the lists
                        # now rather than relying on each job doing it
                        segdir = os.path.join(self.basedir, "segments")
                        if not os.path.exists(segdir):
                            os.makedirs(segdir)
                        seginfo["segmentlist"] = os.path.join(
                            segdir,
                            "segments_{0:d}-{1:d}_{2}.txt".format(
                                starttime,
                                endtime,
                                includeflags[det][idx].replace(":", "_"),
                            ),
                        )
                        segmentlist = generate_segments(
                            starttime=starttime,
                            endtime=endtime,
                            includeflags=includeflags[det][idx],
                            excludeflags=(
                                None
                                if excludeflags is None
                                else excludeflags[det][idx].split(",")
                            ),
                            writesegments=seginfo["segmentlist"],
                            usegwosc=usegwosc,
                            server=segmentserver,
                        )

                        if len(segmentlist) == 0:
                            raise ValueError(
                                f"No science data segments exist for {det}"
                            )

                    # make segment list a list of lists, so values are not immutable
                    segmentlist = [list(seg) for seg in segmentlist]

                    frinfo = {}
                    if heterodyneddata is None:
                        if frametypes is not None:
                            # generate the frame caches now rather than relying on
                            # each job doing it
                            frcachedir = os.path.join(self.basedir, "cache")
                            if not os.path.exists(frcachedir):
                                os.makedirs(frcachedir)
                            frinfo["framecache"] = os.path.join(
                                frcachedir,
                                "frcache_{0:d}-{1:d}_{2}.txt".format(
                                    starttime, endtime, frametypes[det][idx]
                                ),
                            )
                            _ = remote_frame_cache(
                                starttime,
                                endtime,
                                channels[det][idx],
                                frametype=frametypes[det][idx],
                                host=config.get("heterodyne", "host", fallback=None),
                                write=frinfo["framecache"],
                            )
                        else:
                            frinfo["framecache"] = framecaches[det][idx]
                    frinfo["channel"] = channels[det][idx]

                    segidx = 0
                    while segidx < len(segmentlist):
                        curstart = segmentlist[segidx][0]

                        # get segments containing up to joblength of data
                        sumseg = 0
                        while sumseg < joblength:
                            sumseg += segmentlist[segidx][1] - segmentlist[segidx][0]
                            segidx += 1

                            if segidx == len(segmentlist):
                                break

                        if segidx < len(segmentlist):
                            overlapamount = sumseg - joblength
                            segidx -= 1
                            curend = segmentlist[segidx][1] - overlapamount
                            segmentlist[segidx][0] = curend
                        else:
                            # ignore final segment if it's less than 30 mins
                            if sumseg < 30 * 60:
                                break

                            # use end value
                            curend = segmentlist[-1][1]

                        starttimes[det].append(int(curstart))
                        endtimes[det].append(int(curend))

                        # append frame data for jobs
                        framedata[det].append(copy.deepcopy(frinfo))

                        segmentdata[det].append(copy.deepcopy(seginfo))
                    idx += 1
        else:
            raise ValueError("Length of each job must be a positive integer")

        # create Heterodyne objects (one for each detector) to get pulsar parameter
        # file information and heterodyned data information
        hets = {}
        for det in detectors:
            # check if heterodyneddata is a dictionary keyed by detectors
            hetdata = heterodyneddata
            if isinstance(heterodyneddata, dict):
                if det in heterodyneddata:
                    hetdata = heterodyneddata[det]

            hets[det] = Heterodyne(
                pulsarfiles=pulsarfiles,
                pulsars=pulsars,
                heterodyneddata=hetdata,
                detector=det,
                ignore_read_fail=config.getboolean(
                    "heterodyne", "ignore_read_fail", fallback=False
                ),
            )

        het = list(hets.values())[0]

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
        # heterodyne will include no components in the first stage, but all
        # components in the second stage. If supplying different values for a
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

            # filter knee frequency (default to 0.1 Hz for single stage heterodyne)
            filterknee = config.getfloat("heterodyne", "filterknee", fallback=0.1)
            filterorder = config.getint("heterodyne", "filterorder", fallback=9)
        else:
            includessb = self.eval(
                config.get("heterodyne", "includessb", fallback="[False, True]")
            )
            includebsb = self.eval(
                config.get("heterodyne", "includebsb", fallback="[False, True]")
            )
            includeglitch = self.eval(
                config.get("heterodyne", "includeglitch", fallback="[False, True]")
            )
            includefitwaves = self.eval(
                config.get("heterodyne", "includefitwaves", fallback="[False, True]")
            )

            # filter knee frequency (default to 0.5 Hz for two stage heterodyne)
            filterknee = config.getfloat("heterodyne", "filterknee", fallback=0.5)
            filterorder = config.getint("heterodyne", "filterorder", fallback=9)

        # get whether using Tempo2 or not and check it's availability
        usetempo2 = config.getboolean("heterodyne", "usetempo2", fallback=False)
        if usetempo2 and not check_for_tempo2():
            raise ImportError(
                "libstempo is not installed so 'usetempo2' option cannot be used"
            )

        # get the required solar system ephemeris types and binary model for
        # the given pulsars
        etypes = []
        binarymodels = []
        for pf in het.pulsarfiles:
            par = PulsarParameters(het.pulsarfiles[pf])
            etypes.append(par["EPHEM"] if par["EPHEM"] is not None else "DE405")
            if par["BINARY"] is not None:
                binarymodels.append(par["BINARY"])
        self.pulsar_files = copy.deepcopy(het.pulsarfiles)

        # remove duplicates
        etypes = set(etypes)
        binarymodels = set(binarymodels)

        # if ephemeris files are given, set the files explicitly
        available_ephem_types = list(JPLDE)
        if isinstance(earthephemeris, dict) and isinstance(sunephemeris, dict):
            if sorted(list(earthephemeris.keys())) != sorted(list(sunephemeris.keys())):
                raise ValueError("Supplied earth and sun ephemerides are not the same")

            for jplde in list(earthephemeris.keys()):
                if jplde in available_ephem_types:
                    # remove files if ephemeris is already available
                    earthephemeris.pop(jplde)
                    sunephemeris.pop(jplde)
                else:
                    available_ephem_types.append(jplde)

        # create copy of each ephemeris file to a unique name in case of identical
        # filenames, which causes problems if requiring files be transferred
        transfer_files = config.get(dagsection, "transfer_files", fallback="YES")
        osg = config.getboolean(dagsection, "osg", fallback=False)
        if (transfer_files == "YES" or osg) and earthephemeris:
            for edat, ename in zip([earthephemeris, sunephemeris], ["earth", "sun"]):
                if (
                    len(set([os.path.basename(edat[etype]) for etype in edat]))
                    != len(edat)
                    and len(edat) > 1
                ):
                    for etype in edat:
                        tmpephem = os.path.join(
                            tempfile.gettempdir(), f"{ename}_{etype}"
                        )
                        shutil.copy(edat[etype], tmpephem)
                        edat[etype] = tmpephem

        # check that ephemeris files exist for all required types
        if not usetempo2:
            for etype in etypes:
                if etype not in available_ephem_types:
                    raise ValueError(
                        f"Pulsar(s) require ephemeris '{etype}' which has not been supplied"
                    )

        # check that binary models exist for all required types
        if not usetempo2:
            for bmodel in binarymodels:
                if bmodel not in LAL_BINARY_MODELS:
                    raise ValueError(
                        f"Pulsar(s) require binary model type '{bmodel}' "
                        "which is not available in LALSuite. Try the "
                        "usetempo2 option."
                    )

        # check output directories and labels lists are correct length
        if stages == 1:
            if label is not None:
                if len(label) == 0:
                    raise ValueError("A label must be supplied")
            if len(outputdirs) == 0:
                raise ValueError("An output directory must be supplied")
        else:
            if label is not None:
                if len(label) != 2:
                    raise ValueError(
                        "Two labels must be supplied, one for each heterodyne stage"
                    )
            if len(outputdirs) != 2:
                raise ValueError(
                    "Two output directories must be supplied, one for each heterodyne stage"
                )

        interpolationstep = config.get("heterodyne", "interpolationstep", fallback=60)
        crop = config.getint("heterodyne", "crop", fallback=60)
        overwrite = config.getboolean("heterodyne", "overwrite", fallback=False)

        merge = config.getboolean("merge", "merge", fallback=True) and joblength > 0

        # dictionary containing the output files for the merge results
        self.mergeoutputs = {
            det: {ff: {psr: None for psr in het.pulsars} for ff in freqfactors}
            for det in detectors
        }

        # dictionary to contain all the heterodyned data files for each pulsar
        self.heterodyned_files = {
            det: {ff: {psr: [] for psr in het.pulsars} for ff in freqfactors}
            for det in detectors
        }

        # loop over sets of pulsars to create heterodyne jobs
        for pidx, pgroup in enumerate(pulsargroups):
            # loop over frequency factors
            for ff in freqfactors:
                # loop over each detector
                for det in detectors:
                    configurations = []  # config files for each time

                    # loop over times
                    idx = 0
                    for starttime, endtime in zip(starttimes[det], endtimes[det]):
                        configdict = {}

                        configdict["starttime"] = starttime
                        configdict["endtime"] = endtime
                        configdict["detector"] = det
                        configdict["freqfactor"] = ff
                        configdict["resamplerate"] = resamplerate[0]
                        configdict["filterknee"] = filterknee
                        configdict["filterorder"] = filterorder
                        configdict["crop"] = crop
                        configdict["overwrite"] = overwrite

                        # set frame data/heterodyned data info
                        configdict.update(framedata[det][idx])
                        configdict["host"] = host
                        configdict["stride"] = stride
                        configdict["heterodyneddata"] = (
                            heterodyneddata
                            if heterodyneddata is None
                            else {psr: hets[det].heterodyneddata[psr] for psr in pgroup}
                        )
                        configdict["strictdatarequirement"] = strictdatarequirement

                        # set segment data info
                        configdict.update(segmentdata[det][idx])

                        configdict["pulsarfiles"] = {
                            psr: het.pulsarfiles[psr] for psr in pgroup
                        }
                        configdict["pulsars"] = copy.deepcopy(pgroup)

                        # set whether to include modulations
                        configdict["includessb"] = includessb[0]
                        configdict["includebsb"] = includebsb[0]
                        configdict["includeglitch"] = includeglitch[0]
                        configdict["includefitwaves"] = includefitwaves[0]
                        configdict["interpolationstep"] = interpolationstep
                        configdict["usetempo2"] = usetempo2

                        # include ephemeris files
                        configdict["earthephemeris"] = earthephemeris
                        configdict["sunephemeris"] = sunephemeris

                        # temporary Heterodyne object to get the output file names
                        tmphet = Heterodyne(
                            starttime=starttime,
                            endtime=endtime,
                            detector=det,
                            freqfactor=ff,
                            output=outputdirs[0][det],
                            label=label[0] if label is not None else None,
                            pulsars=copy.deepcopy(pgroup),
                            pulsarfiles=pulsarfiles,
                        )

                        # get lists of set of output heterodyned files for each pulsar/detector
                        for psr in pgroup:
                            self.heterodyned_files[det][ff][psr].append(
                                copy.deepcopy(tmphet.outputfiles[psr])
                            )

                        # set the final merged output files
                        for psr in pgroup:
                            if merge and self.mergeoutputs[det][ff][psr] is None:
                                # use full start and end times
                                tmphet.starttime = starttimes[det][0]
                                tmphet.endtime = endtimes[det][-1]
                                self.mergeoutputs[det][ff][psr] = os.path.join(
                                    outputdirs[0][det], tmphet.outputfiles[psr]
                                )

                        configdict["output"] = outputdirs[0][det]
                        configdict["label"] = label[0] if label is not None else None

                        # store configurations
                        configurations.append(
                            {
                                key: copy.deepcopy(value)
                                for key, value in configdict.items()
                                if value is not None
                            }
                        )

                        idx += 1

                    _ = HeterodyneLayer(
                        self.dag,
                        config,
                        configurations,
                        layer_name=f"cwinpy_heterodyne_{pidx}_{ff}_{det}",
                        require_gwosc=require_gwosc,
                    )

        # if performing sky-shifting, generate the new shifted pulsar parameter files
        skyshift = config.has_section("skyshift")
        if skyshift:
            pulsardir = os.path.join(config["run"]["basedir"], "pulsars")
            if not os.path.exists(pulsardir):
                os.makedirs(pulsardir)

            # copy original file par file into pulsar directory
            shutil.copy2(pulsarfiles, pulsardir)

            origpsrname = pulsargroups[0][0]  # single pulsar name
            pulsargroups = []  # overwrite pulsargroups with sky-shifted pulsar names

            parfiles = [pulsarfiles]  # add original (non-sky-shifted) parameters

            nshifts = config.getint("skyshift", "nshifts", fallback=0)

            if nshifts == 0:
                raise ValueError("Number of sky shifts must be specified")

            hemisphere = config.get("skyshift", "hemisphere", fallback=None)

            if hemisphere not in ["north", "south"]:
                raise ValueError("Hemisphere must be set to 'north' or 'south'")

            exclusion = config.getfloat("skyshift", "exclusion", fallback=0.01)
            overlapfrac = config.getfloat("skyshift", "overlap", fallback=0.0)
            tobs = config.getfloat("skyshift", "tobs", fallback=0.0)

            if overlapfrac > 0.0 and tobs == 0.0:
                raise ValueError(
                    "Total observation time must be given to calculate allowed overlap when sky-shifting"
                )

            # get pulsar position
            psr = PulsarParameters(pulsarfiles)

            # generate new positions
            ra = psr["RAJ"] if psr["RAJ"] is not None else psr["RA"]
            dec = psr["DECJ"] if psr["DECJ"] is not None else psr["DEC"]
            pulsargroups.append([get_psr_name(psr)])
            pos = SkyCoord(ra, dec, unit="rad")  # current position

            for i in range(nshifts):
                # generate points while checking angular distance from true position
                while True:
                    # draw points from same ecliptic hemisphere as source
                    ranew, decnew = draw_ra_dec(eclhemi=hemisphere)
                    newpos = SkyCoord(ranew, decnew, unit="rad")

                    # check new position is outside exclusion region
                    if np.abs(pos.separation(newpos).rad) > exclusion:
                        if overlapfrac == 0.0:
                            break

                        # check new position has small fractional overlap
                        if (
                            overlap(pos, newpos, psr["F0"], tobs, starttime, dt=600)
                            < overlapfrac
                        ):
                            break

                # output par files for all new positions
                newpsr = copy.deepcopy(psr)
                if newpsr["RAJ"] is not None:
                    newpsr["RAJ"] = newpos.ra.rad
                else:
                    newpsr["RA"] = newpos.ra.rad
                if newpsr["DECJ"] is not None:
                    newpsr["DECJ"] = newpos.dec.rad
                else:
                    newpsr["DEC"] = newpos.dec.rad

                # make name unique with additional alphabetical values
                anum = int_to_alpha(i + 1)
                newpsr["PSRJ"] = get_psr_name(newpsr) + anum
                pulsargroups.append([newpsr["PSRJ"]])

                parfiles.append(
                    os.path.join(pulsardir, "{}.par".format(newpsr["PSRJ"]))
                )

                # output parameter file
                newpsr.pp_to_par(parfiles[-1])

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
                            configdict["pulsarfiles"] = (
                                pulsarfiles if not skyshift else parfiles[i]
                            )
                            configdict["resamplerate"] = resamplerate[-1]

                            # include all modulations
                            configdict["includessb"] = includessb[-1]
                            configdict["includebsb"] = includebsb[-1]
                            configdict["includeglitch"] = includeglitch[-1]
                            configdict["includefitwaves"] = includefitwaves[-1]

                            # include ephemeris files
                            configdict["earthephemeris"] = earthephemeris
                            configdict["sunephemeris"] = sunephemeris

                            # input the data
                            configdict["heterodyneddata"] = {
                                psr: self.heterodyned_files[det][ff][
                                    psr if not skyshift else origpsrname
                                ]
                            }

                            # output structure (always overwrite for 2nd stage)
                            configdict["overwrite"] = True
                            configdict["output"] = outputdirs[1][det]
                            configdict["label"] = (
                                label[1] if label is not None else None
                            )

                            # temporary Heterodyne object to get the output file names
                            tmphet = Heterodyne(
                                starttime=configdict["starttime"],
                                endtime=configdict["endtime"],
                                detector=det,
                                freqfactor=ff,
                                output=outputdirs[1][det],
                                label=configdict["label"],
                                pulsars=psr,
                                pulsarfiles=copy.deepcopy(configdict["pulsarfiles"]),
                            )

                            self.mergeoutputs[det][ff][psr] = os.path.join(
                                outputdirs[1][det], tmphet.outputfiles[psr]
                            )

                            HeterodyneLayer(
                                self.dag,
                                config,
                                [
                                    {
                                        key: copy.deepcopy(value)
                                        for key, value in configdict.items()
                                        if value is not None
                                    }
                                ],
                                layer_name=f"cwinpy_heterodyne_{ff}_{det}_{psr.replace('+', 'plus')}",
                                parentname=f"cwinpy_heterodyne_*_{ff}_{det}"
                                if not skyshift
                                else f"cwinpy_heterodyne_0_{ff}_{det}",
                            )
        elif merge:
            # set output merge jobs
            for i, pgroup in enumerate(pulsargroups):
                for psr in pgroup:
                    for ff in freqfactors:
                        for det in detectors:
                            MergeLayer(
                                self.dag,
                                config,
                                {
                                    "heterodynedfiles": copy.deepcopy(
                                        self.heterodyned_files[det][ff][psr]
                                    ),
                                    "freqfactor": ff,
                                    "detector": det,
                                    "pulsar": psr,
                                    "output": copy.deepcopy(
                                        self.mergeoutputs[det][ff][psr]
                                    ),
                                },
                                layer_name=f"cwinpy_heterodyne_{ff}_{det}_{psr.replace('+', 'plus')}",
                                parentname=f"cwinpy_heterodyne_{i}_{ff}_{det}"
                                if not skyshift
                                else f"cwinpy_heterodyne_0_{ff}_{det}",
                            )

        if self.build:
            # write out the DAG and submit files
            submitdir = config.get(
                dagsection, "submit", fallback=os.path.join(self.basedir, "submit")
            )
            if not os.path.exists(submitdir):
                os.makedirs(submitdir)
            self.dag_file = write_dag(
                self.dag, submitdir, dag_file_name="cwinpy_heterodyne.dag"
            )

            # submit the DAG if requested
            if config.getboolean(dagsection, "submitdag", fallback=False):
                submit_dag(self.dag_file)

    def eval(self, arg):
        """
        Evaluate a string using simpleeval.

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
            seval = EvalWithCompoundTypes()
            return seval.eval(newobj)
        except (ValueError, SyntaxError, NameNotDefined):
            return newobj


def heterodyne_pipeline(**kwargs):
    """
    Run heterodyne_pipeline within Python. This will create a
    `HTCondor <https://htcondor.readthedocs.io/>`__ DAG for running multiple
    ``cwinpy_heterodyne`` instances on a computer cluster. Optional parameters
    that can be used instead of a configuration file (for "quick setup") are
    given in the "Other parameters" section.

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
        given run. If no ``pulsar`` argument is given then all hardware
        injections will be analysed. To specify particular hardware injections
        the names can be given using the ``pulsar`` argument.
    samplerate: str:
        Select the sample rate of the data to use. This can either be 4k or
        16k for data sampled at 4096 or 16384 Hz, respectively. The default
        is 4k, except if running on hardware injections for O1 or later, for
        which 16k will be used due to being requred for the highest frequency
        source. For the S5 and S6 runs only 4k data is avaialble from GWOSC,
        so if 16k is chosen it will be ignored.
    pulsar: str, list
        The path to, or list of paths to, a Tempo(2)-style pulsar parameter
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
    accounting_group: str
        For LVK users this sets the computing accounting group tag.
    usetempo2: bool
        Set this flag to use Tempo2 (if installed) for calculating the signal
        phase evolution rather than the default LALSuite functions.

    Returns
    -------
    dag:
        An object containing a pycondor :class:`pycondor.Dagman` object.
    """

    if "config" in kwargs:
        configfile = kwargs.pop("config")
    else:  # pragma: no cover
        parser = ArgumentParser(
            description=(
                "A script to create a HTCondor DAG to process GW strain data "
                "by heterodyning it based on the expected phase evolution for "
                "a selection of pulsars."
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
                "this case, in which case all hardware injections will be "
                "used. To specific particular hardware injections, the "
                "required names can be set with the '--pulsar' flag."
            ),
        )
        optional.add_argument(
            "--samplerate",
            help=(
                "Select the sample rate of the data to use. This can either "
                "be 4k or 16k for data sampled at 4096 or 16384 Hz, "
                "respectively. The default is 4k, except if running on "
                "hardware injections for O1 or later, for which 16k will be "
                "used due to being requred for the highest frequency source. "
                "For the S5 and S6 runs only 4k data is avaialble from GWOSC, "
                "so if 16k is chosen it will be ignored."
            ),
            default="4k",
        )
        optional.add_argument(
            "--pulsar",
            action="append",
            help=(
                "The path to a Tempo(2)-style pulsar parameter file, or "
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
            "--accounting-group",
            dest="accgroup",
            help=("For LVK users this sets the computing accounting group tag"),
        )
        optional.add_argument(
            "--usetempo2",
            action="store_true",
            help=(
                "Set this flag to use Tempo2 (if installed) for calculating "
                "the signal phase evolution rather than the default LALSuite "
                "functions."
            ),
        )

        args = parser.parse_args()
        if args.config is not None:
            configfile = args.config
        else:
            configfile = heterodyne_quick_setup(args, **kwargs)

    if isinstance(configfile, configparser.ConfigParser):
        config = configfile
    else:
        config = configparser.ConfigParser()

        try:
            config.read_file(open(configfile, "r"))
        except Exception as e:
            raise IOError(f"Problem reading configuration file '{configfile}'\n: {e}")

    return HeterodyneDAGRunner(config, **kwargs)


def heterodyne_pipeline_cli(**kwargs):  # pragma: no cover
    """
    Entry point to the cwinpy_heterodyne_pipeline script. This just calls
    :func:`cwinpy.heterodyne.heterodyne_pipeline`, but does not return any objects.
    """

    _ = heterodyne_pipeline(**kwargs)


def heterodyne_quick_setup(args, **kwargs):
    """
    Setup the configuration based on the "quick setup" command line arguments.

    Parameters
    ----------
    args:
        The output of :meth:`argparse.ArgumentParser.parse_args`.

    Returns
    -------
    configfile:
        A :class:`configparser:ConfigParser` object containing the required
        settings.
    """

    # create the ConfigParser object
    configfile = configparser.ConfigParser()

    run = kwargs.get("run", args.run)
    if run not in RUNTIMES:
        raise ValueError(f"Requested run '{run}' is not available")

    pulsars = []
    hwinj = kwargs.get("hwinj", args.hwinj)
    if hwinj:
        # use hardware injections for the run
        runtimes = HW_INJ_RUNTIMES
        segments = HW_INJ_SEGMENTS

        # check if requesting any particular hardware injections
        argpulsars = kwargs.get("pulsar", args.pulsar)
        if argpulsars is not None:
            argpulsars = argpulsars if isinstance(argpulsars, list) else [argpulsars]

            for pf in argpulsars:
                hwinjf = is_hwinj(pf, return_file=True)
                if hwinjf:
                    pulsars.append(hwinjf)
        else:
            # use all HW injections
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

        pulsars.extend(pulsar if isinstance(pulsar, list) else [pulsar])

        # get sample rate
        srate = "16k" if (args.samplerate[0:2] == "16" and run[0] == "O") else "4k"

        # check if any pulsar par files are hardware injection and if so use
        # appropriate segments
        for pf in pulsars:
            if is_hwinj(pf):
                runtimes = HW_INJ_RUNTIMES
                segments = HW_INJ_SEGMENTS
                srate = "16k" if run[0] == "O" else "4k"
                hwinj = True
                break

    detector = kwargs.get("detector", args.detector)
    if args.detector is None:
        detectors = list(runtimes[run].keys())
    else:
        detector = detector if isinstance(detector, list) else [detector]
        detectors = [det for det in detector if det in runtimes[run]]
        if len(detectors) == 0:
            raise ValueError(
                f"Provided detectors '{detector}' are not valid for the given run"
            )

    # create required settings
    configfile["run"] = {}
    configfile["run"]["basedir"] = kwargs.get("output", args.output)

    configfile["heterodyne_dag"] = {}
    configfile["heterodyne_dag"]["submitdag"] = "True"
    if kwargs.get("osg", args.osg):
        configfile["heterodyne_dag"]["osg"] = "True"

    configfile["heterodyne_job"] = {}
    configfile["heterodyne_job"]["getenv"] = "True"

    if args.accgroup is not None:
        configfile["heterodyne_job"]["accounting_group"] = kwargs.get(
            "accounting_group", args.accgroup
        )

    # add pulsars/pulsar ephemerides
    configfile["ephemerides"] = {}
    configfile["ephemerides"]["pulsarfiles"] = str(pulsars)

    # add heterodyne settings
    configfile["heterodyne"] = {}
    configfile["heterodyne"]["detectors"] = str(detectors)

    if run == "O3":
        # for full O3 we need to set times for O3a and O3b separately
        configfile["heterodyne"]["starttimes"] = str(
            {
                det: [runtimes[o3run][det][0] for o3run in ["O3a", "O3b"]]
                for det in detectors
            }
        )
        configfile["heterodyne"]["endtimes"] = str(
            {
                det: [runtimes[o3run][det][1] for o3run in ["O3a", "O3b"]]
                for det in detectors
            }
        )

        configfile["heterodyne"]["frametypes"] = str(
            {
                det: [
                    CVMFS_GWOSC_DATA_TYPES[o3run][srate][det]
                    for o3run in ["O3a", "O3b"]
                ]
                for det in detectors
            }
        )
    else:
        configfile["heterodyne"]["starttimes"] = str(
            {det: runtimes[run][det][0] for det in detectors}
        )
        configfile["heterodyne"]["endtimes"] = str(
            {det: runtimes[run][det][1] for det in detectors}
        )

        configfile["heterodyne"]["frametypes"] = str(
            {det: CVMFS_GWOSC_DATA_TYPES[run][srate][det] for det in detectors}
        )

    configfile["heterodyne"]["channels"] = str(
        {det: CVMFS_GWOSC_FRAME_CHANNELS[run][srate][det] for det in detectors}
    )
    configfile["heterodyne"]["host"] = CVMFS_GWOSC_DATA_SERVER
    if hwinj:
        configfile["heterodyne"]["includeflags"] = str(
            {det: segments[run][det]["includesegments"] for det in detectors}
        )
        configfile["heterodyne"]["excludeflags"] = str(
            {det: segments[run][det]["excludesegments"] for det in detectors}
        )
    else:
        configfile["heterodyne"]["includeflags"] = str(
            {det: segments[run][det] for det in detectors}
        )
    configfile["heterodyne"]["outputdir"] = str(
        {
            det: os.path.join(kwargs.get("output", os.path.abspath(args.output)), det)
            for det in detectors
        }
    )
    configfile["heterodyne"]["overwrite"] = "False"

    # set whether to use Tempo2 for phase evolution
    if kwargs.get("usetempo2", args.usetempo2):
        configfile["heterodyne"]["usetempo2"] = "True"

    # split the analysis into on average day long chunks
    if kwargs.get("joblength", args.joblength) is None:
        configfile["heterodyne"]["joblength"] = "86400"
    else:
        configfile["heterodyne"]["joblength"] = str(
            kwargs.get("joblength", args.joblength)
        )

    # merge the resulting files and remove individual files
    configfile["merge"] = {}
    configfile["merge"]["merge"] = "True"
    configfile["merge"]["remove"] = "True"
    configfile["merge"]["overwrite"] = "True"

    return configfile
