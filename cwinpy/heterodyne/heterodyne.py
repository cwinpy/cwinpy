"""
Run heterodyne pre-processing of gravitational-wave data.
"""

import ast
import sys

import cwinpy
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import BilbyPipeError, parse_args

# from ..utils import sighandler
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
            'parameter files found through the "--pulsars" argument by '
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
    het.heterodyne()

    return het


def heterodyne_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_heterodyne`` script. This just calls
    :func:`cwinpy.heterodyne.heterodyne`, but does not return any objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = heterodyne(**kwargs)
