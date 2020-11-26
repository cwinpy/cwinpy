"""
Run heterodyne pre-processing of gravitational-wave data.
"""

import ast
import sys

import cwinpy
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import BilbyPipeError, parse_args

# from ..utils import sighandler
from .heterodyne import Heterodyne


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
    nsattrs = ["framecache", "segmentlist", "includeflags"]
    for attr in nsattrs:
        value = hetkwargs.pop(attr, None)

        if value is not None:
            # check whether the value can be evaluated as a Python object
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass

            hetkwargs[attr] = value

    # set up the run
    het = Heterodyne(**hetkwargs)

    # heterodyne the data
    het.heterodyne()

    return het


def heterodyne_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_heterodyne script``. This just calls
    :func:`cwinpy.heterodyne.heterodyne`, but does not return any objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = heterodyne(**kwargs)
