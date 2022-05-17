"""
A selection of general utility functions.
"""

import ast
import ctypes
import os
import pathlib
import re
import string
import sys
from copy import deepcopy
from functools import reduce
from math import gcd

import appdirs
import lalpulsar
import numpy as np
import requests
from astropy import units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from numba import jit, njit
from numba.extending import get_cython_function_address

from .parfile import PulsarParameters

#: exit code to return when checkpointing
CHECKPOINT_EXIT_CODE = 77

#: URL for LALSuite solar system ephemeris files
LAL_EPHEMERIS_URL = "https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/{}"

#: the current solar system ephemeris types in LALSuite
LAL_EPHEMERIS_TYPES = ["DE200", "DE405", "DE421", "DE430"]

#: the location for caching ephemeris files
EPHEMERIS_CACHE_DIR = appdirs.user_cache_dir(appname="cwinpy", appauthor=False)

#: the current TEMPO-compatible binary system model types provided in LALSuite
LAL_BINARY_MODELS = [
    "BT",
    "BT1P",
    "BT2P",
    "BTX",
    "ELL1",
    "DD",
    "DDS",
    "MSS",
    "T2",
]

#: aliases between GW detector prefixes and TEMPO2 observatory names
TEMPO2_GW_ALIASES = {
    "G1": "GEO600",
    "GEO_600": "GEO600",
    "H1": "HANFORD",
    "H2": "HANFORD",
    "LHO_4k": "HANFORD",
    "LHO_2k": "HANFORD",
    "K1": "KAGRA",
    "KAGRA": "KAGRA",
    "L1": "LIVINGSTON",
    "LLO_4k": "LIVINGSTON",
    "V1": "VIRGO",
    "VIRGO": "VIRGO",
}

# create a numba-ified version of scipy's gammaln function (see, e.g.
# https://github.com/numba/numba/issues/3086#issuecomment-403469308)
addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_int)
gammaln_fn = functype(addr)


@njit
def gammaln(x):
    return gammaln_fn(x, 0)  # 0 is for the required int!


@jit(nopython=True)
def logfactorial(n):
    """
    The natural logarithm of the factorial of an integer using the fact that

    .. math::

        \\ln{(n!)} = \\ln{\\left(\\Gamma (n+1)\\right)}

    Parameters
    ----------
    n: int
        An integer for which the natural logarithm of its factorial is
        required.

    Returns
    -------
    float
    """

    return gammaln(n + 1)


@jit(nopython=True)
def allzero(array):
    """
    Check if an array is all zeros. See https://stackoverflow.com/a/53474543/1862861.

    Parameters
    ----------
    array: array_like
        A :class:`numpy.ndarray` to check.

    Returns
    -------
    bool:
        True if all zero, False otherwise.
    """

    for x in array.flat:
        if x:
            return False
    return True


def gcd_array(denominators):
    """
    Function to calculate the greatest common divisor of a list of values.
    """

    if not isinstance(denominators, (tuple, list, np.ndarray)):
        raise TypeError("Must have a list or array")

    denoms = np.asarray(denominators).flatten()  # 1d array
    denoms = denoms.astype(int)

    if len(denoms) < 2:
        raise ValueError("Must have more than two values")

    return reduce(lambda a, b: gcd(a, b), denoms)


def is_par_file(parfile):
    """
    Check if a TEMPO-style pulsar parameter file is valid.

    Parameters
    ----------
    parfile: str
        The path to a file.

    Returns
    -------
    ispar: bool
        Returns True is if it is a valid parameter file.
    """

    try:
        psr = PulsarParameters(parfile)
    except (ValueError, IOError):
        return False

    # file must contain a frequency, right ascension, declination and
    # pulsar name
    if (
        psr["F"] is None
        or (psr["RAJ"] is None and psr["RA"] is None)
        or (psr["DECJ"] is None and psr["DEC"] is None)
        or get_psr_name(psr) is None
    ):
        return False
    else:
        return True


def get_psr_name(psr):
    """
    Get the pulsar name from the Tempo(2)-style parameter files by trying the
    keys "PSRJ", "PSRB", "PSR", and "NAME" in that order of precedence.

    Parameters
    ----------
    psr: PulsarParameter
        A :class:`~cwinpy.parfile.PulsarParameters` object

    Returns
    -------
    name: str
        The string containing the name or None if not found.
    """

    for name in ["PSRJ", "PSRB", "PSR", "NAME"]:
        if psr[name] is not None:
            return psr[name]

    return None


def int_to_alpha(pos, case="upper"):
    """
    For an integer number generate a corresponding alphabetical string
    following the mapping: 1 -> "A", 2 -> "B", ..., 27 -> "AA", 28 -> "AB", ...

    Parameters
    ----------
    pos: int
        A positive integer greater than 0.
    case: str
        Set whether to use upper or lower case. Default is "upper".

    Returns
    -------
    alpha: str
        The alphabetical equivalent
    """

    if case == "lower":
        alphas = list(string.ascii_lowercase)
    else:
        alphas = list(string.ascii_uppercase)

    if not isinstance(pos, int):
        raise TypeError("Value must be an integer")

    if pos < 1:
        raise ValueError("Value must be greater than zero")

    count = pos - 1
    result = ""
    while count >= 0:
        result = alphas[int(count) % len(alphas)] + result
        count /= len(alphas)
        count -= 1

    return result


def ellipticity_to_q22(epsilon, units=False):
    """
    Convert the fiducial ellipticity :math:`\\varepsilon` to the mass
    quadrupole :math:`Q_{22}` (in units of kg m\ :sup:`2`) via (see, e.g.,
    Equation A3 of [1]_):

    .. math::

        Q_{22} = \\varepsilon I_{38} \\sqrt{\\frac{15}{8\\pi}},

    where :math:`I_{38} = 10^{38}` kg m\ :sup:`2` is the canonical moment of
    inertia of the star.


    Parameters
    ----------
    epsilon: float, array_like
        A value, or array of values, of the mass quadrupole.
    units: bool
        If True add units to the output.

    Returns
    -------
    q22: float, array_like
        A value, or array of values, of the mass quadrupole.
    """

    if isinstance(epsilon, float):
        q22 = epsilon * 1e38 * np.sqrt(15.0 / (8.0 * np.pi))
    else:
        q22 = np.array(epsilon) * 1e38 * np.sqrt(15.0 / (8.0 * np.pi))

    if units:
        # add units
        return q22 * u.kg * u.m**2
    else:
        return q22


def q22_to_ellipticity(q22):
    """
    Convert mass quadrupole :math:`Q_{22}` (in units of kg m\ :sup:`2`) to
    fidicual ellipticity via (see, e.g., Equation A3 of [1]_):

    .. math::

        \\varepsilon = \\frac{Q_{22}}{I_{38}}\\sqrt{\\frac{8\\pi}{15}},

    where :math:`I_{38} = 10^{38}` kg m\ :sup:`2` is the canonical moment of
    inertia of the star.

    Parameters
    ----------
    q22: float, array_like
        A value, or array of values, of the mass quadrupole.

    Returns
    -------
    ellipticity: float, array_like
        A value, or array of values, of the fiducial ellipticity.
    """

    if isinstance(q22, list):
        ellipticity = (np.array(q22) / 1e38) * np.sqrt(8.0 * np.pi / 15.0)
    else:
        ellipticity = (q22 / 1e38) * np.sqrt(8.0 * np.pi / 15.0)

    if hasattr(q22, "value"):
        # remove units if value/array has astropy units
        return ellipticity.value
    else:
        return ellipticity


def lalinference_to_bilby_result(postfile):
    """
    Convert LALInference-derived pulsar posterior samples file, as created by
    ``lalapps_pulsar_parameter_estimation_nested``, into a
    :class:`bilby.core.result.Result` object.

    Parameters
    ----------
    postfile: str
        The path to a posterior samples file.

    Returns
    -------
    result:
        The results as a :class:`bilby.core.result.Result` object
    """

    import h5py

    from bilby.core.result import Result
    from lalinference import bayespputils as bppu
    from pandas import DataFrame

    try:
        peparser = bppu.PEOutputParser("hdf5")
        nsResultsObject = peparser.parse(postfile)
        pos = bppu.Posterior(nsResultsObject, SimInspiralTableEntry=None)
    except Exception as e:
        raise IOError(f"Could not import posterior samples from {postfile}: {e}")

    # remove any unchanging variables and randomly shuffle the rest
    pnames = pos.names
    nsamps = len(pos[pnames[0]].samples)
    permarr = np.arange(nsamps)
    np.random.shuffle(permarr)

    posdict = {}
    for pname in pnames:
        # ignore if all samples are the same
        if not pos[pname].samples.tolist().count(pos[pname].samples[0]) == len(
            pos[pname].samples
        ):
            # shuffle and store
            posdict[pname] = pos[pname].samples[permarr, 0]

    # get evidence values from HDF5 file
    logZ = None
    logZn = None
    logbayes = None
    try:
        hdf = h5py.File(postfile, "r")
        a = hdf["lalinference"]["lalinference_nest"]
        logZ = a.attrs["log_evidence"]
        logZn = a.attrs["log_noise_evidence"]
        logbayes = logZ - logZn
        hdf.close()
    except KeyError:
        pass

    return Result(
        posterior=DataFrame(posdict),
        log_evidence=logZ,
        log_noise_evidence=logZn,
        log_bayes_factor=logbayes,
    )


def draw_ra_dec(n=1, eqhemi=None, eclhemi=None):
    """
    Draw right ascension and declination values uniformly on the sky or
    uniformly from a single equatorial or ecliptic hemisphere.

    Parameters
    ----------
    n: int
        The number of points to draw on the sky.
    eqhemi: str
        A string that this either "north" or "south" to restrict drawn points
        to a single equatorial hemisphere.
    eclhemi: str
        A string that this either "north" or "south" to restrict drawn points
        to a single ecliptic hemisphere.

    Returns
    -------
    radec: tuple
        A tuple containing the pair of values (ra then dec) or a pair of NumPy
        arrays containing the values.
    """

    # check hemisphere arguments are valid
    for hemi in [eqhemi, eclhemi]:
        if hemi is not None:
            if not isinstance(hemi, str):
                raise TypeError("hemisphere arguments must be a string")
            elif hemi.lower() != "north" and hemi.lower() != "south":
                raise ValueError("hemisphere argument must be 'north' or 'south'")

    # draw points
    rng = np.random.default_rng()
    lon = rng.uniform(0.0, 2.0 * np.pi, n)
    lat = np.arcsin(2.0 * rng.uniform(0, 1, n) - 1.0)

    if eclhemi is None and eqhemi is not None:
        # get points from one hemisphere if required
        lat = np.abs(lat) if eqhemi.lower() == "north" else -np.abs(lat)
    elif eclhemi is not None:
        # get points from one hemisphere
        lat = np.abs(lat) if eclhemi.lower() == "north" else -np.abs(lat)

        # convert to right ascension and declination
        pos = SkyCoord(lon, lat, unit="rad", frame="barycentrictrueecliptic")
        lon = pos.icrs.ra.rad
        lat = pos.icrs.dec.rad

    return (lon[0], lat[0]) if n == 1 else (lon, lat)


def overlap(pos1, pos2, f0, T, t0=1000000000, dt=60, det="H1"):
    """
    Calculate the overlap (normalised cross-correlation) between a heterodyned
    signal at one position and a signal re-heterodyned assuming a different
    position. This will assume a circularly polarised signal

    Parameters
    ----------
    pos1: SkyCoord
        The sky position, as an :class:`astropy.coordinates.SkyCoord` object,
        of the heterodyned signal.
    pos2: SkyCoord
        Another position at which the signal has been re-heterodyned.
    f0: int, float
        The signal frequency
    T: int, float
        The signal duration (seconds)
    t0: int, float
        The GPS start time of the signal (default is 1000000000)
    dt: int, float
        The time steps over which to calculate the signal.
    det: str
        A detector name, e.g., "H1" (the default).

    Returns
    -------
    overlap: float
        The fractional overlap between the two models.
    """

    from .signal import HeterodynedCWSimulator

    # set up a pulsar
    p1 = PulsarParameters()
    p1["H0"] = 1.0
    p1["IOTA"] = 0.0
    p1["PSI"] = 0.0
    p1["PHI0"] = 0.0
    p1["F"] = [float(f0)]
    p1["RAJ"] = pos1.ra.rad
    p1["DECJ"] = pos1.dec.rad

    # set the times at which to calculate the model
    times = np.arange(t0, t0 + T, dt)

    het = HeterodynedCWSimulator(p1, det, times=times)

    # calculate the model at pos1
    hetmodel = het.model()

    denom = abs(np.vdot(hetmodel, hetmodel).real)

    p2 = deepcopy(p1)
    p2["RAJ"] = pos2.ra.rad
    p2["DECJ"] = pos2.dec.rad

    # calculate the model re-heterodyned at pos2
    hetmodel2 = het.model(newpar=p2, updateSSB=True)

    # get the overlap
    numer = abs(np.vdot(hetmodel, hetmodel2).real)

    # the fractional overlap
    overlap = numer / denom

    return overlap


def initialise_ephemeris(
    ephem="DE405",
    units="TCB",
    earthfile=None,
    sunfile=None,
    timefile=None,
    ssonly=False,
    timeonly=False,
    filenames=False,
):
    """
    Download/read and return solar system ephemeris and time coordinate data.
    If files are provided these will be used and read. If not provided then,
    using supplied ``ephem`` and ``units`` values, it will first attempt to
    find files locally (either in your current path or in a path supplied by
    a ``LAL_DATA_PATH`` environment variable), and if not present will then
    attempt to download the files from a repository.

    To do
    -----

    Add the ability to create ephemeris files using astropy.

    Parameters
    ----------
    earthfile: str
        A file containing the Earth's position/velocity ephemeris
    sunfile: str
        A file containing the Sun's position/velocity ephemeris
    timefile: str
        A file containing time corrections for the TCB or TDB time coordinates.
    ephem: str
        The JPL ephemeris name, e.g., DE405
    units: str
        The time coordinate system, which can be either "TDB" or "TCB" (TCB is
        the default).
    ssonly: bool
        If True only return the initialised solar system ephemeris data.
        Default is False.
    timeonly: bool
        If True only return the initialised time correction ephemeris data.
        Default is False.
    filenames: bool
        If True return the paths to the ephemeris files. Default is False.

    Returns
    -------
    edat, sdat, filenames:
        The LAL EphemerisData object and TimeCorrectionData object.
    """

    earth = "earth00-40-{}.dat.gz".format(ephem) if earthfile is None else earthfile
    sun = "sun00-40-{}.dat.gz".format(ephem) if sunfile is None else sunfile

    filepaths = []

    if not timeonly:
        try:
            with MuteStream():
                # get full file path
                earthf = lalpulsar.PulsarFileResolvePath(earth)
                sunf = lalpulsar.PulsarFileResolvePath(sun)
                edat = lalpulsar.InitBarycenter(earthf, sunf)
            filepaths = [edat.filenameE, edat.filenameS]
        except RuntimeError:
            # try downloading the ephemeris files
            try:
                efile = download_ephemeris_file(LAL_EPHEMERIS_URL.format(earth))
                sfile = download_ephemeris_file(LAL_EPHEMERIS_URL.format(sun))
                edat = lalpulsar.InitBarycenter(efile, sfile)
                filepaths = [efile, sfile]
            except Exception as e:
                raise IOError("Could not read in ephemeris files: {}".format(e))

        if ssonly:
            return (edat, filepaths) if filenames else edat

    unit = None
    if timefile is None:
        if units.upper() in ["TCB", "TDB"]:
            unit = dict(TCB="te405", TDB="tdb")[units.upper()]
        else:
            raise ValueError("units must be TCB or TDB")

    time = "{}_2000-2040.dat.gz".format(unit) if timefile is None else timefile

    try:
        with MuteStream():
            # get full file path
            timef = lalpulsar.PulsarFileResolvePath(time)
            tdat = lalpulsar.InitTimeCorrections(timef)
        filepaths.append(timef)
    except RuntimeError:
        try:
            # try downloading the time coordinate file
            tfile = download_ephemeris_file(LAL_EPHEMERIS_URL.format(time))
            tdat = lalpulsar.InitTimeCorrections(tfile)
            filepaths.append(tfile)
        except Exception as e:
            raise IOError("Could not read in time correction file: {}".format(e))

    if timeonly:
        return (tdat, filepaths) if filenames else tdat
    else:
        return (edat, tdat, filepaths) if filenames else (edat, tdat)


def download_ephemeris_file(url):
    """
    Download and cache an ephemeris files from a given URL. If the file has
    already been downloaded and cached it will just be retrieved from the cache
    location.

    Parameters
    ----------
    url: str
        The URL of the file to download.
    """

    fname = os.path.basename(url)  # extract the file name
    fpath = os.path.join(EPHEMERIS_CACHE_DIR, fname)

    if os.path.isfile(fpath):
        # return previously cached file
        return fpath

    # try downloading the file
    try:
        ephdata = requests.get(url)
    except Exception as e:
        raise RuntimeError(f"Error downloading from {url}\n{e}")

    if ephdata.status_code != 200:
        raise RuntimeError(f"Error downloading from {url}")

    if not os.path.exists(EPHEMERIS_CACHE_DIR):
        try:
            os.makedirs(EPHEMERIS_CACHE_DIR)
        except OSError:
            if not os.path.exists(EPHEMERIS_CACHE_DIR):
                raise
    elif not os.path.isdir(EPHEMERIS_CACHE_DIR):
        raise OSError(f"Cache directory {EPHEMERIS_CACHE_DIR} is not a directory")

    # write out file to cache
    with open(fpath, "wb") as fp:
        fp.write(ephdata.content)

    return fpath


def check_for_tempo2():
    """
    Check whether the `libstempo <https://github.com/vallis/libstempo>`_
    package (v2.4.2 or greater), and therefore also TEMPO2, is available.
    """

    from packaging import version

    try:
        import libstempo

        hastempo2 = (
            True
            if version.parse(libstempo.__version__) >= version.parse("2.4.2")
            else False
        )
    except ImportError:
        hastempo2 = False

    return hastempo2


def relative_topdir(path, reference):
    """
    Returns the top-level directory name of a path relative to a reference.
    """

    try:
        return os.path.relpath(
            pathlib.Path(path).resolve(), pathlib.Path(reference).resolve()
        )
    except ValueError as exc:
        exc.args = (f"cannot format {path} relative to {reference}",)
        raise


def sighandler(signum, frame):
    # perform periodic eviction with exit code 77
    # see https://git.ligo.org/lscsoft/bilby_pipe/-/commit/c63c3e718f20ce39b0340da27fb696c49409fcd8  # noqa: E501
    sys.exit(CHECKPOINT_EXIT_CODE)


class MuteStream(object):
    """
    Class used to mute the output from a stream, e.g., ``stderr`` or
    ``stdout``.

    This is heavily based on the StackOverflow answer at
    https://stackoverflow.com/a/29834357/1862861, but only mutes and doesn't
    capture the output from the given stream.

    Parameters
    ----------
    stream:
        The stream to be muted. Defaults to ``sys.stderr``.
    """

    def __init__(self, stream=None):
        # don't do this if in a Jupyter notebook as it doesn't work due to how
        # the notebook captures outputs (the check for whether in a notebook is
        # from https://stackoverflow.com/a/37661854/1862861)
        self.in_notebook = "ipykernel" in sys.modules

        if not self.in_notebook:
            self.origstream = sys.stderr if stream is None else stream
            self.origstreamfd = self.origstream.fileno()
            # Create a pipe so the stream can be captured:
            self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        if not self.in_notebook:
            self.start()
        return self

    def __exit__(self, type, value, traceback):
        if not self.in_notebook:
            self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """

        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)

    def stop(self):
        """
        Stop capturing the stream data.
        """
        # Flush the streams
        self.origstream.flush()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)


def parse_args(input_args, parser):
    """
    Parse an argument list using parser generated by create_parser(). This
    function is based on the bilby_pipe.utils.parse_args function from the
    bilby_pipe package.

    Parameters
    ----------
    input_args: list
        A list of arguments

    Returns
    -------
    args: argparse.Namespace
        A simple object storing the input arguments
    unknown_args: list
        A list of any arguments in `input_args` unknown by the parser

    """

    if len(input_args) == 0:
        raise IOError("No command line arguments provided")

    ini_file = input_args[0]
    if os.path.isfile(ini_file) is False:
        if os.path.isfile(os.path.basename(ini_file)):
            input_args[0] = os.path.basename(ini_file)

    args, unknown_args = parser.parse_known_args(input_args)
    return args, unknown_args


def convert_string_to_dict(string, key=None):
    """
    Convert a string repr of a string to a python dictionary. This is based on
    bilby_pipe.utils.convert_string_to_dict from the bilby_pipe package.

    Parameters
    ----------
    string: str
        The string to convert
    key: str (None)
        A key, used for debugging
    """

    if string == "None":
        return None

    string = strip_quotes(string)
    # Convert equals to colons
    string = string.replace("=", ":")
    string = string.replace(" ", "")

    string = re.sub(r'([A-Za-z/\.0-9\-\+][^\[\],:"}]*)', r'"\g<1>"', string)

    # Force double quotes around everything
    string = string.replace('""', '"')

    # Evaluate as a dictionary of str: str
    try:
        dic = ast.literal_eval(string)
        if isinstance(dic, str):
            raise TypeError(f"Unable to format {string} into a dictionary")
    except (ValueError, SyntaxError) as e:
        if key is not None:
            raise TypeError(f"Error {e}. Unable to parse {key}: {string}")
        else:
            raise TypeError(f"Error {e}. Unable to parse {string}")

    # Convert values to bool/floats/ints where possible
    dic = convert_dict_values_if_possible(dic)

    return dic


def convert_dict_values_if_possible(dic):
    """
    Taken from bilby_pipe.utils.convert_dict_values_if_possible from the
    bilby_pipe package.
    """

    for key in dic:
        if isinstance(dic[key], str) and dic[key].lower() == "true":
            dic[key] = True
        elif isinstance(dic[key], str) and dic[key].lower() == "false":
            dic[key] = False
        elif isinstance(dic[key], str):
            dic[key] = string_to_int_float(dic[key])
        elif isinstance(dic[key], dict):
            dic[key] = convert_dict_values_if_possible(dic[key])
    return dic


def strip_quotes(string):
    """
    Taken from bilby_pipe.utils.strip_quotes from the bilby_pipe package.
    """

    try:
        return string.replace('"', "").replace("'", "")
    except AttributeError:
        return string


def string_to_int_float(s):
    """
    Taken from bilby_pipe.utils.string_to_int_float from the bilby_pipe
    package.
    """

    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
