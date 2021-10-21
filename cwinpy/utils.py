"""
A selection of general utility functions.
"""

import ctypes
import os
import string
import sys
from functools import reduce
from math import gcd

import lalpulsar
import numpy as np
from astropy import units as u
from bilby_pipe.utils import CHECKPOINT_EXIT_CODE
from numba import jit, njit
from numba.extending import get_cython_function_address

from .parfile import PulsarParameters

#: URL for LALSuite solar system ephemeris files
LAL_EPHEMERIS_URL = "https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/{}"

#: the current solar system ephemeris types in LALSuite
LAL_EPHEMERIS_TYPES = ["DE200", "DE405", "DE421", "DE430"]

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

#: aliases between GW detector prefixes a TEMPO2 observatory names
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
        return q22 * u.kg * u.m ** 2
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
                from astropy.utils.data import download_file

                efile = download_file(LAL_EPHEMERIS_URL.format(earth), cache=True)
                sfile = download_file(LAL_EPHEMERIS_URL.format(sun), cache=True)
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
            from astropy.utils.data import download_file

            tfile = download_file(LAL_EPHEMERIS_URL.format(time), cache=True)
            tdat = lalpulsar.InitTimeCorrections(tfile)
            filepaths.append(tfile)
        except Exception as e:
            raise IOError("Could not read in time correction file: {}".format(e))

    if timeonly:
        return (tdat, filepaths) if filenames else tdat
    else:
        return (edat, tdat, filepaths) if filenames else (edat, tdat)


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
        self.origstream = sys.stderr if stream is None else stream
        self.origstreamfd = self.origstream.fileno()
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
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
