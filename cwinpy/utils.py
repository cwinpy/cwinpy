"""
A selection of general utility functions.
"""

import ctypes
import os
import string
import subprocess
from functools import reduce
from math import gcd

import lalpulsar
import numpy as np
from astropy import units as u
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy
from numba import jit, njit
from numba.extending import get_cython_function_address

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
    denoms = denoms.astype(np.int)

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

    # check that the file is ASCII text (see, e.g.,
    # https://stackoverflow.com/a/1446571/1862861)

    if os.path.isfile(parfile):
        msg = subprocess.Popen(
            ["file", "--mime", "--dereference", parfile], stdout=subprocess.PIPE
        ).communicate()[0]
        if "text/plain" in msg.decode():
            psr = PulsarParametersPy(parfile)
            # file must contain a frequency, right ascension, declination and
            # pulsar name
            # file must contain right ascension and declination
            if (
                psr["F"] is None
                or (psr["RAJ"] is None and psr["RA"] is None)
                or (psr["DECJ"] is None and psr["DEC"] is None)
                or get_psr_name(psr) is None
            ):
                return False

            return True

    return False


def get_psr_name(psr):
    """
    Get the pulsar name from the TEMPO(2)-style parameter files by trying the
    keys "PSRJ", "PSRB", "PSR", and "NAME" in that order of precedence.

    Parameters
    ----------
    psr: PulsarParameterPy
        A PulsarParameterPy object

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

    References
    ----------

    .. [1] B. Abbott, et al., Ap. J., 879, 10, 2019
       (`arXiv:1902.08507 <https://arxiv.org/abs/1902.08507>`_)

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

    References
    ----------

    .. [1] B. Abbott, et al., Ap. J., 879, 10, 2019
       (`arXiv:1902.08507 <https://arxiv.org/abs/1902.08507>`_)

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


def initialise_ephemeris(
    ephem="DE405", units="TCB", earthfile=None, sunfile=None, timefile=None
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
    """

    DOWNLOAD_URL = "https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/{}"

    unit = None
    if timefile is None:
        if units.upper() in ["TCB", "TDB"]:
            unit = dict(TCB="te405", TDB="tdb")[units.upper()]
        else:
            raise ValueError("units must be TCB or TDB")

    earth = "earth00-40-{}.dat.gz".format(ephem) if earthfile is None else earthfile
    sun = "sun00-40-{}.dat.gz".format(ephem) if sunfile is None else sunfile
    time = "{}_2000-2040.dat.gz".format(unit) if timefile is None else timefile

    try:
        edat = lalpulsar.InitBarycenter(earth, sun)
    except RuntimeError:
        # try downloading the ephemeris files
        try:
            from astropy.utils.data import download_file

            efile = download_file(DOWNLOAD_URL.format(earth), cache=True)
            sfile = download_file(DOWNLOAD_URL.format(sun), cache=True)
            edat = lalpulsar.InitBarycenter(efile, sfile)
        except Exception as e:
            raise IOError("Could not read in ephemeris files: {}".format(e))

    try:
        tdat = lalpulsar.InitTimeCorrections(time)
    except RuntimeError:
        try:
            # try downloading the time coordinate file
            from astropy.utils.data import download_file

            tfile = download_file(DOWNLOAD_URL.format(time), cache=True)
            tdat = lalpulsar.InitTimeCorrections(tfile)
        except Exception as e:
            raise IOError("Could not read in time correction file: {}".format(e))

    return edat, tdat
