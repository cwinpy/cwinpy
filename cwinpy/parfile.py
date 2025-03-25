import os
import pathlib
import re

import astropy.coordinates as coords
import lal
import lalpulsar
import numpy as np
from astropy import units as u
from astropy.time import Time

# set units of parameters in the PulsarParameters structure
PPUNITS = {
    "F": u.Hz,  # Hz
    "P": u.s,  # seconds
    "DIST": u.m,  # metres
    "PX": u.rad,  # radians
    "DM": u.pc / (u.cm) ** 3,  # cm^-3 pc
    "DM1": u.pc / (u.cm**3 * u.yr),  # pc cm^-3 yr^-1
    "RA": u.rad,  # radians
    "RAJ": u.rad,  # radians
    "DEC": u.rad,  # radians
    "DECJ": u.rad,  # radians
    "PMRA": u.rad / u.s,  # rad/s
    "PMDEC": u.rad / u.s,  # rad/s
    "ELONG": u.rad,  # rad
    "ELAT": u.rad,  # rad
    "PMELONG": u.rad / u.s,  # rad/s
    "PMELAT": u.rad / u.s,  # rad/s
    "BETA": u.rad,  # rad
    "LAMBDA": u.rad,  # rad
    "PMBETA": u.rad / u.s,  # rad/s
    "PMLAMBDA": u.rad / u.s,  # rad/s
    "PEPOCH": u.s,  # GPS seconds
    "POSEPOCH": u.s,  # GPS seconds
    "DMEPOCH": u.s,  # GPS seconds
    "GLEP": u.s,  # GPS seconds
    "GLPH": u.rad,  # rad
    "GLF0": u.Hz,  # Hz
    "GLF1": u.Hz / u.s,  # Hz/s
    "GLF2": u.Hz / u.s**2,  # Hz s^-2
    "GLF0D": u.Hz,  # Hz
    "GLTD": u.s,  # sec
    "A1": u.s,  # light seconds
    "OM": u.rad,  # rad
    "PB": u.s,  # seconds
    "T0": u.s,  # GPS seconds
    "TASC": u.s,  # GPS seconds
    "EPS1": u.dimensionless_unscaled,
    "EPS2": u.dimensionless_unscaled,
    "GAMMA": u.s,  # seconds
    "OMDOT": u.rad / u.s,  # rad/s
    "XDOT": u.s / u.s,  # light seconds/sec
    "PBDOT": u.s / u.s,  # s/s
    "EDOT": 1.0 / u.s,  # 1/sec
    "EPSDOT1": 1.0 / u.s,  # 1/sec
    "EPSDOT2": 1.0 / u.s,  # 1/sec
    "XPBDOT": u.s / u.s,  # s/s
    "SINI": u.dimensionless_unscaled,
    "MTOT": u.kg,  # kg
    "M2": u.kg,  # kg
    "DR": u.dimensionless_unscaled,
    "DTHETA": u.dimensionless_unscaled,
    "SHAPMAX": u.dimensionless_unscaled,
    "A1_2": u.s,  # light seconds
    "A1_3": u.s,  # light seconds
    "OM_2": u.rad,  # radians
    "OM_3": u.rad,  # radians
    "PB_2": u.s,  # seconds
    "PB_3": u.s,  # seconds
    "T0_2": u.s,  # GPS seconds
    "T0_3": u.s,  # GPS seconds
    "FB": u.Hz,  # Hz
    "A0": u.s,  # seconds
    "B0": u.s,  # seconds
    "D_AOP": 1.0 / u.rad,  # 1/rad
    "KIN": u.rad,  # radians
    "KOM": u.rad,  # radians
    "WAVE_OM": u.Hz,  # Hz
    "WAVEEPOCH": u.s,  # GPS seconds
    "WAVESIN": u.s,  # seconds
    "WAVECOS": u.s,  # seconds
    "START": u.s,  # GPS seconds
    "FINISH": u.s,  # GPS seconds
    "TRES": u.s,  # seconds
    "H0": u.dimensionless_unscaled,
    "APLUS": u.dimensionless_unscaled,
    "ACROSS": u.dimensionless_unscaled,
    "PHI0": u.rad,  # radians
    "PSI": u.rad,  # radians
    "COSIOTA": u.dimensionless_unscaled,
    "IOTA": u.rad,  # radians
    "C22": u.dimensionless_unscaled,
    "C21": u.dimensionless_unscaled,
    "PHI22": u.rad,  # radians
    "PHI21": u.rad,  # radians
    "CGW": u.dimensionless_unscaled,
    "LAMBDAPIN": u.rad,  # radians
    "COSTHETA": u.dimensionless_unscaled,
    "THETA": u.rad,
    "I21": u.dimensionless_unscaled,
    "I31": u.dimensionless_unscaled,
    "Q22": u.kg * u.m**2,  # kg m^2
    "H0_F": u.dimensionless_unscaled,
    "HPLUS": u.dimensionless_unscaled,
    "HCROSS": u.dimensionless_unscaled,
    "PSITENSOR": u.rad,  # radians
    "PHI0TENSOR": u.rad,  # radians
    "HSCALARB": u.dimensionless_unscaled,
    "HSCALARL": u.dimensionless_unscaled,
    "PSISCALAR": u.rad,  # radians
    "PHI0SCALAR": u.rad,  # radians
    "HVECTORX": u.dimensionless_unscaled,
    "HVECTORY": u.dimensionless_unscaled,
    "PSIVECTOR": u.rad,  # radians
    "PHI0VECTOR": u.rad,  # radians
    "HPLUS_F": u.dimensionless_unscaled,
    "HCROSS_F": u.dimensionless_unscaled,
    "PSITENSOR_F": u.rad,  # radians
    "PHI0TENSOR_F": u.rad,  # radians
    "HSCALARB_F": u.dimensionless_unscaled,
    "HSCALARL_F": u.dimensionless_unscaled,
    "PSISCALAR_F": u.rad,  # radians
    "PHI0SCALAR_F": u.rad,  # radians
    "HVECTORX_F": u.dimensionless_unscaled,
    "HVECTORY_F": u.dimensionless_unscaled,
    "PSIVECTOR_F": u.rad,  # radians
    "PHI0VECTOR_F": u.rad,  # radians
    "TRANSIENTSTARTTIME": u.s,  # GPS seconds
    "TRANSIENTTAU": u.s,  # seconds
}


# set units of parameters in a TEMPO-style parameter file if different from above
TEMPOUNITS = {
    "DIST": u.kpc,  # kpc
    "PX": u.mas,  # milliarcsecs
    "RA": u.hourangle,  # hh:mm:ss.s
    "RAJ": u.hourangle,  # hh:mm:ss.s
    "DEC": u.deg,  # hh:mm:ss.s
    "DECJ": u.deg,  # hh:mm:ss.s
    "PMRA": u.mas / u.yr,  # milliarcsecs/year
    "PMDEC": u.mas / u.yr,  # milliarcsecs/year
    "ELONG": u.deg,  # degrees
    "ELAT": u.deg,  # degrees
    "PMELONG": u.mas / u.yr,  # milliarcsecs/year
    "PMELAT": u.mas / u.yr,  # milliarcsecs/year
    "BETA": u.deg,  # degrees
    "LAMBDA": u.deg,  # degrees
    "PMBETA": u.mas / u.yr,  # milliarcsecs/year
    "PMLAMBDA": u.mas / u.yr,  # milliarcsecs/year
    "PEPOCH": u.d,  # MJD(TT) (day)
    "POSEPOCH": u.d,  # MJD(TT) (day)
    "DMEPOCH": u.d,  # MJD(TT) (day)
    "GLEP": u.d,  # MJD(TT) (day)
    "GLTD": u.d,  # days
    "OM": u.deg,  # degs
    "PB": u.d,  # day
    "T0": u.d,  # MJD(TT) (day)
    "TASC": u.d,  # MJD(TT) (day)
    "OMDOT": u.deg / u.yr,  # deg/yr
    "MTOT": u.solMass,  # M_sun
    "M2": u.solMass,  # M_sun
    "OM_2": u.deg,  # degrees
    "OM_3": u.deg,  # degrees
    "PB_2": u.d,  # days
    "PB_3": u.d,  # days
    "T0_2": u.d,  # MJD(TT) (days)
    "T0_3": u.d,  # MJD(TT) (days)
    "D_AOP": 1.0 / u.arcsec,  # 1/arcsec
    "KIN": u.deg,  # degrees
    "KOM": u.deg,  # degrees
    "WAVEEPOCH": u.d,  # MJD(TT) (days)
    "START": u.d,  # MJD(TT) (days)
    "FINISH": u.d,  # MJD(TT) (days)
    "TRES": u.us,  # microsecs
    "TRANSIENTSTARTTIME": u.d,  # MJD(TT) (day)
    "TRANSIENTTAU": u.d,  # days
}


# set units of error values in tempo if different from above
TEMPOERRUNITS = {
    "RA": u.s,  # second
    "RAJ": u.s,  # second
    "DEC": u.arcsec,  # arcsecond
    "DECJ": u.arcsec,  # arcsecond
}


# for certain binary parameters there is an anomoly that their value
# may have been rescaled (I think this is a hangover from a TEMPO definition compared to
# the TEMPO2 definition)
BINARYUNITS = ["XDOT", "PBDOT", "EPS1DOT", "EPS2DOT", "XPBDOT"]

# the names of epoch parameters that are held as GPS times, but must be converted back to
# MJD for a TEMPO-style par file
EPOCHPARS = [
    "POSEPOCH",
    "PEPOCH",
    "WAVEEPOCH",
    "T0",
    "TASC",
    "T0_2",
    "T0_3",
    "START",
    "FINISH",
    "DMEPOCH",
    "GLEP",
    "TRANSIENTSTARTTIME",
]

# the names of parameters that are held in seconds, but can be converted back into days if required
TIMESCALEPARS = [
    "PB",
    "PB_2",
    "PB_3",
    "GLTD",
    "TRANSIENTSTARTTIME",
]

# names of parameters that are actually stored as integers (otherwise store as float)
INTPARAMS = ["NTOA"]


# parameter aliases; entries are ``aliaspar: (realpar, getfunc, setfunc)``
_aliases = {}


def is_alias_param(name):
    """
    Check if ``name`` is an alias parameter name.
    """

    return name.startswith("ALIAS_")


def add_alias(aliaspar, getfunc, realpar, setfunc):
    """
    Add an alias for a PulsarParameters paramter.

    Parameters
    ----------
    aliaspar: str
        Name of the alias parameter; must start with ``ALIAS_``.
    getfunc: (pp: PulsarParameters) -> float
        Function which returns the value of the alias parameter,
        computed from real parameters in a PulsarParameters instance.
    realpar: str
        Name of the real parameter being aliased.
    setfunc: (aliasval: float, pp: PulsarParameters) -> float
        Function which returns the value of the real parameter,
        computed from the alias parameter value and other real
        parameters in a PulsarParameters instance.
    """

    if not is_alias_param(aliaspar):
        raise ValueError(
            "Alias parameter name '{}' must start with 'ALIAS_'".format(aliaspar)
        )

    _aliases[aliaspar.upper()] = (realpar, getfunc, setfunc)


def get_real_param_from_alias(name):
    """
    Return the real parameter name being aliases by ``name``. If not an alias, return ``name``.
    """

    if is_alias_param(name):
        realpar, getfunc, setfunc = _aliases[name.upper()]
        return realpar
    else:
        return name


# standard parameter aliases
add_alias(
    "ALIAS_N",
    lambda pp: pp["F0"] * pp["F2"] / pp["F1"] ** 2,
    "F2",
    lambda n, pp: n * pp["F1"] ** 2 / pp["F0"],
)

# obliquity from PINT file ecliptic.dat
OBL = {
    "TEMPO": 84381.412 * u.arcsec,
    "TEMPO2": 84381.4059 * u.arcsec,
}


class PulsarEcliptic(coords.BaseCoordinateFrame):
    """
    A Pulsar Ecliptic coordinate system is defined by rotating ICRS coordinate
    about x-axis by obliquity angle. Historical, This coordinate is used by
    tempo/tempo2 for a better fitting error treatment.
    The obliquity angle values respect to time are given in the file named
    "ecliptic.dat" in the pint/datafile directory.

    This is based on the :class:`~pint.pulsar_ecliptic.PulsarEcliptic` class
    from PINT.
    """

    default_representation = coords.SphericalRepresentation
    default_differential = coords.SphericalCosLatDifferential
    obliquity = coords.QuantityAttribute(
        default=84381.4059 * u.arcsecond, unit=u.arcsec
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _ecliptic_rotation_matrix_pulsar(obl) -> np.ndarray:
    """
    Here we only do the obliquity angle rotation. Astropy will add the
    precession-nutation correction.

    This is copied from PINT.
    """
    return coords.matrix_utilities.rotation_matrix(obl, "x")


@coords.frame_transform_graph.transform(
    coords.DynamicMatrixTransform, coords.ICRS, PulsarEcliptic
)
def icrs_to_pulsarecliptic(
    from_coo: coords.BaseCoordinateFrame, to_frame: coords.BaseCoordinateFrame
) -> np.ndarray:
    return _ecliptic_rotation_matrix_pulsar(to_frame.obliquity)


@coords.frame_transform_graph.transform(
    coords.DynamicMatrixTransform, PulsarEcliptic, coords.ICRS
)
def pulsarecliptic_to_icrs(
    from_coo: coords.BaseCoordinateFrame, to_frame: coords.BaseCoordinateFrame
) -> np.ndarray:
    return icrs_to_pulsarecliptic(to_frame, from_coo).T


class PulsarParameters:
    keynames = []  # parameter names in PulsarParameters structure
    length = 0  # number of parameters
    _pulsarparameters = None

    def __init__(self, pp=None):
        """
        A class to wrap the SWIG-wrapped ``lalpulsar.PulsarParameters`` structure.

        This class lets you access the structure in a more Pythonic way, as
        well as providing a nice format for holding pulsar (``.par``) parameter
        files.

        The class can be used to set numerical values (double precision,
        unsigned integers), strings, or vectors of floating point values, e.g.:

        >>> pp = PulsarParameters()   # an empty structure
        >>> pp['DECJ'] = 0.23         # set a numerical value
        >>> pp['BINARY'] = 'BT'       # set a string value
        >>> pp['F'] = [10.2, 1.4e-11] # set a vector of float values

        Examples
        --------

        An example of initialising the class using a parameter file is:

        >>> pppy = PulsarParameters('apulsar.par')

        or, equivalently with:

        >>> pppy = PulsarParameters()
        >>> pppy.read('apulsar.par')

        Parameters
        ----------
        pp: PulsarParameters, str
            A ``lalpulsar.PulsarParameters`` structure, or a string giving the
            path to a Tempo-style (``.par``) pulsar parameter file. If nothing
            is given then an empty :class:`~cwinpy.parfile.PulsarParameters`
            structure is created. The
            :meth:`~cwinpy.parfile.PulsarParameters.read` method can
            subsequently be used to read in a ``.par`` file, or parameters can
            be added.
        """

        # if pp is None create empty PulsarParameters structure
        if pp is None:
            self._pulsarparameters = lalpulsar.PulsarParameters()
        else:
            # check if pp is a pulsar parameters type or a (par file)
            if not isinstance(pp, lalpulsar.PulsarParameters) and (
                isinstance(pp, str) or isinstance(pp, pathlib.Path)
            ):
                if os.path.isfile(pp):
                    # try reading in file
                    self.read(pp)
                else:
                    raise ValueError("Input string does not point to a file")
            elif isinstance(pp, lalpulsar.PulsarParameters):
                self._pulsarparameters = pp
            else:
                raise ValueError(
                    "Expected 'lalpulsar.PulsarParameters' type, string, or None"
                )

    def __len__(self):
        _ = self.keys()  # length is counted in the keys() method

        return self.length

    def __str__(self):
        return self.pp_to_str()

    def __repr__(self):
        return self.pp_to_str()

    def __getitem__(self, key):
        """
        Get value from pulsar parameters
        """

        if self._pulsarparameters is None:
            return None

        # check if the key is asking for an aliased parameter;
        # if so, get value of aliased parameter from real parameters
        if is_alias_param(key.upper()):
            realpar, getfunc, setfunc = _aliases[key.upper()]
            return getfunc(self)

        # check if key finishes with "_ERR", in which case check for error value
        geterr = False
        tkey = key
        if key[-4:].upper() == "_ERR":
            geterr = True
            tkey = key[:-4]  # get the actual parameter key name

        # check if key is asking for equatorial coordinate value when only
        # ecliptic coordinates exist
        if key.upper() in ["RAJ", "DECJ", "RA", "DEC", "PMRA", "PMDEC"]:
            if not lalpulsar.PulsarCheckParam(self._pulsarparameters, key.upper()):
                poskeys = [["ELONG", "ELAT"], ["LAMBDA", "BETA"]]

                for pvals in poskeys:
                    pvals.extend([f"PM{p}" for p in pvals])

                    if any([self[p] is not None for p in pvals]):
                        longkey = pvals[0]
                        latkey = pvals[1]
                        pmlongkey = pvals[2]
                        pmlatkey = pvals[3]

                        long = 0.0 if self[longkey] is None else self[longkey]
                        lat = 0.0 if self[latkey] is None else self[latkey]
                        pmlong = 0.0 if self[pmlongkey] is None else self[pmlongkey]
                        pmlat = 0.0 if self[pmlatkey] is None else self[pmlatkey]

                        eclcoords = coords.SkyCoord(
                            obliquity=OBL[
                                "TEMPO" if self["T2CMETHOD"] == "TEMPO" else "TEMPO2"
                            ],
                            lon=long * PPUNITS[longkey],
                            lat=lat * PPUNITS[latkey],
                            pm_lon_coslat=pmlong * PPUNITS[pmlongkey],
                            pm_lat=pmlat * PPUNITS[pmlatkey],
                            obstime=Time(self["POSEPOCH"], format="gps"),
                            frame=PulsarEcliptic,
                        ).transform_to(coords.ICRS)

                        if key.upper() == "RAJ":
                            return eclcoords.ra.to(PPUNITS[key.upper()]).value
                        elif key.upper() == "DECJ":
                            return eclcoords.dec.to(PPUNITS[key.upper()]).value
                        elif key.upper() == "PMRA":
                            return eclcoords.pm_ra_cosdec.to(PPUNITS[key.upper()]).value
                        else:
                            return eclcoords.pm_dec.to(PPUNITS[key.upper()]).value
                else:
                    return None

        # check if the key is asking for an individual parameter from a vector parameter
        # (e.g. 'F0' gets the first value from the 'F' vector.
        # NOTE: this is problematic for glitch parameters, e.g., GLF0, which could provide
        # values to multiple glitches, so this cannot be used to get individual glitch
        # parameters).
        sname = re.sub(r"_\d", "", tkey) if "_" in tkey else re.sub(r"\d", "", tkey)
        sidx = None
        indkey = None
        if sname != tkey and tkey[0:2] != "GL":
            # check additional index is an integer
            try:
                sidx = (
                    int(tkey.split("_")[-1]) if "_" in tkey else int(tkey[len(sname) :])
                )
            except ValueError:
                pass

            # change tkey for checking parameter exists
            if sidx is not None:
                indkey = tkey  # key with index
                tkey = sname

            # check if parameter key is present otherwise switch back to original name
            # (needed for, e.g., 'H0', 'PHI0', ...)
            if not lalpulsar.PulsarCheckParam(self._pulsarparameters, tkey):
                tkey = indkey

        # check if parameter given by the key is present
        if not lalpulsar.PulsarCheckParam(self._pulsarparameters, tkey):
            return None

        # get type of parameter
        ptype = lalpulsar.PulsarGetParamType(self._pulsarparameters, tkey)

        if ptype == lalpulsar.PULSARTYPE_REAL8_t:
            if not geterr:
                value = lalpulsar.PulsarGetREAL8Param(self._pulsarparameters, tkey)
            else:
                value = lalpulsar.PulsarGetREAL8ParamErr(self._pulsarparameters, tkey)
        elif ptype == lalpulsar.PULSARTYPE_REAL8Vector_t:
            if not geterr:
                if sidx is None:
                    tmpvalue = lalpulsar.PulsarGetREAL8VectorParam(
                        self._pulsarparameters, tkey
                    )
                    value = (
                        tmpvalue.data
                    )  # 'data' in a REAL8Vector gets returned as a numpy array
                else:
                    value = lalpulsar.PulsarGetREAL8VectorParamIndividual(
                        self._pulsarparameters, indkey
                    )
            else:
                if sidx is None:
                    tmpvalue = lalpulsar.PulsarGetREAL8VectorParamErr(
                        self._pulsarparameters, tkey
                    )
                    value = tmpvalue.data
                else:
                    value = lalpulsar.PulsarGetREAL8VectorParamErrIndividual(
                        self._pulsarparameters, indkey
                    )
        elif ptype == lalpulsar.PULSARTYPE_string_t:
            if not geterr:
                value = lalpulsar.PulsarGetStringParam(self._pulsarparameters, tkey)
            else:
                raise ValueError("String-type cannot have an error")
        elif ptype == lalpulsar.PULSARTYPE_UINT4_t:
            if not geterr:
                value = lalpulsar.PulsarGetUINT4Param(self._pulsarparameters, tkey)
            else:
                raise ValueError("UINT4-type cannot have an error")
        else:
            raise ValueError("Unrecognised type")

        return value

    def __setitem__(self, key, value):
        """
        Set the value of a key
        """

        # check if the key is asking for an aliased parameter;
        # if so, set value of real parameter from aliased parameter
        if is_alias_param(key.upper()):
            realpar, getfunc, setfunc = _aliases[key.upper()]
            self.__setitem__(realpar, setfunc(value, self))
            return

        # if parameter exists remove it
        if lalpulsar.PulsarCheckParam(self._pulsarparameters, key):
            lalpulsar.PulsarRemoveParam(self._pulsarparameters, key)

        if isinstance(value, (float, int)):
            if value in INTPARAMS:
                lalpulsar.PulsarAddUINT4Param(self._pulsarparameters, key, int(value))
            else:
                lalpulsar.PulsarAddREAL8Param(self._pulsarparameters, key, float(value))
        elif isinstance(value, str):
            lalpulsar.PulsarAddStringParam(self._pulsarparameters, key, value)
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            tarray = lal.CreateREAL8Vector(len(value))
            for i, tv in enumerate(value):
                if isinstance(tv, (float, int)):
                    tarray.data[i] = float(tv)
                else:
                    raise ValueError("Non-float value in list or array")
            lalpulsar.PulsarAddREAL8VectorParam(self._pulsarparameters, key, tarray)
        else:
            raise ValueError("Data-type not one of know types")

        self._updated = True

    @property
    def updated(self):
        """
        Flag to state whether any new parameters have been set, or current
        parameters updated.
        """

        return getattr(self, "_updated", False)

    @staticmethod
    def convert_to_units(name, value):
        """
        Convert parameter values to equivalent dimensional versions.

        Parameters
        ----------
        name: str
            The name of the parameter to convert
        value:
            The value to the parameter to convert

        Return
        ------
        value
            A class:`~astropy.unit.Unit` object with dimensions for float
            parameters, or a list containing class:`~astropy.unit.Unit` object
            for a list or :class:`~numpy.ndarray`.
        """

        if isinstance(value, str):
            # don't make any changes for a string type
            return value

        uname = name.upper()

        ppunit = u.dimensionless_unscaled
        if uname in PPUNITS:
            ppunit = PPUNITS[uname]

        if isinstance(value, np.ndarray) or isinstance(value, list):
            cvalue = []
            bunit = ppunit
            for v in value:
                cvalue.append(v * ppunit)
                if uname in ["F", "FB", "P"]:  # frequency/period values
                    ppunit *= bunit  # increment unit (e.g. Hz -> Hz/s, Hz/s -> Hz/s^2)
        else:
            cvalue = value * ppunit

        return cvalue

    def convert_to_tempo_units(self, name, value, iserr=False):
        """
        Convert from PulsarParameter units to TEMPO-par-file units

        Parameters
        ----------
        name: str
            A parameter name
        value: float, array_like
            The value of the parameter
        iserr: bool
            State whether where converting the parameter's error value

        Returns
        -------
        value
            A class:`~astropy.unit.Unit` object with dimensions for float
            parameters, or a list containing class:`~astropy.unit.Unit` object
            for a list or :class:`~numpy.ndarray`.
        """

        from astropy.coordinates import ICRS, Angle
        from astropy.time import Time

        uname = name.upper()

        ppunit = None
        if uname in PPUNITS:
            ppunit = PPUNITS[uname]

        tempounit = None
        if uname in TEMPOUNITS:
            if not iserr:
                tempounit = TEMPOUNITS[uname]
            else:
                if uname not in TEMPOERRUNITS:
                    tempounit = TEMPOUNITS[uname]
                else:
                    tempounit = TEMPOERRUNITS[uname]

        if ppunit is None:
            if uname in BINARYUNITS:
                # for these binary parameters there is a TEMPO2 oddity that has to be corrected for
                if abs(value) / 1e-12 > 1e-7:
                    value /= 1e-12

            if isinstance(value, str):
                return value
            else:
                return value * u.dimensionless_unscaled

        # convert to dimensionful value
        pvalue = self.convert_to_units(uname, value)

        if ppunit == tempounit or tempounit is None:
            tempounit = ppunit

        if uname in ["RA", "RAJ"]:
            if not iserr:
                # convert right ascension in radians into a string output format
                c = ICRS(pvalue, 0.0 * u.rad)
                cvalue = c.ra.to_string(unit=tempounit, sep=":", precision=12, pad=True)
            else:
                angle = Angle(pvalue)
                cvalue = (
                    angle.hms[0] * (60.0**2) + angle.hms[1] * 60.0 + angle.hms[2]
                ) * tempounit
        elif uname in ["DEC", "DECJ"] and not iserr:
            c = ICRS(0.0 * u.rad, pvalue)
            cvalue = c.dec.to_string(unit=tempounit, sep=":", precision=12, pad=True)
        elif uname in EPOCHPARS and not iserr:
            if isinstance(pvalue, list):
                cvalue = []
                for pv in pvalue:
                    t = Time(pv, format="gps", scale="tt")
                    cvalue.append(t.mjd * tempounit)
            else:
                t = Time(pvalue, format="gps", scale="tt")
                cvalue = t.mjd * tempounit
        elif uname in BINARYUNITS and not iserr:
            # for these binary parameters there is a TEMPO2 oddity that has to be corrected for
            if abs(pvalue.value) / 1e-12 > 1e-7:
                pvalue.value /= 1e-12
            cvalue = pvalue.to(tempounit)
        else:
            # perform conversion
            if isinstance(pvalue, list):
                cvalue = []
                bunit = tempounit
                for pv in pvalue:
                    cvalue.append(pv.to(tempounit))
                    if uname in ["F", "FB", "P"]:  # frequency/period values
                        tempounit *= (
                            bunit  # increment unit (e.g. Hz -> Hz/s, Hz/s -> Hz/s^2)
                        )
            else:
                cvalue = pvalue.to(tempounit)

        return cvalue

    def parameter(self, name, withunits=False, tempounits=False):
        """
        Return the parameter given by name.

        Parameters
        ----------
        name: str
            The name of the parameter to return
        withunits: bool
            If True return the parameter in a form with its appropriate units
        tempounits: bool
            If True return the parameter converted into the units required in a
            Tempo-style parameter file
        """

        value = self[name]

        if value is None:
            return None

        if name.upper()[-4:] == "_ERR":
            uname = name.upper()[:-4]
            iserr = True
        else:
            uname = name
            iserr = False

        if tempounits:
            ovalue = self.convert_to_tempo_units(uname, value, iserr=iserr)
        elif withunits:
            ovalue = self.convert_to_units(uname, value)
        else:
            ovalue = value

        return ovalue

    def keys(self):
        """
        Return a list of the parameter names stored in the PulsarParameters structure
        """

        thisitem = self._pulsarparameters.head
        self.keynames = []  # clear any previous key names
        self.length = self._pulsarparameters.nparams
        while thisitem:
            tname = thisitem.name
            self.keynames.append(tname)

            thisitem = thisitem.next  # move on to next value

        # reverse the order of the names, so they're in the same order as read from a par file
        self.keynames = self.keynames[::-1]

        return self.keynames

    def values(self):
        """
        Return the values of each parameter in the structure
        """

        tvalues = []

        keys = self.keys()
        for key in keys:
            if lalpulsar.PulsarCheckParam(self._pulsarparameters, key):
                # get type of parameter
                ptype = lalpulsar.PulsarGetParamType(self._pulsarparameters, key)

                if ptype == lalpulsar.PULSARTYPE_REAL8_t:
                    value = lalpulsar.PulsarGetREAL8Param(self._pulsarparameters, key)
                elif ptype == lalpulsar.PULSARTYPE_REAL8Vector_t:
                    tmpvalue = lalpulsar.PulsarGetREAL8VectorParam(
                        self._pulsarparameters, key
                    )
                    value = (
                        tmpvalue.data
                    )  # 'data' in a REAL8Vector gets returned as a numpy array
                elif ptype == lalpulsar.PULSARTYPE_string_t:
                    value = lalpulsar.PulsarGetStringParam(self._pulsarparameters, key)
                elif ptype == lalpulsar.PULSARTYPE_UINT4_t:
                    value = lalpulsar.PulsarGetUINT4Param(self._pulsarparameters, key)
                else:
                    raise ValueError("UINT4-type cannot have an error")
            else:
                raise ValueError("Could not find {} in strcuture".format(key))

            tvalues.append(value)

        return tvalues

    def as_dict(self):
        """
        Return the contents (not error at the moment) of the structure as a dictionary
        """

        tdict = {}

        for tpair in self.items():
            tdict[tpair[0]] = tpair[1]

        return tdict

    def items(self):
        """
        Return list of item tuples for each parameter in the structure
        """

        tkeys = self.keys()
        tvalues = self.values()

        titems = []

        for (tk, tv) in zip(tkeys, tvalues):
            titems.append((tk, tv))

        return titems

    def read(self, filename):
        """
        Read a TEMPO-style parameter file into a PulsarParameters structure

        Parameters
        ----------
        filename: str
            The path to the pulsar ``.par`` file.
        """

        # remove existing pulsarparameters
        if self._pulsarparameters is not None:
            del self._pulsarparameters

        pp = lalpulsar.ReadTEMPOParFile(str(filename))

        if pp is None:
            raise IOError(
                "Problem reading in pulsar parameter file '{}'".format(filename)
            )

        self._pulsarparameters = pp

        # store copy of the contents of the par file
        with open(filename, "r") as fp:
            self._parcontent = fp.read()

    def get_error(self, name):
        """
        Return the error value for a particular parameter

        Args:
            name (str): the name of the parameter
        """

        try:
            if name.upper()[-4:] == "_ERR":
                return self[name.upper()]
            else:
                uname = name.upper() + "_ERR"
                return self[uname]
        except ValueError:
            return None

    def get_fitflag(self, name):
        """
        Return the "fit flag" (a 1 or 0 depending whether the parameter with fit for by TEMPO(2)
        in the ``.par`` file).

        Args:
            name (str): the name of the parameter
        """

        if name.upper() not in self.keys():
            return 0.0

        fitflag = lalpulsar.PulsarGetParamFitFlagAsVector(self._pulsarparameters, name)

        if len(fitflag.data) > 1 or isinstance(self[name.upper()], (list, np.ndarray)):
            return fitflag.data
        else:
            return fitflag.data[0]

    def pp_to_str(self, precision=19):
        """
        Convert the PulsarParameter structure to a string in the format of a
        Tempo-style ``.par`` file. If the structure contains information from a
        read-in ``.par`` file, that content will be returned.

        Parameters
        ----------
        precision: int
            The number of decimal places for an output value

        Return
        ------
        str
            The contents in Tempo-format
        """

        if hasattr(self, "_parcontent") and not self.updated:
            # if structure contains store par file return that
            return self._parcontent

        # output string format (set so that values should line up)
        mkl = (
            max([len(kn) for kn in self.keys()]) + 5
        )  # max key length for output alignment
        vlb = precision + 10  # allow extra space for minus sign/exponents
        outputstr = "{{name: <{0}}}{{value: <{1}}}{{fitflag}}\t{{error}}".format(
            mkl, vlb
        )

        parstr = ""

        parsedwaves = False

        # get names of parameters in file
        for item in self.items():
            key = item[0]
            value = item[1]

            if key in ["WAVESIN", "WAVECOS"] and parsedwaves:
                # already added the FITWAVES values
                continue

            # get error
            evalue = self.get_error(key)

            # check for required conversion back to TEMPO-par file units
            if key in TEMPOUNITS:
                uvalue = self.convert_to_tempo_units(key, value)
                if evalue is not None:
                    uevalue = self.convert_to_tempo_units(key, evalue, iserr=True)

                # just get values without units
                if isinstance(uvalue, list):
                    tvalue = []
                    tevalue = []
                    for tv, te in zip(uvalue, uevalue):
                        tvalue.append(tv.value)
                        tevalue.append(te.value)
                elif isinstance(uvalue, str):
                    tvalue = uvalue
                    tevalue = uevalue.value
                else:
                    tvalue = uvalue.value
                    tevalue = uevalue.value
            else:
                tvalue = value
                tevalue = evalue

            fitflag = None
            if evalue is not None:
                fitflag = self.get_fitflag(key)

            oevalue = ""  # default output error value
            ofitflag = " "  # default output fit flag value (a single space for alignment purposes)
            if isinstance(tvalue, list) or isinstance(tvalue, np.ndarray):
                idxoffset = 0
                idxsep = ""
                if key in [
                    "WAVESIN",
                    "WAVECOS",
                    "GLEP",
                    "GLPH",
                    "GLF0",
                    "GLF1",
                    "GLF2",
                    "GLF0D",
                    "GLTD",
                ]:
                    # the TEMPO variable name for these parameter start with an index a 1
                    idxoffset = 1

                    if key[:2] == "GL":
                        # glitch parameters have an "_" seperating the name from the index
                        idxsep = "_"

                if key in ["WAVESIN", "WAVECOS"]:
                    # do things differently for FITWAVES parameters
                    parsedwaves = True

                    if key == "WAVESIN":
                        wavesin = tvalue
                        wavecos = self["WAVECOS"]
                    else:
                        wavesin = self["WAVESIN"]
                        wavecos = tvalue

                    for ws, wc in zip(wavesin, wavecos):
                        precstrs = "{{0:.{}f}}".format(precision)  # print out float
                        if ws < 1e-6 or ws > 1e6:
                            # print out float in scientific notation
                            precstrs = "{{0:.{}e}}".format(precision)

                        precstrc = "{{0:.{}f}}".format(precision)  # print out float
                        if wc < 1e-6 or wc > 1e6:
                            # print out float in scientific notation
                            precstrc = "{{0:.{}e}}".format(precision)

                        outputdic = {}
                        outputdic["name"] = "WAVE{}{}".format(idxsep, idxoffset)
                        idxoffset += 1
                        outputdic["value"] = "{}\t{}".format(
                            precstrs.format(ws), precstrc.format(wc)
                        )
                        outputdic["fitflag"] = ""
                        outputdic["error"] = ""

                        parstr += outputstr.format(**outputdic).strip() + "\n"
                else:
                    for tv, te, tf in zip(tvalue, tevalue, fitflag):
                        precstr = "{{0:.{}f}}".format(precision)  # print out float
                        if tv < 1e-6 or tv > 1e6:
                            # print out float in scientific notation
                            precstr = "{{0:.{}e}}".format(precision)

                        precstre = "{{0:.{}f}}".format(precision)  # print out float
                        if te < 1e-6 or te > 1e6:
                            # print out float in scientific notation
                            precstre = "{{0:.{}e}}".format(precision)

                        outputdic = {}
                        outputdic["name"] = "{}{}{}".format(key, idxsep, idxoffset)
                        idxoffset += 1
                        outputdic["value"] = precstr.format(tv)
                        outputdic["fitflag"] = "1" if tf == 1 else ""
                        outputdic["error"] = precstre.format(te) if te != 0.0 else ""

                        parstr += outputstr.format(**outputdic).strip() + "\n"
            else:
                if isinstance(tvalue, float) or key in ["RA", "RAJ", "DEC", "DECJ"]:
                    if isinstance(tvalue, float):
                        precstr = "{{0:.{}f}}".format(precision)  # print out float
                        if tvalue < 1e-6 or tvalue > 1e6:
                            # print out float in scientific notation
                            precstr = "{{0:.{}e}}".format(precision)

                        ovalue = precstr.format(tvalue)
                    else:
                        ovalue = tvalue

                    precstre = "{{0:.{}f}}".format(precision)  # print out float
                    oevalue = ""
                    if tevalue is not None:
                        if tevalue != 0.0:
                            if tevalue < 1e-6 or tevalue > 1e6:
                                # print out float in scientific notation
                                precstre = "{{0:.{}e}}".format(precision)

                            oevalue = precstre.format(tevalue)

                            if fitflag is not None:
                                if fitflag == 1:
                                    ofitflag = "1"
                else:
                    ovalue = tvalue

                outputdic = {}
                outputdic["name"] = key
                outputdic["value"] = ovalue
                outputdic["fitflag"] = ofitflag
                outputdic["error"] = oevalue

                parstr += outputstr.format(**outputdic).strip() + "\n"

        return parstr

    def pp_to_par(self, filename, precision=19):
        """
        Output the PulsarParameter structure to a ``.par`` file.

        Parameters
        ----------
        filename: str
            The path to the output file
        precision: int
            The number of decimal places for an output value
        """

        try:
            fp = open(filename, "w")
        except IOError:
            raise IOError("Could not open file '{}' for writing".format(filename))

        fp.write(self.pp_to_str(precision=precision))
        fp.close()

    def PulsarParameters(self):
        """
        Return the PulsarParameters structure
        """

        return self._pulsarparameters

    def __deepcopy__(self, memo):
        """
        Create a copy of the parameters.
        """

        newpar = PulsarParameters()
        memo[id(self)] = newpar
        lalpulsar.PulsarCopyParams(self.PulsarParameters(), newpar.PulsarParameters())
        return newpar
