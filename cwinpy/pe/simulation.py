"""
Generate simulations for running the hierarchical analysis.
"""

import os
from configparser import ConfigParser

import astropy.units as u
import bilby
import numpy as np
from astropy.coordinates import ICRS, Galactic, Galactocentric

from ..hierarchical import BaseDistribution
from ..parfile import PulsarParameters
from ..utils import ellipticity_to_q22, int_to_alpha, is_par_file
from .pe import pe_dag


class PEPulsarSimulationDAG(object):
    """
    This class will generate a HTCondor Dagman job to create a number of
    simulated gravitational-wave signals from pulsars. These signals will
    either be added into random Gaussian noise, or real heterodyned data.
    The generated signals have source parameters drawn from an input
    distribution. They can be generated using real pulsars parameters,
    including the pulsar distance, or using fake pulsars with sky
    locations and distances drawn from input distribution (by default the sky
    location distribution will be uniform on the sky).

    These signals will be analysed using the ``cwinpy_pe`` script to sample
    the posterior probability distributions of the required parameter space,
    including the mass quadrupole.

    If not supplied, the default priors that the parameter estimation will use
    are:

    >>> from bilby.core.prior import PriorDict, Uniform, Sine
    >>> import numpy as np
    >>> prior = PriorDict({
    ...     "h0": Uniform(0.0, 1e-22, name="h0"),
    ...     "iota": Sine(name="iota"),
    ...     "phi0": Uniform(0.0, np.pi, name="phi0"),
    ...     "psi": Uniform(0.0, np.pi / 2, name="psi"),
    ... })

    Parameters
    ----------
    ampdist: ``bilby.core.prior.Prior``, :class:`~cwinpy.hierarchical.BaseDistribution`
        The distribution from which the signal amplitude values will be drawn
        for the simulated signals. This can contain the gravitational-wave
        strain amplitude keyed with ``"h0"``, the mass quadrupole keyed with
        ``"q22"``, or the fiducial ellipticity keyed with ``"epsilon"``
        (``"ell"``, or ``"ellipticity"``).
    prior: str, dict
        This can be a single bilby-style prior dictionary, or a path to a file,
        giving a prior distribution to use for signal parameter estimation for
        all pulsars, or a dictionary keyed to pulsar names containing prior
        dictionaries or prior files for each pulsar. If not given the default
        distribution given above will used used for all pulsars.
    distance_err: float, dict
        If this is given, then a prior on distance will be added to the
        ``prior`` dictionary for parameter estimation. If this is a float, then
        that value will be taken as a fractional uncertainty on the pulsar
        distance and a Gaussian prior will be added using that as the standard
        deviation. If this is a dictionary it should be keyed to pulsar names
        and provide either a `bilby.core.prior.Prior` named ``"dist"`` or a
        standard deviation on the distance in kpc.
    parfiles: dict, str
        If using real pulsar parameter files pass a dictionary of paths to
        individual files keyed to the pulsar name, or pass the path to a
        directory containing the parameter files.
    overwrite_parameters: bool
        If using real/pre-created pulsar parameter files set this to False if
        you want any orientation or amplitude parameters that they contain to
        not be overwritten by those drawn from the supplied distributions.
        Default is True.
    datafiles: dict, str
        If using real data for each pulsar then pass a dictionary of paths to
        the data files keyed to the pulsar name (or detector then pulsar name),
        or a directory containing the heterodyned data files.
    npulsars: int
        The number of pulsars to include in the simulation.
    posdist: ``bilby.core.prior.PriorDict``
        The distribution from which to randomly draw pulsar positions if
        required. This defaults to a uniform distribution on the sky and
        uniform in distance between 0.1 and 10 kpc. If you want to specify a
        distribution in right ascension, declination and distance they must be
        in the prior dictionary with the keys ``"ra"`` (radians), ``"dec"``
        (radians) and ``"dist"`` (kpc), respectively. Alternatively, you
        can specify a distribution in
        :class:`~astropy.coordinates.Galactocentric` coordinates using the
        dictionary keys ``"x"``, ``"y"`` and ``"z"`` all in units of kpc, or
        :class:`~astropy.coordinates.Galactic` coordinates using the
        dictionary keys ``"l"`` (Galactic longitude in radians), ``"b"``
        (Galactic latitude in radians) and ``"dist"`` (kpc). These
        will be converted into the equivalent RA, declination and distance.
    oridist: ``bilby.core.prior.PriorDict``
        The distribution if the pulsar orientation parameters. This defaults
        to uniform over a hemisphere in inclination and polarisation angle,
        and uniform over :math:`\\pi` radians in rotational phase. The
        dictionary keys must be ``"phi0"`` (initial phase), ``"iota"``
        (inclination angle), and ``"psi"`` (polarisation angle), all in
        radians.
    fdist: ``bilby.core.prior.Prior``
        The distribution from which to draw pulsar spin frequencies if
        required. This defaults to a uniform distribution between 10 and 750
        Hz.
    basedir: str
        The base directory into which the simulations and outputs will be
        placed. If None then the current working directory will be used.
    detector: str, list
        A string, or list of strings, of detector prefixes for the
        simulated data. This defaults to a single detector - the LIGO Hanford
        Observatory - from which the simulated noise will be drawn from the
        advanced detector design sensitivity curve (e.g., [3]_).
    starttime: int, float, dict
        A GPS time, or dictionary of GPS times keyed to detectors, giving the
        start time for any simulated data being generated. If not given the
        start time defaults to 1000000000.
    endtime: int, float, dict
        A GPS time, or dictionary of GPS times keyed to detectors, giving the
        end time for any simulated data being generated. If not given the end
        time defaults to 1000086400.
    timestep: int, float
        The time step, in seconds, between data points for simulated data. If
        not given this defaults to 60.
    submit: bool
        Set whether to submit the Condor DAG or not.
    accountuser: str
        Value to give to the 'account_user' parameter in the Condor submit
        file. Default is to set no value.
    accountgroup: str
        Value to give to the 'account_user_group' parameter in the Condor
        submit file. Default is to set no value.
    getenv: bool
        Set the value for the ``'getenv'`` parameter in the Condor submit file.
        Default is False.
    sampler: str
        The sampler to use. Defaults to dynesty.
    sampler_kwargs: dict
        A dictionary of keyword arguments for the sampler. Defaults to None.
    outputsnr: bool
        Set whether to output the injected and recovered signal-to-noise
        ratios. Defaults to True.
    numba: bool
        Set whether or not to use the likelihood with numba enabled. Defaults
        to True.
    n_parallel: int
        Set the number of parallel sampler jobs to run for each pulsar, which
        will be combined to form the final samplers. Defaults to 1.
    """

    def __init__(
        self,
        ampdist=None,
        prior=None,
        distance_err=None,
        parfiles=None,
        overwrite_parameters=True,
        datafiles=None,
        npulsars=None,
        basedir=None,
        detector="H1",
        starttime=None,
        endtime=None,
        timestep=None,
        posdist=None,
        oridist=None,
        fdist=None,
        submit=False,
        accountuser=None,
        accountgroup=None,
        getenv=False,
        sampler="dynesty",
        sampler_kwargs=None,
        outputsnr=True,
        numba=True,
        n_parallel=1,
    ):
        if basedir is not None:
            self.basedir = basedir
            self.makedirs(basedir)
        else:
            self.basedir = os.getcwd()

        self.parfiles = parfiles
        self.overwrite = overwrite_parameters
        self.ampdist = ampdist
        self.distance_err = distance_err
        self.prior = prior

        # create pulsar parameter files if none are given
        if self.parfiles is None:
            if not isinstance(npulsars, int):
                raise TypeError("Number of pulsars must be an integer")

            if npulsars < 1:
                raise ValueError("A positive number of injection must be given")
            self.npulsars = int(npulsars)

        # set sky location, orientation and frequency distributions
        self.posdist = posdist
        self.oridist = oridist
        self.fdist = fdist

        self.create_pulsars()

        # check whether detectors or data files are specified
        self.detector = detector
        if self.detector is not None:
            if isinstance(self.detector, str):
                self.detector = [self.detector]
            if not isinstance(self.detector, list):
                raise TypeError("Detector must be a string or list of strings")
        self.datafiles = datafiles
        self.starttime = starttime
        self.endtime = endtime
        self.timestep = timestep

        # posterior sample results directory
        self.resultsdir = os.path.join(self.basedir, "results")
        self.makedirs(self.resultsdir)

        # create dag configuration file
        self.accountuser = accountuser
        self.accountgroup = accountgroup
        self.getenv = getenv
        self.submit = submit
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs
        self.outputsnr = outputsnr
        self.numba = numba
        self.n_parallel = n_parallel
        self.create_config()

        # create the DAG for cwinpy_knope jobs
        self.runner = pe_dag(config=self.config)
        self.runner.dag.build()

    @property
    def ampdist(self):
        return self._ampdist

    @ampdist.setter
    def ampdist(self, ampdist):
        if isinstance(ampdist, (bilby.core.prior.Prior, BaseDistribution)):
            if ampdist.name.lower() not in [
                "h0",
                "q22",
                "epsilon",
                "ell",
                "ellipticity",
            ]:
                raise KeyError(
                    "Amplitude distribution must contain 'h0', 'q22', or 'epsilon'"
                )
            self._ampdist = ampdist
        else:
            if ampdist is None:
                if self.parfiles is not None:
                    self._ampdist = ampdist
                else:
                    raise ValueError(
                        "An amplitude distribution must be set if not supplying parameter files"
                    )
            else:
                raise TypeError("Amplitude distribution must be a bilby Prior")

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, prior):
        if isinstance(prior, dict):
            if type(prior) == bilby.core.prior.PriorDict:
                self._prior = prior
            else:
                self._prior = {}
                for key, value in prior.items():
                    if isinstance(value, bilby.core.prior.Prior):
                        if value.name == key:
                            self._prior = bilby.core.prior.PriorDict(prior)
                            break
                    else:
                        # dictionary contains individual priors for pulsars
                        self._prior[key] = bilby.core.prior.PriorDict(value)

            if len(self._prior) == 0:
                raise ValueError("Prior is empty!")
        elif isinstance(prior, str):
            try:
                self._prior = bilby.core.prior.PriorDict(filename=prior)
            except FileNotFoundError:
                raise FileNotFoundError("Couldn't parse prior file")
        elif prior is None:
            # set default priors
            self._prior = bilby.core.prior.PriorDict(
                {
                    "h0": bilby.core.prior.Uniform(
                        0.0, 1e-22, name="h0", latex_label="$h_0$"
                    ),
                    "iota": bilby.core.prior.Sine(name="iota", latex_label=r"$\iota$"),
                    "phi0": bilby.core.prior.Uniform(
                        0.0, np.pi, name="phi0", latex_label=r"$\phi_0$"
                    ),
                    "psi": bilby.core.prior.Uniform(
                        0.0, np.pi / 2, name="psi", latex_label=r"$\psi$"
                    ),
                }
            )
        else:
            raise TypeError("Prior must be a dictionary-type object")

    @property
    def distance_err(self):
        return self._distance_err

    @distance_err.setter
    def distance_err(self, err):
        if (
            err is None
            or isinstance(err, float)
            or (isinstance(err, dict) and self.parfiles is not None)
        ):
            self._distance_err = err
        else:
            raise TypeError("Distance error is the wrong type")

    @property
    def parfiles(self):
        return self._parfiles

    @parfiles.setter
    def parfiles(self, parfiles):
        pfs = {}

        if parfiles is not None:
            if isinstance(parfiles, dict):
                pfs = parfiles

                for psr in pfs:
                    if not is_par_file(pfs[psr]):
                        raise IOError(
                            "{} is not a pulsar parameter file".format(pfs[psr])
                        )
            elif isinstance(parfiles, str):
                if os.path.isdir(parfiles):
                    if parfiles == os.path.join(self.basedir, "pulsars"):
                        raise ValueError(
                            "Parameter files directory must be different from "
                            "'{}', which is reserved for the output directories".format(
                                parfiles
                            )
                        )

                    for pf in os.listdir(parfiles):
                        parfile = os.path.join(parfiles, pf)
                        if is_par_file(parfile):
                            psr = PulsarParameters(parfile)

                            # add parfile to dictionary
                            for name in ["PSRJ", "PSRB", "PSR", "NAME"]:
                                if psr[name] is not None:
                                    pfs[psr[name]] = parfile
                else:
                    raise ValueError("Path is not a valid directory")
            else:
                raise TypeError(
                    "Must give directory or dictionary of pulsar parameter files"
                )
            self._parfiles = pfs
            self.npulsars = len(self._parfiles)
        else:
            self._parfiles = None
            self.npulsars = None

    @property
    def fdist(self):
        return self._fdist

    @fdist.setter
    def fdist(self, fdist):
        if fdist is None:
            # set default frequency distribution
            self._fdist = bilby.core.prior.Uniform(10.0, 750.0, name="frequency")
        elif isinstance(fdist, bilby.core.prior.Prior):
            self._fdist = fdist
        else:
            raise TypeError("Frequency distribution is not the correct type")

    @property
    def posdist(self):
        return self._posdist

    @posdist.setter
    def posdist(self, posdist):
        if posdist is None:
            # set default position distribution (0.1 to 10 kpc) and uniform on
            # the sky
            ddist = bilby.core.prior.Uniform(0.1, 10.0, name="dist")
            radist = bilby.core.prior.Uniform(0.0, 2.0 * np.pi, name="ra")
            decdist = bilby.core.prior.Cosine(name="dec")
            self._posdist = bilby.core.prior.PriorDict(
                {"dist": ddist, "ra": radist, "dec": decdist}
            )
            self._posdist_type = "equatorial"
        elif isinstance(posdist, bilby.core.prior.PriorDict):
            # check that distribution contains distance, ra and dec
            if self.parfiles is None:
                # check whether Galactic, Galactocentric or equatorial coordinates
                if set(["ra", "dec", "dist"]) == set(posdist.keys()):
                    self._posdist_type = "equatorial"
                elif set(["l", "b", "dist"]) == set(posdist.keys()):
                    self._posdist_type = "galactic"
                elif set(["x", "y", "z"]) == set(posdist.keys()):
                    self._posdist_type = "galactocentric"
                else:
                    raise ValueError(
                        "Position distribution '{}' does not contain all required values".format(
                            posdist.keys()
                        )
                    )
            else:
                self._posdist_type = "equatorial"
                if "dist" not in posdist:
                    raise KeyError("Position distribution must contain distance")

            self._posdist = posdist
        else:
            raise TypeError("Position distribution is not correct type")

    @property
    def oridist(self):
        return self._oridist

    @oridist.setter
    def oridist(self, oridist):
        if oridist is None:
            # set default orientation distribution
            phase = bilby.core.prior.Uniform(0.0, np.pi, name="phi0")
            psi = bilby.core.prior.Uniform(0.0, np.pi / 2.0, name="psi")
            iota = bilby.core.prior.Sine(name="iota")

            self._oridist = bilby.core.prior.PriorDict(
                {"phi0": phase, "psi": psi, "iota": iota}
            )
        elif isinstance(oridist, dict):
            # check that distribution contains phi0, psi and iota
            # and add defaults for any not included
            self._oridist = bilby.core.prior.PriorDict(oridist)

            if "phi0" not in oridist:
                self._oridist["phi0"] = bilby.core.prior.Uniform(
                    0.0, np.pi, name="phi0"
                )

            if "psi" not in oridist:
                self._oridist["psi"] = bilby.core.prior.Uniform(
                    0.0, np.pi / 2.0, name="psi"
                )

            if "iota" not in oridist:
                self._oridist["iota"] = bilby.core.prior.Sine(name="iota")
        else:
            raise TypeError("Orientation distribution is not correct type")

    def makedirs(self, dir):  # pragma: no cover
        """
        Make a directory tree recursively.
        """

        try:
            os.makedirs(dir, exist_ok=True)
        except Exception as e:
            raise IOError("Could not create directory: {}\n{}".format(dir, e))

    def create_pulsars(self):
        """
        Create the pulsar parameter files based on the supplied distributions.
        """

        # pulsar parameter/injection file directory
        self.pulsardir = os.path.join(self.basedir, "pulsars")
        self.makedirs(self.pulsardir)

        self.pulsars = {}
        self.priors = {}
        for i in range(self.npulsars):
            # generate fake pulsar parameters
            if self.posdist is not None:
                skyloc = self.posdist.sample()

                if self._posdist_type in ["galactocentric", "galactic"]:
                    # transform coordinates if required
                    gpos = (
                        Galactocentric(
                            x=skyloc["x"] * u.kpc,
                            y=skyloc["y"] * u.kpc,
                            z=skyloc["z"] * u.kpc,
                        )
                        if self._posdist_type == "galactocentric"
                        else Galactic(
                            l=skyloc["l"] * u.rad,  # noqa: E741
                            b=skyloc["b"] * u.rad,
                            distance=skyloc["dist"] * u.kpc,
                        )
                    )
                    eqpos = gpos.transform_to(ICRS)
                    skyloc["ra"] = eqpos.ra.rad
                    skyloc["dec"] = eqpos.dec.rad
                    skyloc["dist"] = eqpos.distance.value

            if self.fdist is not None:
                freq = self.fdist.sample()

            # generate orientation parameters
            orientation = self.oridist.sample()

            if self.parfiles is None:
                pulsar = PulsarParameters()
                pulsar["RAJ"] = skyloc["ra"]
                pulsar["DECJ"] = skyloc["dec"]
                pulsar["DIST"] = (skyloc["dist"] * u.kpc).to("m").value

                # set pulsar name from sky position
                rastr = "".join(
                    pulsar.convert_to_tempo_units("RAJ", pulsar["RAJ"]).split(":")[:2]
                )
                decstr = "".join(
                    pulsar.convert_to_tempo_units("DECJ", pulsar["DECJ"]).split(":")[:2]
                )
                decstr = decstr if decstr[0] == "-" else ("+" + decstr)
                pname = "J{}{}".format(rastr, decstr)
                pnameorig = str(pname)  # copy of original name
                counter = 1
                while pname in self.pulsars:
                    anum = int_to_alpha(counter)
                    pname = pnameorig + anum
                    counter += 1

                pulsar["PSRJ"] = pname
                pulsar["F"] = [freq]

                for param in ["psi", "iota", "phi0"]:
                    pulsar[param.upper()] = orientation[param]

                # output file name
                pfile = os.path.join(self.pulsardir, "{}.par".format(pname))
                injfile = pfile

                self.priors[pname] = self.prior
            else:
                pfile = list(self.parfiles.values())[i]
                pulsar = PulsarParameters(pfile)
                pname = list(self.parfiles.keys())[i]
                injfile = os.path.join(self.pulsardir, "{}.par".format(pname))

                if pulsar["DIST"] is None or (self.overwrite and "dist" in skyloc):
                    # add distance if not present in parameter file
                    pulsar["DIST"] = (skyloc["dist"] * u.kpc).to("m").value

                for param in ["psi", "iota", "phi0"]:
                    # add orientation values if not present in parameter file
                    if pulsar[param.upper()] is None or (
                        self.overwrite and param in orientation
                    ):
                        pulsar[param.upper()] = orientation[param]

                if isinstance(self.prior, dict) and pname in self.prior:
                    # there are individual pulsar priors
                    self.priors[pname] = self.prior[pname]
                else:
                    self.priors[pname] = self.prior

            # check whether to add distance uncertainties to priors
            if self.distance_err is not None:
                dist = None
                if isinstance(self.distance_err, float):
                    # set truncated Gaussian prior
                    dist = bilby.core.prior.TruncatedGaussian(
                        pulsar["DIST"],
                        self.distance_err * pulsar["DIST"],
                        0.0,
                        np.inf,
                        name="dist",
                    )
                elif pname in self.distance_err:
                    if isinstance(self.distance_err[pname], float):
                        dist = bilby.core.prior.TruncatedGaussian(
                            pulsar["DIST"],
                            self.distance_err[pname] * pulsar["DIST"],
                            0.0,
                            np.inf,
                            name="dist",
                        )
                    elif isinstance(self.distance_err[pname], bilby.core.prior.Prior):
                        dist = self.distance_err[pname]

                if dist is not None and "dist" not in self.priors[pname]:
                    # only add distance if not already specified
                    self.priors[pname]["dist"] = dist

            # set amplitude value
            amp = self.ampdist.sample() if self.ampdist is not None else None
            if self.ampdist.name == "q22" and (pulsar["Q22"] is None or self.overwrite):
                pulsar["Q22"] = amp
            elif self.ampdist.name == "h0" and (pulsar["H0"] is None or self.overwrite):
                pulsar["H0"] = amp
            elif (self.ampdist.name.lower() in ["epsilon", "ell", "ellipticity"]) and (
                pulsar["Q22"] is None or self.overwrite
            ):
                # convert ellipticity to Q22 using fiducial moment of inertia
                pulsar["Q22"] = ellipticity_to_q22(amp)

            with open(injfile, "w") as fp:
                fp.write(str(pulsar))

            self.pulsars[pname] = {}
            self.pulsars[pname]["file"] = pfile
            self.pulsars[pname]["injection_file"] = injfile
            self.pulsars[pname]["parameters"] = pulsar

    def create_config(self):
        """
        Create the configuration parser for the DAG.
        """

        self.config = ConfigParser()

        self.config["run"] = {"basedir": self.basedir}

        self.config["pe_dag"] = {"build": False}
        self.config["pe_dag"] = {"submitdag": self.submit}

        self.config["pe_job"] = {}
        self.config["pe_job"]["getenv"] = str(self.getenv)

        if self.accountgroup is not None:  # pragma: no cover
            self.config["pe_job"]["accounting_group"] = self.accountgroup
        if self.accountuser is not None:  # pragma: no cover
            self.config["pe_job"]["accounting_group_user"] = self.accountuser

        self.config["ephemerides"] = {}
        self.config["ephemerides"]["pulsars"] = str(
            [self.pulsars[pname]["file"] for pname in self.pulsars]
        )
        self.config["ephemerides"]["injections"] = str(
            [self.pulsars[pname]["injection_file"] for pname in self.pulsars]
        )

        self.config["pe"] = {}
        self.config["pe"]["results"] = self.resultsdir
        self.config["pe"]["numba"] = str(self.numba)
        self.config["pe"]["n_parallel"] = str(self.n_parallel)

        # set fake data
        if self.datafiles is None and self.detector is not None:
            self.config["pe"]["fake-asd-2f"] = str(self.detector)
        elif self.datafiles is not None:
            self.config["pe"]["data-file-2f"] = str(self.datafiles)
        else:
            raise ValueError("No data files of fake detectors are given!")

        if self.starttime is not None:
            self.config["pe"]["fake-start"] = str(int(self.starttime))
        if self.endtime is not None:
            self.config["pe"]["fake-end"] = str(int(self.endtime))
        if self.timestep is not None:
            self.config["pe"]["fake-dt"] = str(int(self.timestep))

        # set the prior files
        priordir = os.path.join(self.basedir, "priors")
        self.makedirs(priordir)
        for pname in self.priors:
            self.priors[pname].to_file(outdir=priordir, label=pname)

        self.config["pe"]["priors"] = priordir
        self.config["pe"]["sampler"] = self.sampler
        if isinstance(self.sampler_kwargs, dict):
            self.config["pe"]["sampler_kwargs"] = str(self.sampler_kwargs)
        self.config["pe"]["output_snr"] = str(self.outputsnr)
