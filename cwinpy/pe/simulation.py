"""
Generate simulations for running the hierarchical analysis.
"""

import os
from configparser import ConfigParser

import astropy.units as u
import bilby
import numpy as np
from astropy.coordinates import SkyCoord
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy

from ..hierarchical import BaseDistribution
from ..utils import is_par_file
from .pe import pe_dag


class PEMassQuadrupoleSimulationDAG(object):
    """
    This class will generate a HTCondor Dagman job to create a number of
    simulated gravitational-wave signals from pulsars. These signals will
    either be added into random Gaussian noise, or real heterodyned data.
    The generated signals will use mass quadrupole values drawn from an
    input distribution. They can be generated using real pulsars parameters,
    including the pulsar distance, or using fake pulsars with sky
    locations and distances drawn from input distribution (by default the sky
    location distribution will be uniform on the sky).

    These signals will be analysed using the ``cwinpy_pe`` script to sample
    the posterior probability distributions of the required parameter space,
    including the mass quadrupole.

    Parameters
    ----------
    q22dist: :class:`~cwinpy.hierarchical.BaseDistribution`, :class:`~bilby.core.prior.Prior`
        The distribution from the the mass quadrupole values will be drawn for
        the simulated signals.
    prior: dict
        A bilby-style prior dictionary giving the prior distributions from
        which to use for signal recovery.
    parfiles: dict, str
        If using real pulsar parameter files pass a dictionary of paths to
        individual files keyed to the pulsar name, or pass the path to a
        directory containing the parameter files.
    datafiles: dict, str, :class:`cwinpy.data.MultiHeterodynedData`
        If using real data for each pulsar then pass a dictionary of paths to
        the data files keyed to the pulsar name (or detector then pulsar name),
        or a directory containing the heterodyned data files, or a
        :class:`~cwinpy.data.MultiHeterodynedData` object containing the data.
    npulsars: int
        The number of pulsars to include in the simulation.
    posdist: :class:`bilby.core.prior.Prior`
        The distribution from which to randomly draw pulsar positions (right
        ascension, declination and distance) if required. This defaults to a
        uniform distribution on the sky and uniform in distance between 0.1
        and 10 kpc.
    oridist: :class:`bilby.core.prior.Prior`
        The distribution if the pulsar orientation parameters. This defaults
        to uniform over a hemisphere in inclination and polarisation angle,
        and uniform over pi radians in rotational phase.
    fdist: :class:`bilby.core.prior.Prior`
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
        advanced detector design sensitivity curve (e.g., [1]_).
    submit: bool
        Set whether to submit the Condor DAG or not.
    accountuser: str
        Value to give to the 'account_user' parameter in the Condor submit
        file. Default is to set no value.
    accountgroup: str
        Value to give to the 'account_user_group' parameter in the Condor
        submit file. Default is to set no value.
    getenv: bool
        Set the value for the 'getenv' parameter in the Condor submit file.
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

    References
    ----------

    .. [1] L. Barsotti, S. Gras, M. Evans, P. Fritschel, `LIGO T1800044-v5
       <https://dcc.ligo.org/LIGO-T1800044/public>`_ (2018)

    """

    def __init__(
        self,
        prior,
        q22dist,
        parfiles=None,
        datafiles=None,
        npulsars=None,
        basedir=None,
        detector="H1",
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
    ):
        if isinstance(q22dist, (BaseDistribution, bilby.core.prior.Prior)):
            self.q22dist = q22dist
        else:
            raise TypeError(
                "Q22 distribution must be a child of a BaseDistribution or bilby Prior"
            )

        if isinstance(prior, (dict, bilby.core.prior.PriorDict)):
            if "q22" in prior:
                self.prior = bilby.core.prior.PriorDict(prior)
            else:
                raise ValueError("Prior must contain 'q22'")
        else:
            raise TypeError("Prior must be a dictionary-type object")

        self.parfiles = parfiles

        if basedir is not None:
            self.basedir = basedir
            self.makedirs(basedir)
        else:
            self.basedir = os.getcwd()

        # create pulsar parameter files if none are given
        if self.parfiles is None:
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
        self.create_config()

        # create the DAG for cwinpy_knope jobs
        self.runner = pe_dag(config=self.config, build=False)

        if self.submit:
            self.runner.dag.submit_dag()

    @property
    def parfiles(self):
        return self._parfiles

    @parfiles.setter
    def parfiles(self, parfiles):
        pfs = {}

        if parfiles is not None:
            if isinstance(parfiles, dict):
                self._parfiles = parfiles

                for psr in self._parfiles:
                    if not is_par_file(self._parfiles[psr]):
                        raise IOError(
                            "{} is not a pulsar parameter file".format(
                                self._parfiles[psr]
                            )
                        )
            elif os.path.isdir(parfiles):
                for pf in os.listdir(parfiles):
                    parfile = os.path.join(parfiles, pf)
                    if is_par_file(parfile):
                        psr = PulsarParametersPy(parfile)

                        # add parfile to dictionary
                        for name in ["PSRJ", "PSRB", "PSR", "NAME"]:
                            if psr[name] is not None:
                                pfs[psr[name]] = parfile
            else:
                raise TypeError(
                    "Must give directory of dictionary of pulsar " "parameter files"
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
            # set default position distribution
            ddist = bilby.core.prior.Uniform(
                (0.1 * u.kpc).to("m"), (10.0 * u.kpc).to("m"), name="distance"
            )
            radist = bilby.core.prior.Uniform(0.0, 2.0 * np.pi, name="ra")
            decdist = bilby.core.prior.Sine(name="dec")
            self._posdist = bilby.core.prior.PriorDict(
                {"distance": ddist, "ra": radist, "dec": decdist}
            )
        elif isinstance(posdist, bilby.core.prior.PriorDict):
            # check that distribution contains distance, ra and dec
            if self.parfiles is None:
                keys = ["ra", "dec", "distance"]
            else:
                keys = ["distance"]
            for key in keys:
                if key not in posdist:
                    raise ValueError(
                        "Position distribution must contain {}".format(keys)
                    )
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
            iota = bilby.core.prior.Sine(0.0, np.pi, name="iota")

            self._oridist = bilby.core.prior.PriorDict(
                {"phi0": phase, "psi": psi, "iota": iota}
            )
        elif isinstance(oridist, bilby.core.prior.PriorDict):
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
                self._oridist["iota"] = bilby.core.prior.Sine(0.0, np.pi, name="iota")
        else:
            raise TypeError("Orientation distribution is not correct type")

    def makedirs(self, dir):
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
        for i in range(self.npulsars):
            # generate fake pulsar parameters
            if self.posdist is not None:
                skyloc = self.posdist.sample()

            if self.fdist is not None:
                freq = self.fdist.sample()

            # generate orientation parameters
            orientation = self.oridist.sample()

            if self.parfiles is None:
                pulsar = {}
                skypos = SkyCoord(skyloc["ra"] * u.rad, skyloc["dec"] * u.rad)
                pulsar["RAJ"] = skypos.ra.to_string(u.hour, fields=3, sep=":", pad=True)
                pulsar["DECJ"] = skypos.dec.to_string(
                    u.deg, fields=3, sep=":", pad=True
                )
                pulsar["DIST"] = skyloc["distance"]

                # set pulsar name from sky position
                rastr = skypos.ra.to_string(u.hour, fields=2, sep="", pad=True)
                decstr = skypos.dec.to_string(
                    u.deg, fields=2, sep="", pad=True, alwayssign=True
                )
                pname = "J{}{}".format(rastr, decstr)
                pnameorig = str(pname)  # copy of original name
                counter = 0
                alphas = ["A", "B", "C", "D", "E", "F", "G"]
                while pname in self.pulsars:
                    if counter == len(alphas):
                        raise RuntimeError("Too many pulsars in the same sky position!")
                    pname = pnameorig + alphas[counter]
                    counter += 1

                pulsar["PSRJ"] = pname
                pulsar["F"] = [freq]

                for param in ["psi", "iota", "phi0"]:
                    pulsar[param.upper()] = orientation[param]

                # output file name
                pfile = os.path.join(self.pulsardir, "{}.par".format(pname))
                injfile = pfile
            else:
                pfile = list(self.parfiles.values())[i]
                pulsar = PulsarParametersPy(pfile)
                pname = list(self.parfiles.keys())[i]
                injfile = os.path.join(self.pulsar, "{}.par".format(pname))

                if pulsar["DIST"] is None:
                    # add distance if not present in parameter file
                    pulsar["DIST"] = skyloc["distance"]

                for param in ["psi", "iota", "phi0"]:
                    # add orientation values if not present in parameter file
                    if pulsar[param.upper()] is None:
                        pulsar[param.upper()] = orientation[param]

            # set Q22 value
            pulsar["Q22"] = self.q22dist.sample()

            with open(injfile, "w") as fp:
                if self.parfiles is None:
                    for param in pulsar:
                        fp.write("{}\t{}\n".format(param, pulsar[param]))
                else:
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

        self.config["dag"] = {"build": False}

        self.config["job"] = {}
        self.config["job"]["getenv"] = str(self.getenv)

        if self.accountgroup is not None:
            self.config["job"]["accounting_group"] = self.accountgroup
        if self.accountuser is not None:
            self.config["job"]["accounting_group_user"] = self.accountuser

        self.config["pe"] = {}
        self.config["pe"]["pulsars"] = str(
            [self.pulsars[pname]["file"] for pname in self.pulsars]
        )
        self.config["pe"]["injections"] = str(
            [self.pulsars[pname]["injection_file"] for pname in self.pulsars]
        )
        self.config["pe"]["results"] = self.resultsdir
        self.config["pe"]["numba"] = str(self.numba)

        # set fake data
        if self.datafiles is None and self.detector is not None:
            self.config["pe"]["fake-asd-2f"] = str(self.detector)
        elif self.datafiles is not None:
            self.config["pe"]["data-file-2f"] = str(self.datafiles)
        else:
            raise ValueError("No data files of fake detectors are given!")

        # set the prior file
        label = "simulation"
        self.priorfile = os.path.join(self.basedir, "{}.prior".format(label))
        self.prior.to_file(outdir=self.basedir, label=label)

        self.config["pe"]["priors"] = self.priorfile
        self.config["pe"]["sampler"] = self.sampler
        if isinstance(self.sampler_kwargs, dict):
            self.config["pe"]["sampler_kwargs"] = str(self.sampler_kwargs)
        self.config["pe"]["output_snr"] = str(self.outputsnr)
