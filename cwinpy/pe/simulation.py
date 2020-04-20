"""
Generate simulations for running the hierarchical analysis.
"""

import os
import shutil
from configparser import ConfigParser

import astropy.units as u
import bilby
import numpy as np
from astropy.coordinates import SkyCoord

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
    parfiles: dict, str
        If using real pulsar parameter files pass a dictionary of paths to
        individual files keyed to the pulsar name, or pass the path to a
        directory containing the parameter files.
    datafiles: dict, str, :class:`cwinpy.data.MultiHeterodynedData`
        If using real data for each pulsar then pass a dictionary of paths to
        the data files keyed to the pulsar name, or a directory containing
        the heterodyned data files, or a
        :class:`~cwinpy.data.MultiHeterodynedData` object containing the data.
    npulsars: int
        The number of pulsars to include in the simulation.
    qdist: :class:`cwinpy.hierarchical.BaseDistribution`
        The distribution from the the mass quadrupole values will be drawn for
        the simulated signals.
    ddist: :class:`bilby.core.prior.Prior`
        The distribution from which to randomly draw pulsar distances if
        required.
    fdist: :class:`bilby.core.prior.Prior`
        The distribution from which to draw pulsar spin frequencies if
        required.
    skydist: :class:`bilby.core.prior.PriorDict`
        The distribution from which to draw pulsar sky positions if needed.
        If this is required the distribution will default to being uniform over
        the sky.

    prior: dict A bilby-style prior dictionary giving the prior distributions from which to draw the
        injected signal values, and to use for signal recovery. ninj: int The number of simulated
        signals to create. Defaults to 100. maxamp: float A maximum on the amplitude parameter(s) to
        use when drawing the injection parameters. If none is given then this will be taken from the
        prior if using an amplitude parameter. basedir: str The base directory into which the
        simulations and outputs will be placed. If None then the current working directory will be
        used. detector: str, list A string, or list of strings, of detector prefixes for the
        simulated data. This defaults to a single detector - the LIGO Hanford Observatory - from
        which the simulated noise will be drawn from the advanced detector design sensitivity curve
        (e.g., [1]_). submit: bool Set whether to submit the Condor DAG or not. accountuser: str
        Value to give to the 'account_user' parameter in the Condor submit file. Default is to set
        no value. accountgroup: str Value to give to the 'account_user_group' parameter in the
        Condor submit file. Default is to set no value. getenv: bool Set the value for the 'getenv'
        parameter in the Condor submit file. Default is False. sampler: str The sampler to use.
        Defaults to dynesty. sampler_kwargs: dict A dictionary of keyword arguments for the sampler.
        Defaults to None. freqrange: list, tuple A pair of values giving the lower and upper
        rotation frequency ranges (in Hz) for the simulated signals. Defaults to (10, 750) Hz.
        outputsnr: bool Set whether to output the injected and recovered signal-to-noise ratios.
        Defaults to True. numba: bool Set whether or not to use the likelihood with numba enabled.

    References
    ----------

    .. [1] L. Barsotti, S. Gras, M. Evans, P. Fritschel, `LIGO T1800044-v5
       <https://dcc.ligo.org/LIGO-T1800044/public>`_ (2018)

    """

    def __init__(
        self,
        prior,
        ninj=100,
        maxamp=None,
        basedir=None,
        detector="AH1",
        submit=False,
        accountuser=None,
        accountgroup=None,
        getenv=False,
        sampler="dynesty",
        sampler_kwargs=None,
        freqrange=(10.0, 750.0),
        outputsnr=True,
        numba=False,
    ):

        if isinstance(prior, dict):
            self.prior = bilby.core.prior.PriorDict(dictionary=prior)
        else:
            raise TypeError("Prior must be a dictionary-type object")

        if ninj < 1:
            raise ValueError("A positive number of injection must be given")
        self.ninj = int(ninj)

        # set maximum amplitude if given
        self.maxamp = None
        if isinstance(maxamp, float):
            if maxamp > 0.0:
                self.maxamp = maxamp
            else:
                raise ValueError("Maximum amplitude must be positive")

        if basedir is not None:
            self.basedir = basedir
            self.makedirs(basedir)
        else:
            self.basedir = os.getcwd()

        # build output directory structure
        self.detector = detector
        if isinstance(self.detector, str):
            self.detector = [self.detector]
        if not isinstance(self.detector, list):
            raise TypeError("Detector must be a string or list of strings")

        # posterior sample results directory
        self.resultsdir = os.path.join(self.basedir, "results")
        self.makedirs(self.resultsdir)

        # create pulsar parameter files
        self.create_pulsars(freqrange=freqrange)

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

        # add PP plot creation DAG
        self.ppplots()

        if self.submit:
            self.runner.dag.submit_dag()

    def makedirs(self, dir):
        """
        Make a directory tree recursively.
        """

        try:
            os.makedirs(dir, exist_ok=True)
        except Exception as e:
            raise IOError("Could not create directory: {}\n{}".format(dir, e))

    def create_pulsars(self, freqrange):
        """
        Create the pulsar parameter files based on the samples from the priors.

        Parameters
        ----------
        freqrange: list, tuple
            A pair of values giving the lower and upper rotation frequency ranges
            (in Hz) for the simulated signals.
        """

        # pulsar parameter file directory
        self.pulsardir = os.path.join(self.basedir, "pulsars")
        self.makedirs(self.pulsardir)

        # "amplitude" parameters
        amppars = ["h0", "c21", "c22", "q22"]

        if not isinstance(freqrange, (list, tuple, np.ndarray)):
            raise TypeError("Frequency range must be a list or tuple")
        else:
            if len(freqrange) != 2:
                raise ValueError(
                    "Frequency range must contain an upper and lower value"
                )

        self.pulsars = {}
        for i in range(self.ninj):
            pulsar = {}

            for param in self.prior:
                pulsar[param.upper()] = self.prior[param].sample()

            # draw sky position uniformly from the sky if no prior is given
            if "ra" not in self.prior:
                raval = np.random.uniform(0.0, 2.0 * np.pi)
            else:
                raval = pulsar.pop("ra")

            if "dec" not in self.prior:
                decval = -(np.pi / 2.0) + np.arccos(np.random.uniform(-1.0, 1.0))
            else:
                decval = pulsar.pop("dec")

            skypos = SkyCoord(raval * u.rad, decval * u.rad)
            pulsar["RAJ"] = skypos.ra.to_string(u.hour, fields=3, sep=":", pad=True)
            pulsar["DECJ"] = skypos.dec.to_string(u.deg, fields=3, sep=":", pad=True)

            # set maximum amplitude if given
            if self.maxamp is not None:
                for amp in amppars:
                    if amp in self.prior:
                        pulsar[amp.upper()] = bilby.core.prior.Uniform(
                            name=amp, minimum=0.0, maximum=self.maxamp
                        ).sample()

            # set (rotation) frequency upper and lower bounds
            if "f0" not in self.prior:
                pulsar["F0"] = np.random.uniform(freqrange[0], freqrange[1])

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

            # output file name
            pfile = os.path.join(self.pulsardir, "{}.par".format(pname))

            with open(pfile, "w") as fp:
                for param in pulsar:
                    fp.write("{}\t{}\n".format(param, pulsar[param]))

            self.pulsars[pname] = {}
            self.pulsars[pname]["file"] = pfile
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

        self.config["knope"] = {}
        self.config["knope"]["pulsars"] = self.pulsardir
        self.config["knope"]["injections"] = self.pulsardir
        self.config["knope"]["results"] = self.resultsdir
        self.config["knope"]["numba"] = str(self.numba)

        # set fake data
        if "h0" in self.prior or "c22" in self.prior or "q22" in self.prior:
            self.config["knope"]["fake-asd-2f"] = str(self.detector)
        if "c21" in self.prior and "c22" in self.prior:
            self.config["knope"]["fake-asd-1f"] = str(self.detector)
        if "c21" in self.prior and "c22 not in self.prior":
            self.config["knope"]["fake-asd-1f"] = str(self.detector)

        # set the prior file
        label = "ppplot"
        self.priorfile = os.path.join(self.basedir, "{}.prior".format(label))
        self.prior.to_file(outdir=self.basedir, label=label)

        self.config["knope"]["priors"] = self.priorfile
        self.config["knope"]["sampler"] = self.sampler
        if isinstance(self.sampler_kwargs, dict):
            self.config["knope"]["sampler_kwargs"] = str(self.sampler_kwargs)
        self.config["knope"]["output_snr"] = str(self.outputsnr)

    def ppplots(self):
        """
        Set up job to create PP plots.
        """

        from pycondor import Job

        # get executable
        jobexec = shutil.which("cwinpy_knope_generate_pp_plots")

        extra_lines = []
        if self.accountgroup is not None:
            extra_lines.append("accounting_group = {}".format(self.accountgroup))
        if self.accountuser is not None:
            extra_lines.append("accounting_group_user = {}".format(self.accountuser))

        # create cwinpy_knope Job
        job = Job(
            "cwinpy_knope_pp_plots",
            jobexec,
            error=self.runner.error,
            log=self.runner.log,
            output=self.runner.output,
            submit=self.runner.jobsubmit,
            universe=self.runner.universe,
            request_memory=self.runner.reqmem,
            getenv=self.getenv,
            queue=1,
            requirements=self.runner.requirements,
            retry=self.runner.retry,
            extra_lines=extra_lines,
            dag=self.runner.dag,
        )

        jobargs = "--path '{}' ".format(os.path.join(self.basedir, "results", "*", "*"))
        jobargs += "--output {} ".format(os.path.join(self.basedir, "ppplot.png"))
        if self.outputsnr:
            jobargs += "--snrs "
        job.add_arg(jobargs)

        job.add_parents(
            self.runner.dag.nodes[:-1]
        )  # exclude cwinpy_knope_pp_plots job itself
        self.runner.dag.build()
