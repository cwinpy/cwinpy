import copy
import os
import pathlib
import tempfile

import pycondor
from bilby_pipe.input import Input
from bilby_pipe.job_creation.node import Node, _log_output_error_submit_lines
from bilby_pipe.utils import check_directory_exists_and_if_not_mkdir, logger
from configargparse import DefaultConfigFileParser

from .base import Heterodyne


class HeterodyneInput(Input):
    def __init__(self, cf):
        """
        Class that sets inputs for the DAG and analysis node generation.

        Parameters
        ----------
        cf: :class:`configparser.ConfigParser`
            The configuration file for the DAG set up.
        """

        self.config = cf
        self.submit = cf.getboolean("dag", "submitdag", fallback=False)
        self.transfer_files = cf.getboolean("dag", "transfer_files", fallback=True)
        self.osg = cf.getboolean("dag", "osg", fallback=False)
        self.require_gwosc = False
        self.label = cf.get("dag", "name", fallback="cwinpy_heterodyne")

        # see bilby_pipe MainInput class
        self.scheduler = cf.get("dag", "scheduler", fallback="condor")
        self.scheduler_args = cf.get("dag", "scheduler_args", fallback=None)
        self.scheduler_module = cf.get("dag", "scheduler_module", fallback=None)
        self.scheduler_env = cf.get("dag", "scheduler_env", fallback=None)
        self.scheduler_analysis_time = cf.get(
            "dag", "scheduler_analysis_time", fallback="7-00:00:00"
        )

        self.outdir = cf.get("run", "basedir", fallback=os.getcwd())

        self.universe = cf.get("job", "universe", fallback="vanilla")
        self.getenv = cf.getboolean("job", "getenv", fallback=True)
        self.heterodyne_log_directory = cf.get(
            "job", "log", fallback=os.path.join(os.path.abspath(self._outdir), "log")
        )
        self.request_memory = cf.get("job", "request_memory", fallback="4 GB")
        self.request_cpus = cf.getint("job", "request_cpus", fallback=1)
        self.accounting = cf.get(
            "job", "accounting_group", fallback="cwinpy"
        )  # cwinpy is a dummy tag
        self.accounting_user = cf.get("job", "accounting_group_user", fallback=None)
        requirements = cf.get("job", "requirements", fallback=None)
        self.requirements = [requirements] if requirements else []
        self.retry = cf.getint("job", "retry", fallback=0)
        self.notification = cf.get("job", "notification", fallback="Never")
        self.email = cf.get("job", "email", fallback=None)
        self.condor_job_priority = cf.getint("job", "condor_job_priority", fallback=0)

    @property
    def submit_directory(self):
        subdir = self.config.get(
            "dag", "submit", fallback=os.path.join(self._outdir, "submit")
        )
        check_directory_exists_and_if_not_mkdir(subdir)
        return subdir

    @property
    def initialdir(self):
        if hasattr(self, "_initialdir"):
            if self._initialdir is not None:
                return self._initialdir
            else:
                return os.getcwd()
        else:
            return os.getcwd()

    @initialdir.setter
    def initialdir(self, initialdir):
        if isinstance(initialdir, str):
            self._initialdir = initialdir
        else:
            self._initialdir = None


class HeterodyneNode(Node):
    """
    Create a HTCondor DAG node for running the cwinpy_heterodyne script.
    """

    # If --osg, run analysis nodes on the OSG
    run_node_on_osg = True

    def __init__(self, inputs, configdict, dag, generation_node=None):
        self.inputs = inputs
        self.request_disk = None
        self.notification = inputs.notification
        self.verbose = 0
        self.condor_job_priority = inputs.condor_job_priority
        self.extra_lines = []
        self.requirements = (
            [self.inputs.requirements] if self.inputs.requirements else []
        )

        self.dag = dag

        self.request_cpus = inputs.request_cpus
        self.retry = inputs.retry
        self.getenv = inputs.getenv
        self._universe = inputs.universe

        starttime = configdict["starttime"]
        endtime = configdict["endtime"]
        detector = configdict["detector"]
        freqfactor = configdict["freqfactor"]
        pulsar = configdict.get("pulsars", None)

        psrstring = (
            ""
            if not isinstance(pulsar, str)
            else "{}_".format(pulsar.replace("+", "plus"))
        )

        self.resdir = configdict["output"]
        check_directory_exists_and_if_not_mkdir(self.resdir)

        # job name prefix
        jobname = inputs.config.get("job", "name", fallback="cwinpy_heterodyne")
        self.base_job_name = "{}_{}{}_{}_{}-{}".format(
            jobname, psrstring, detector, int(freqfactor), starttime, endtime
        )
        self.job_name = self.base_job_name

        # output the configuration file
        configdir = inputs.config.get("heterodyne", "config", fallback="configs")
        configlocation = os.path.join(inputs.outdir, configdir)
        check_directory_exists_and_if_not_mkdir(configlocation)
        configfile = os.path.join(
            configlocation,
            "{}{}_{}_{}-{}.ini".format(
                psrstring, detector, int(freqfactor), starttime, endtime
            ),
        )

        # output the DAG configuration file to a temporary file, which will
        # later be read and stored in the HeterodynedData objects
        _, dagconfigpath = tempfile.mkstemp(
            prefix="dag_config", suffix=".ini", text=True
        )
        with open(dagconfigpath, "w") as cfp:
            inputs.config.write(cfp)
        configdict["cwinpy_heterodyne_dag_config_file"] = dagconfigpath

        self.setup_arguments(
            add_ini=False, add_unknown_args=False, add_command_line_args=False
        )

        # add files for transfer
        if self.inputs.transfer_files or self.inputs.osg:
            tmpinitialdir = self.inputs.initialdir

            self.inputs.initialdir = self.resdir

            input_files_to_transfer = [
                self._relative_topdir(configfile, self.inputs.initialdir)
            ]

            output_files_to_transfer = []

            # create temporary Heterodyne object to get output files
            tmphet = Heterodyne(
                output=configdict["output"],
                label=configdict.get("label", None),
                pulsarfiles=copy.deepcopy(configdict["pulsarfiles"]),
                pulsars=copy.deepcopy(configdict["pulsars"]),
            )

            # if resume is set transfer any created files
            if not configdict["overwrite"]:
                labeldict = {
                    "det": detector,
                    "gpsstart": int(starttime),
                    "gpsend": int(endtime),
                    "freqfactor": int(freqfactor),
                }
                for psr in tmphet.outputfiles.copy():
                    labeldict["psr"] = psr
                    psrfile = tmphet.outputfiles[psr].format(**labeldict)

                    # create empty dummy files, so Condor doesn't complain about files not existing
                    # see https://stackoverflow.com/a/12654798/1862861
                    with open(psrfile, "a"):
                        pass

                    input_files_to_transfer.append(
                        self._relative_topdir(psrfile, self.inputs.initialdir)
                    )

            # remove "output" so result files get written to the cwd
            configdict.pop("output")

            # set output files to transfer back
            for psr in tmphet.outputfiles.copy():
                labeldict["psr"] = psr
                psrfile = tmphet.outputfiles[psr].format(**labeldict)
                output_files_to_transfer.append(os.path.basename(psrfile))

            # transfer pulsar parameter files
            for psr in configdict["pulsarfiles"].copy():
                input_files_to_transfer.append(
                    self._relative_topdir(
                        configdict["pulsarfiles"][psr], self.inputs.initialdir
                    )
                )

                # set job to only use file (without further path) as the transfer directory is flat
                configdict["pulsarfiles"][psr] = os.path.basename(
                    configdict["pulsarfiles"][psr]
                )

            # transfer ephemeris files
            for ephem in ["earthephemeris", "sunephemeris", "timeephemeris"]:
                for etype in configdict[ephem].copy():
                    input_files_to_transfer.append(
                        self._relative_topdir(
                            configdict[ephem][etype], self.inputs.initialdir
                        ),
                    )

                    configdict[ephem][etype] = os.path.basename(
                        configdict[ephem][etype]
                    )

            # transfer frame cache files
            if "framecache" in configdict:
                if os.path.isfile(configdict["framecache"]):
                    input_files_to_transfer.append(
                        self._relative_topdir(
                            configdict["framecache"], self.inputs.initialdir
                        )
                    )
                    configdict["framecache"] = os.path.basename(
                        configdict["framecache"]
                    )

            # transfer segment list files
            if "segmentlist" in configdict:
                input_files_to_transfer.append(
                    self._relative_topdir(
                        configdict["segmentlist"], self.inputs.initialdir
                    )
                )
                configdict["segmentlist"] = os.path.basename(configdict["segmentlist"])

            # transfer heterodyned data files
            if "heterodyneddata" in configdict:
                for psr in configdict["heteroyneddata"].copy():
                    psrfiles = []
                    for psrfile in configdict["heteroyneddata"][psr]:
                        input_files_to_transfer.append(
                            self._relative_topdir(psrfile, self.inputs.initialdir)
                        )
                        psrfiles.append(os.path.basename(psrfile))

                    configdict["heteroyneddata"][psr] = psrfiles

            # transfer DAG config file
            input_files_to_transfer.append(
                self._relative_topdir(
                    configdict["cwinpy_heterodyne_dag_config_file"],
                    self.inputs.initialdir,
                )
            )
            configdict["cwinpy_heterodyne_dag_config_file"] = os.path.basename(
                configdict["cwinpy_heterodyne_dag_config_file"]
            )

            self.extra_lines.extend(
                self._condor_file_transfer_lines(
                    list(set(input_files_to_transfer)), output_files_to_transfer
                )
            )

            self.arguments.add("config", os.path.basename(configfile))
        else:
            tmpinitialdir = None
            self.arguments.add("config", configfile)

        self.extra_lines.extend(self._checkpoint_submit_lines())

        # add accounting user
        if self.inputs.accounting_user is not None:
            self.extra_lines.append(
                "accounting_group_user = {}".format(self.inputs.accounting_user)
            )

        # add use_x509userproxy = True to pass on proxy certificate to jobs if
        # needing access to proprietory data
        if not self.inputs.require_gwosc:
            self.extra_lines.append("use_x509userproxy = True")

        parseobj = DefaultConfigFileParser()
        with open(configfile, "w") as fp:
            fp.write(parseobj.serialize(configdict))

        self.create_pycondor_job()

        # reset initial directory
        if tmpinitialdir is not None:
            self.inputs.initialdir = tmpinitialdir

        if generation_node is not None:
            # for fine heterodyne, add previous jobs as parent
            if isinstance(generation_node, Node):
                self.job.add_parent(generation_node.job)
            elif isinstance(generation_node, list):
                self.job.add_parents(
                    [gnode.job for gnode in generation_node if isinstance(gnode, Node)]
                )

    def create_pycondor_job(self):
        """
        Overwritten version of create_pycondor_job from the bilby_pipe Node
        class to allow for proprietory LIGO frames to be used over the OSG.
        """

        job_name = self.job_name
        self.extra_lines.extend(
            _log_output_error_submit_lines(self.log_directory, job_name)
        )

        if self.inputs.scheduler.lower() == "condor":
            self.add_accounting()

        self.extra_lines.append(f"priority = {self.condor_job_priority}")
        if self.inputs.email is not None:
            self.extra_lines.append(f"notify_user = {self.inputs.email}")

        if self.universe != "local" and self.inputs.osg:
            if self.run_node_on_osg:
                # if using LIGO accounting tag use proprietory frames on CVMFS
                has_ligo_frames = False
                if not self.inputs.require_gwosc and self.inputs.accounting:
                    if self.inputs.accounting[0:5] == "ligo.":
                        has_ligo_frames = True

                _osg_lines, _osg_reqs = self._osg_submit_options(
                    self.executable, has_ligo_frames=has_ligo_frames
                )

                # check if using GWOSC frames from CVMFS
                if self.inputs.require_gwosc:
                    _osg_reqs.append("HAS_CVMFS_gwosc_osgstorage_org =?= TRUE")

                self.extra_lines.extend(_osg_lines)
                self.requirements.append(_osg_reqs)
            else:
                osg_local_node_lines = [
                    "+flock_local = True",
                    '+DESIRED_Sites = "nogrid"',
                    "+should_transfer_files = NO",
                ]
                self.extra_lines.extend(osg_local_node_lines)

        self.job = pycondor.Job(
            name=job_name,
            executable=self.executable,
            submit=self.inputs.submit_directory,
            request_memory=self.request_memory,
            request_disk=self.request_disk,
            request_cpus=self.request_cpus,
            getenv=self.getenv,
            universe=self.universe,
            initialdir=self.inputs.initialdir,
            notification=self.notification,
            requirements=" && ".join(self.requirements),
            extra_lines=self.extra_lines,
            dag=self.dag.pycondor_dag,
            arguments=self.arguments.print(),
            retry=self.retry,
            verbose=self.verbose,
        )

        # Hack to allow passing walltime down to slurm
        setattr(self.job, "slurm_walltime", self.slurm_walltime)

        logger.debug(f"Adding job: {job_name}")

    @property
    def executable(self):
        jobexec = self.inputs.config.get(
            "job", "executable", fallback="cwinpy_heterodyne"
        )
        return self._get_executable_path(jobexec)

    @property
    def request_memory(self):
        return self.inputs.request_memory

    @property
    def log_directory(self):
        check_directory_exists_and_if_not_mkdir(self.inputs.heterodyne_log_directory)
        return self.inputs.heterodyne_log_directory

    @property
    def result_directory(self):
        """
        The path to the directory where result output will be stored.
        """
        check_directory_exists_and_if_not_mkdir(self.resdir)
        return self.resdir

    @staticmethod
    def _relative_topdir(path, reference):
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


class MergeHeterodyneNode(Node):
    """
    Create a HTCondor DAG node for running the cwinpy_heterodyne_merge script.
    """

    def __init__(self, inputs, configdict, dag, generation_node=None):
        self.inputs = inputs
        self.request_disk = None
        self.notification = inputs.notification
        self.verbose = 0
        self.condor_job_priority = inputs.condor_job_priority
        self.extra_lines = []
        self.requirements = (
            [self.inputs.requirements] if self.inputs.requirements else []
        )

        self.dag = dag

        self.retry = inputs.retry
        self.getenv = inputs.getenv
        self.request_cpus = 1

        # run merge jobs locally
        self._universe = "local"

        self.setup_arguments(
            add_command_line_args=False,
            add_ini=False,
            add_unknown_args=False,
        )

        detector = configdict["detector"]
        freqfactor = configdict["freqfactor"]
        pulsar = configdict["pulsar"]
        output = configdict["output"]
        heterodynedfiles = configdict["heterodynedfiles"]

        psrstring = (
            ""
            if not isinstance(pulsar, str)
            else "{}_".format(pulsar.replace("+", "plus"))
        )

        configdir = self.inputs.config.get("heterodyne", "config", fallback="configs")
        configlocation = os.path.join(self.inputs.outdir, configdir)
        check_directory_exists_and_if_not_mkdir(configlocation)
        configfile = os.path.join(
            configlocation,
            "{}{}_{}_merge.ini".format(
                psrstring,
                detector,
                int(freqfactor),
            ),
        )

        self.arguments.add("config", configfile)

        # add accounting user
        if self.inputs.accounting_user is not None:
            self.extra_lines.append(
                "accounting_group_user = {}".format(self.inputs.accounting_user)
            )

        # output merge job configuration
        configs = {}
        configs["remove"] = self.inputs.config.getboolean(
            "merge", "remove", fallback=False
        )
        configs["heterodynedfiles"] = heterodynedfiles
        configs["output"] = output
        configs["overwrite"] = self.inputs.config.getboolean(
            "merge", "overwrite", fallback=True
        )

        parseobj = DefaultConfigFileParser()
        with open(configfile, "w") as fp:
            fp.write(parseobj.serialize(configs))

        # job name prefix
        jobname = "cwinpy_heterodyne_merge"
        self.base_job_name = "{}_{}{}_{}".format(
            jobname, psrstring, detector, int(freqfactor)
        )
        self.job_name = self.base_job_name

        self.online_pe = False  # required for create_pycondor_job()
        self.create_pycondor_job()

        if generation_node is not None:
            self.job.add_parents(
                [gnode.job for gnode in generation_node if isinstance(gnode, Node)]
            )

    @property
    def executable(self):
        jobexec = "cwinpy_heterodyne_merge"
        return self._get_executable_path(jobexec)

    @property
    def request_memory(self):
        return self.inputs.request_memory

    @property
    def log_directory(self):
        check_directory_exists_and_if_not_mkdir(self.inputs.heterodyne_log_directory)
        return self.inputs.heterodyne_log_directory
