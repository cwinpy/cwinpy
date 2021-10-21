import os
import pathlib

import bilby
from bilby_pipe.input import Input
from bilby_pipe.job_creation.node import Node
from bilby_pipe.utils import check_directory_exists_and_if_not_mkdir
from configargparse import DefaultConfigFileParser


class PEInput(Input):
    def __init__(self, cf):
        """
        Class that sets inputs for the DAG and analysis node generation.

        Parameters
        ----------
        cf: :class:`configparser.ConfigParser`
            The configuration file for the DAG set up.
        """

        self.config = cf

        dagsection = "pe_dag" if cf.has_section("pe_dag") else "dag"

        self.submit = cf.getboolean(dagsection, "submitdag", fallback=False)
        self.transfer_files = cf.getboolean(dagsection, "transfer_files", fallback=True)
        self.osg = cf.getboolean(dagsection, "osg", fallback=False)
        self.label = cf.get(dagsection, "name", fallback="cwinpy_pe")

        # see bilby_pipe MainInput class
        self.scheduler = cf.get(dagsection, "scheduler", fallback="condor")
        self.scheduler_args = cf.get(dagsection, "scheduler_args", fallback=None)
        self.scheduler_module = cf.get(dagsection, "scheduler_module", fallback=None)
        self.scheduler_env = cf.get(dagsection, "scheduler_env", fallback=None)
        self.scheduler_analysis_time = cf.get(
            dagsection, "scheduler_analysis_time", fallback="7-00:00:00"
        )

        self.outdir = cf.get("run", "basedir", fallback=os.getcwd())

        jobsection = "pe_job" if cf.has_section("pe_job") else "job"

        self.universe = cf.get(jobsection, "universe", fallback="vanilla")
        self.getenv = cf.getboolean(jobsection, "getenv", fallback=False)
        self.pe_log_directory = cf.get(
            jobsection,
            "log",
            fallback=os.path.join(os.path.abspath(self._outdir), "log"),
        )
        self.request_memory = cf.get(jobsection, "request_memory", fallback="4 GB")
        self.request_cpus = cf.getint(jobsection, "request_cpus", fallback=1)
        self.accounting = cf.get(
            jobsection, "accounting_group", fallback="cwinpy"
        )  # cwinpy is a dummy tag
        self.accounting_user = cf.get(
            jobsection, "accounting_group_user", fallback=None
        )
        requirements = cf.get(jobsection, "requirements", fallback=None)
        self.requirements = [requirements] if requirements else []
        self.retry = cf.getint(jobsection, "retry", fallback=0)
        self.notification = cf.get(jobsection, "notification", fallback="Never")
        self.email = cf.get(jobsection, "email", fallback=None)
        self.condor_job_priority = cf.getint(
            jobsection, "condor_job_priority", fallback=0
        )

        # number of parallel runs for each job
        self.n_parallel = cf.getint("pe", "n_parallel", fallback=1)
        self.sampler_kwargs = cf.get("pe", "sampler_kwargs", fallback=None)

        # needs to be set for the bilby_pipe Node initialisation, but is not a
        # requirement for cwinpy_pe
        self.online_pe = False
        self.extra_lines = []
        self.run_local = False

    @property
    def submit_directory(self):
        dagsection = "pe_dag" if self.config.has_section("pe_dag") else "dag"
        subdir = self.config.get(
            dagsection, "submit", fallback=os.path.join(self._outdir, "submit")
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


class PulsarPENode(Node):
    # If --osg, run analysis nodes on the OSG
    run_node_on_osg = True

    def __init__(
        self, inputs, configdict, psrname, dets, parallel_idx, dag, generation_node=None
    ):
        super().__init__(inputs)
        self.dag = dag

        self.parallel_idx = parallel_idx
        self.request_cpus = inputs.request_cpus
        self.retry = inputs.retry
        self.getenv = inputs.getenv
        self._universe = inputs.universe

        self.psrname = psrname

        resdir = inputs.config.get("pe", "results", fallback="results")
        self.psrbase = os.path.join(inputs.outdir, resdir, psrname)
        if self.inputs.n_parallel > 1:
            self.resdir = os.path.join(self.psrbase, "par{}".format(parallel_idx))
        else:
            self.resdir = self.psrbase

        check_directory_exists_and_if_not_mkdir(self.resdir)
        configdict["outdir"] = self.resdir

        # job name prefix
        jobname = inputs.config.get("pe_job", "name", fallback="cwinpy_pe")

        # replace any "+" in the pulsar name for the job name as Condor does
        # not allow "+"s in the name
        self.label = "{}_{}_{}".format(jobname, "".join(dets), psrname)
        self.base_job_name = "{}_{}_{}".format(
            jobname, "".join(dets), psrname.replace("+", "plus")
        )
        if inputs.n_parallel > 1:
            self.job_name = "{}_{}".format(self.base_job_name, parallel_idx)
            self.label = "{}_{}".format(self.label, parallel_idx)
        else:
            self.job_name = self.base_job_name

        configdict["label"] = self.label

        # output the configuration file
        configdir = inputs.config.get("pe", "config", fallback="configs")
        configlocation = os.path.join(inputs.outdir, configdir)
        check_directory_exists_and_if_not_mkdir(configlocation)
        if inputs.n_parallel > 1:
            configfile = os.path.join(
                configlocation,
                "{}_{}_{}.ini".format("".join(dets), psrname, parallel_idx),
            )
        else:
            configfile = os.path.join(
                configlocation, "{}_{}.ini".format("".join(dets), psrname)
            )

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

            # make paths relative
            for key in ["par_file", "inj_par", "data_file_1f", "data_file_2f", "prior"]:
                if key in list(configdict.keys()):
                    if key in ["data_file_1f", "data_file_2f"]:
                        for detkey in configdict[key]:
                            input_files_to_transfer.append(
                                self._relative_topdir(
                                    configdict[key][detkey], self.inputs.initialdir
                                )
                            )

                            # exclude full path as the transfer directory is flat
                            configdict[key][detkey] = os.path.basename(
                                configdict[key][detkey]
                            )
                    else:
                        input_files_to_transfer.append(
                            self._relative_topdir(
                                configdict[key], self.inputs.initialdir
                            )
                        )

                        # exclude full path as the transfer directory is flat
                        configdict[key] = os.path.basename(configdict[key])

            configdict["outdir"] = "results/"

            # add output directory to inputs in case resume file exists
            input_files_to_transfer.append(".")

            self.extra_lines.extend(
                self._condor_file_transfer_lines(
                    list(set(input_files_to_transfer)), [configdict["outdir"]]
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

        parseobj = DefaultConfigFileParser()
        with open(configfile, "w") as fp:
            fp.write(parseobj.serialize(configdict))

        self.process_node()

        # reset initial directory
        if tmpinitialdir is not None:
            self.inputs.initialdir = tmpinitialdir

        if generation_node is not None:
            # This is for the future when implementing a full pipeline
            # the generation node will be, for example, a heterodyning job
            if isinstance(generation_node, Node):
                self.job.add_parent(generation_node.job)
            elif isinstance(generation_node, list):
                self.job.add_parents(
                    [gnode.job for gnode in generation_node if isinstance(gnode, Node)]
                )

    @property
    def executable(self):
        jobexec = self.inputs.config.get("pe_job", "executable", fallback="cwinpy_pe")
        return self._get_executable_path(jobexec)

    @property
    def request_memory(self):
        return self.inputs.request_memory

    @property
    def log_directory(self):
        check_directory_exists_and_if_not_mkdir(self.inputs.pe_log_directory)
        return self.inputs.pe_log_directory

    @property
    def result_directory(self):
        """The path to the directory where result output will be stored"""
        check_directory_exists_and_if_not_mkdir(self.resdir)
        return self.resdir

    @property
    def result_file(self):
        extension = self.inputs.sampler_kwargs.get("save", "hdf5")
        gzip = self.inputs.sampler_kwargs.get("gzip", False)
        return bilby.core.result.result_file_name(
            self.result_directory, self.label, extension=extension, gzip=gzip
        )

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


class MergePENode(Node):
    def __init__(self, inputs, parallel_node_list, dag):
        super().__init__(inputs)
        self.dag = dag

        self.job_name = "{}_merge".format(parallel_node_list[0].base_job_name)

        jobname = inputs.config.get("pe_job", "name", fallback="cwinpy_pe")
        self.label = "{}_{}".format(jobname, parallel_node_list[0].psrname)
        self.request_cpus = 1
        self.setup_arguments(
            add_ini=False, add_unknown_args=False, add_command_line_args=False
        )
        self.arguments.append("--result")
        for pn in parallel_node_list:
            self.arguments.append(pn.result_file)
        self.arguments.add("outdir", parallel_node_list[0].psrbase)
        self.arguments.add("label", self.label)
        self.arguments.add_flag("merge")

        extension = self.inputs.sampler_kwargs.get("save", "hdf5")
        gzip = self.inputs.sampler_kwargs.get("gzip", False)
        self.arguments.add("extension", extension)
        if gzip and extension == "json":
            self.arguments.add_flag("gzip")

        self.process_node()
        for pn in parallel_node_list:
            self.job.add_parent(pn.job)

    @property
    def executable(self):
        return self._get_executable_path("bilby_result")

    @property
    def request_memory(self):
        return "16 GB"

    @property
    def log_directory(self):
        return self.inputs.pe_log_directory

    @property
    def result_file(self):
        extension = self.inputs.sampler_kwargs.get("save", "hdf5")
        gzip = self.inputs.sampler_kwargs.get("gzip", False)
        return bilby.core.result.result_file_name(
            self.inputs.result_directory, self.label, extension=extension, gzip=gzip
        )
