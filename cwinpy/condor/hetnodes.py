import copy
import os
import re
import tempfile

from bilby_pipe.job_creation.node import Node
from bilby_pipe.utils import check_directory_exists_and_if_not_mkdir
from configargparse import DefaultConfigFileParser

from ..heterodyne.base import Heterodyne
from ..utils import relative_topdir
from . import CondorLayer

# from htcondor import dags


class HeterodyneLayer(CondorLayer):
    def __init__(self, dag, cf, configurations, **kwargs):
        """
        Class to create submit file for heterodyne jobs.

        Parameters
        ----------
        dag: :class:`hcondor.dags.DAG`
            The HTCondor DAG associated with this layer.
        cf: :class:`configparser.ConfigParser`
            The configuration file for the DAG set up.
        configurations: list
            A list of configuration dictionaries for each node in the layer.
        """

        super().__init__(
            dag,
            cf,
            section_prefix="heterodyne",
            default_executable="cwinpy_heterodyne",
            layer_name=kwargs.pop("layer_name", "cwinpy_heterodyne"),
            **kwargs,
        )

        self.submit = self.get_option("submitdag", default=False)

        # check for use of OSG
        self.osg = self.get_option("osg", default=False)
        self.outdir = self.get_option("basedir", section="run", default=os.getcwd())

        self.log_directory = self.get_option(
            "log", default=os.path.join(os.path.abspath(self.outdir), "log")
        )
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        requirements = self.get_option("requirements", default=None)
        if requirements is not None:
            self.requirements = [requirements]

        # set memory
        self.set_option("request_memory", default="8 GB")
        self.set_option("request_cpus", otype=int, default=1)

        self.set_option(
            "condor_job_priority", optionname="priority", otype=int, default=0
        )
        self.set_option("email", optionname="notify_user")

        if self.osg:
            # make sure files are transferred if using the OSG
            self.submit_options["should_transfer_files"] = "YES"
        else:
            self.set_option(
                "transfer_files", optionname="should_transfer_files", default="YES"
            )

        # check whether GWOSC is required
        self.require_gwosc = kwargs.get("require_gwosc", False)

        # add use_x509userproxy = True to pass on proxy certificate to jobs if
        # needing access to proprietary data
        if not self.require_gwosc:
            self.submit_options["use_x509userproxy"] = True

        # additional options
        additional_options = {}
        macro_options = {}
        if self.submit_options["transfer_files"]:
            macro_options["initialdir"] = "$(INITIALDIR)"
            macro_options["transfer_input_files"] = "$(TRANSFERINPUT)"
            macro_options["transfer_output_files"] = "$(TRANSFEROUTPUT)"

            additional_options["when_to_transfer_output"] = "ON_EXIT_OR_EVICT"
            additional_options["stream_error"] = True
            additional_options["stream_output"] = True

        macro_options["log"] = "$(LOGFILE)"
        macro_options["output"] = "$(OUTPUTFILE)"
        macro_options["error"] = "$(ERRORFILE)"

        additional_options["+SuccessCheckpointExitCode"] = "77"
        additional_options["+WantFTOnCheckpoint"] = True

        if self.osg:
            if self.submit_options.get("accounting_group", "").startswith("ligo."):
                # set to check that proprietary LIGO frames are available
                self.requirements.append("(HAS_LIGO_FRAMES=?=True)")

            # NOTE: the next two statements are currently only require for OSG running,
            # but at the moment not all local pools advertise the CVMFS repo flags
            if self.submit_options["executable"].startswith("/cvmfs"):
                repo = self.submit_options["executable"].split(os.path.sep, 3)[2]
                self.requirements.append(
                    f"(HAS_CVMFS_{re.sub('[.-]', '_', repo)}=?=True)"
                )

            # check if using GWOSC frames from CVMFS
            if self.require_gwosc:
                self.requirements.append("(HAS_CVMFS_gwosc_osgstorage_org =?= TRUE)")

        # generate the node variables
        self.generate_node_vars(configurations)

        # generate layer
        self.generate_layer(
            self.vars,
            parentname=kwargs.get("parentname", None) ** additional_options,
            **macro_options,
        )

    def generate_node_vars(self, configurations):
        """
        Generate the node variables for this layer.
        """

        self.vars = []

        # get location to output individual configuration files to
        configdir = self.get_option("config", section="heterodyne", default="configs")
        configlocation = os.path.join(self.outdir, configdir)
        if not os.path.exists(configlocation):
            os.makedirs(configlocation)

        transfer_files = self.submit_options.get("should_transfer_files", "NO")

        for config in configurations:
            vardict = {}

            # get the results directory
            resultsdir = config["output"]
            if not os.path.exists(resultsdir):
                os.makedirs(resultsdir)

            starttime = config["starttime"]
            endtime = config["endtime"]
            detector = config["detector"]
            freqfactor = config["freqfactor"]
            pulsar = config.get("pulsars", None)

            psrstring = (
                ""
                if not isinstance(pulsar, str)
                else "{}_".format(pulsar.replace("+", "plus"))
            )

            # configuration file
            configfile = os.path.join(
                configlocation,
                "{}{}_{}_{}-{}.ini".format(
                    psrstring, detector, int(freqfactor), starttime, endtime
                ),
            )

            # output the DAG configuration file to a temporary file, which will
            # later be read and stored in the HeterodynedData objects
            _, dagconfigpath = tempfile.mkstemp(
                prefix="pipeline_config", suffix=".ini", text=True
            )
            with open(dagconfigpath, "w") as cfp:
                self.cf.write(cfp)
            config["cwinpy_heterodyne_pipeline_config_file"] = dagconfigpath

            if transfer_files == "YES":
                transfer_input = []
                transfer_output = []

                # add files for transfer
                transfer_input.append(relative_topdir(configfile, resultsdir))

                # create temporary Heterodyne object to get output files
                tmphet = Heterodyne(
                    output=config["output"],
                    label=config.get("label", None),
                    pulsarfiles=copy.deepcopy(config["pulsarfiles"]),
                    pulsars=copy.deepcopy(config["pulsars"]),
                    starttime=starttime,
                    endtime=endtime,
                    detector=detector,
                    freqfactor=freqfactor,
                )

                # if resume is set transfer any created files
                if not config["overwrite"]:
                    for psr in copy.deepcopy(tmphet.outputfiles):
                        psrfile = tmphet.outputfiles[psr]

                        # create empty dummy files, so Condor doesn't complain about files not existing
                        # see https://stackoverflow.com/a/12654798/1862861
                        with open(psrfile, "a"):
                            pass

                        transfer_input.append(relative_topdir(psrfile, resultsdir))

                # remove "output" so result files get written to the cwd
                config.pop("output")

                # set output files to transfer back
                for psr in copy.deepcopy(tmphet.outputfiles):
                    psrfile = tmphet.outputfiles[psr]
                    transfer_output.append(os.path.basename(psrfile))

                # transfer pulsar parameter files
                if isinstance(config["pulsarfiles"], dict):
                    for psr in copy.deepcopy(config["pulsarfiles"]):
                        transfer_input.append(
                            relative_topdir(config["pulsarfiles"][psr], resultsdir)
                        )

                        # set job to only use file (without further path) as the transfer directory is flat
                        config["pulsarfiles"][psr] = os.path.basename(
                            config["pulsarfiles"][psr]
                        )
                else:
                    # pulsarfiles points to a single file
                    transfer_input.append(
                        relative_topdir(config["pulsarfiles"], resultsdir)
                    )

                    config["pulsarfiles"] = os.path.basename(config["pulsarfiles"])

                # transfer ephemeris files
                for ephem in ["earthephemeris", "sunephemeris", "timeephemeris"]:
                    for etype in copy.deepcopy(config[ephem]):
                        transfer_input.append(
                            relative_topdir(config[ephem][etype], resultsdir)
                        )

                        config[ephem][etype] = os.path.basename(config[ephem][etype])

                # transfer frame cache files
                if "framecache" in config:
                    if os.path.isfile(config["framecache"]):
                        transfer_input.append(
                            relative_topdir(config["framecache"], resultsdir)
                        )
                        config["framecache"] = os.path.basename(config["framecache"])

                # transfer segment list files
                if "segmentlist" in config:
                    transfer_input.append(
                        relative_topdir(config["segmentlist"], resultsdir)
                    )
                    config["segmentlist"] = os.path.basename(config["segmentlist"])

                # transfer heterodyned data files
                if "heterodyneddata" in config:
                    for psr in copy.deepcopy(config["heterodyneddata"]):
                        psrfiles = []
                        for psrfile in config["heterodyneddata"][psr]:
                            transfer_input.append(relative_topdir(psrfile, resultsdir))
                            psrfiles.append(os.path.basename(psrfile))

                        config["heterodyneddata"][psr] = psrfiles

                # transfer DAG config file
                transfer_input.append(
                    relative_topdir(
                        config["cwinpy_heterodyne_pipeline_config_file"],
                        resultsdir,
                    )
                )
                config["cwinpy_heterodyne_pipeline_config_file"] = os.path.basename(
                    config["cwinpy_heterodyne_pipeline_config_file"]
                )

                vardict["ARGS"] = f"--config {os.path.basename(configfile)}"
                vardict["INITIALDIR"] = resultsdir
                vardict["TRANSFERINPUT"] = ",".join(transfer_input)
                vardict["TRANSFEROUTPUT"] = ",".join(transfer_output)
            else:
                vardict["ARGS"] = f"--config {configfile}"

            # set log files
            logstr = f"{self.layer_name}_{int(starttime)}-{int(endtime)}"
            vardict["LOGFILE"] = os.path.join(self.log_directory, logstr + ".log")
            vardict["OUTPUTFILE"] = os.path.join(self.log_directory, logstr + ".out")
            vardict["ERRORFILE"] = os.path.join(self.log_directory, logstr + ".err")

            # write out the configuration files for each job
            parseobj = DefaultConfigFileParser()
            with open(configfile, "w") as fp:
                fp.write(parseobj.serialize(config))

            self.vars.append(vardict)


class MergeLayer(CondorLayer):
    def __init__(self, dag, cf, configurations, **kwargs):
        """
        Class to create submit file for heterodyne jobs.

        Parameters
        ----------
        dag: :class:`hcondor.dags.DAG`
            The HTCondor DAG associated with this layer.
        cf: :class:`configparser.ConfigParser`
            The configuration file for the DAG set up.
        configurations: list
            A list of configuration dictionaries for each node in the layer.
        """

        super().__init__(
            dag,
            cf,
            default_executable="cwinpy_heterodyne_merge",
            layer_name="cwinpy_heterodyne_merge",
            **kwargs,
        )

        self.outdir = self.get_option("basedir", section="run", default=os.getcwd())

        self.submit_options["request_memory"] = "2 GB"
        self.submit_options["request_cpus"] = 1
        self.submit_options["universe"] = "local"

        # generate the node variables
        self.generate_node_vars(configurations)

        # generate layer
        self.generate_layer(self.vars, parentname=kwargs.get("parentname", None))

    def generate_node_vars(self, configdict):
        """
        Generate the node variables for this layer.

        Parameters
        ----------
        configdict: dict
            A dictionary of configuration information for
            cwinpy_heterodyne_merge.
        """

        self.vars = []

        vardict = {}

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

        # create merge job configuration file
        configdir = self.get_option("config", section="heterodyne", default="configs")
        configlocation = os.path.join(self.outdir, configdir)
        if not os.path.exists(configlocation):
            os.makedirs(configlocation)
        configfile = os.path.join(
            configlocation,
            "{}{}_{}_merge.ini".format(psrstring, detector, int(freqfactor)),
        )

        # output merge job configuration
        configs = {}
        configs["remove"] = self.get_option(
            "remove", section="merge", otype=bool, default=False
        )
        configs["heterodynedfiles"] = heterodynedfiles
        configs["output"] = output
        configs["overwrite"] = self.get_options(
            "overwrite", section="merge", otype=bool, default=True
        )

        parseobj = DefaultConfigFileParser()
        with open(configfile, "w") as fp:
            fp.write(parseobj.serialize(configs))

        vardict["ARGS"] = f"--config {configfile}"

        self.vars.append(vardict)


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
            add_command_line_args=False, add_ini=False, add_unknown_args=False
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
            "{}{}_{}_merge.ini".format(psrstring, detector, int(freqfactor)),
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
