import ast
import copy
import os
import re

import bilby
from configargparse import DefaultConfigFileParser

from ..utils import relative_topdir
from . import CondorLayer


class PulsarPELayer(CondorLayer):
    def __init__(self, dag, cf, config, **kwargs):
        """
        Class to create submit file for parameter estimation jobs.

        Parameters
        ----------
        dag: :class:`hcondor.dags.DAG`
            The HTCondor DAG associated with this layer.
        cf: :class:`configparser.ConfigParser`
            The configuration file for the DAG set up.
        config: dict
            A configuration dictionary for each pulsar node in the layer.
        """

        self.psrname = kwargs.pop("psrname")
        self.dets = kwargs.pop("dets")

        super().__init__(
            dag,
            cf,
            section_prefix="pe",
            default_executable="cwinpy_pe",
            layer_name=kwargs.pop("layer_name", "cwinpy_pe"),
            **kwargs,
        )

        # check for use of OSG
        self.osg = self.get_option("osg", default=False)
        self.outdir = self.get_option("basedir", section="run", default=os.getcwd())

        # check for use of tempo2
        self.usetempo2 = self.get_option("usetempo2", default=False)

        # check number of parallel runs
        self.n_parallel = self.get_option("n_parallel", otype=int, default=1)

        # store sampler kwargs
        self.sampler_kwargs = ast.literal_eval(
            self.get_option("sampler_kwargs", default="{}")
        )

        self.log_directories = {}
        for logtype in ["log", "error", "out"]:
            self.log_directories[logtype] = self.get_option(
                logtype, default=os.path.join(os.path.abspath(self.outdir), "log")
            )
            if not os.path.exists(self.log_directories[logtype]):
                os.makedirs(self.log_directories[logtype])

        requirements = self.get_option("requirements", default=None)
        if requirements is not None:
            self.requirements = [requirements]

        # set memory
        self.set_option("request_memory", default="8 GB")
        self.set_option("request_cpus", otype=int, default=1)
        self.set_option("request_disk", default="1 GB")

        self.set_option(
            "condor_job_priority", optionname="priority", otype=int, default=0
        )
        self.set_option("email", optionname="notify_user")

        if self.usetempo2:
            tempo2 = os.environ.get("TEMPO2", None)

            if tempo2 is None:
                raise ValueError("No TEMPO2 environment variable exists")

            # add TEMPO2 environment variable to the submit file
            self.submit_options["environment"] = f'"{tempo2}"'

        additional_options = {}
        if self.osg:
            # make sure files are transferred if using the OSG
            self.submit_options["should_transfer_files"] = "YES"

            # allow use of local pool (https://computing.docs.ligo.org/guide/htcondor/access/#local-access-points)
            additional_options["MY.flock_local"] = "True"

            if self.submit_options.get("desired_sites", ""):
                # allow specific OSG sites to be requested
                additional_options["MY.DESIRED_Sites"] = self.submit_options[
                    "desired_sites"
                ]
                self.requirements.append("IS_GLIDEIN=?=True")

            if self.submit_options.get("undesired_sites", ""):
                # disallow certain OSG sites to be used
                additional_options["MY.UNDESIRED_Sites"] = self.submit_options[
                    "undesired_sites"
                ]

            # use development CWInPy singularity container
            singularity = self.get_option("singularity", default=False)
            if singularity:
                self.submit_options[
                    "executable"
                ] = "/opt/conda/envs/python38/bin/cwinpy_pe"
                additional_options["MY.SingularityImage"] = (
                    '"/cvmfs/singularity.opensciencegrid.org/matthew-pitkin/'
                    'cwinpy-containers/cwinpy-dev-python38:latest"'
                )
                self.requirements.append("(HAS_SINGULARITY =?= True)")
                self.submit_options["transfer_executable"] = False

            if (
                self.submit_options["executable"].startswith("/cvmfs")
                and "igwn" in self.submit_options["executable"]
            ) or "MY.SingularityImage" in additional_options:
                if "MY.SingularityImage" not in additional_options:
                    repo = self.submit_options["executable"].split(os.path.sep, 3)[2]
                    self.requirements.append(
                        f"(HAS_CVMFS_{re.sub('[.-]', '_', repo)} =?= True)"
                    )
            else:
                raise RuntimeError(
                    "If running on the OSG you must be using an IGWN "
                    "environment or the CWInPy developement singularity "
                    "container."
                )
        else:
            self.set_option(
                "transfer_files", optionname="should_transfer_files", default="YES"
            )

        # additional options
        if self.submit_options["should_transfer_files"] == "YES":
            additional_options["initialdir"] = "$(INITIALDIR)"
            additional_options["transfer_input_files"] = "$(TRANSFERINPUT)"
            additional_options["transfer_output_files"] = "$(TRANSFEROUTPUT)"

            additional_options["when_to_transfer_output"] = "ON_EXIT_OR_EVICT"
            additional_options["stream_error"] = True
            additional_options["stream_output"] = True

        additional_options["MY.SuccessCheckpointExitCode"] = "77"
        additional_options["MY.WantFTOnCheckpoint"] = True

        additional_options["log"] = "$(LOGFILE)"
        additional_options["output"] = "$(OUTPUTFILE)"
        additional_options["error"] = "$(ERRORFILE)"

        # generate the node variables
        self.generate_node_vars(config)

        # generate layer
        self.generate_layer(
            self.vars,
            parentname=kwargs.get("parentname", None),
            submitoptions=additional_options,
        )

    def generate_node_vars(self, config):
        """
        Generate the node variables for this layer.
        """

        self.vars = []

        # get location to output individual configuration files to
        configdir = self.get_option("config", section="pe", default="configs")
        configlocation = os.path.join(self.outdir, configdir)
        if not os.path.exists(configlocation):
            os.makedirs(configlocation)

        dagconfigfile = os.path.join(configlocation, "pe_pipeline_config.ini")

        # get results directory
        self.resbase = os.path.join(
            self.outdir, self.get_option("results", default="results")
        )
        self.resdir = os.path.join(self.resbase, self.psrname)
        if not os.path.exists(self.resdir):
            os.makedirs(self.resdir)

        transfer_files = self.submit_options.get("should_transfer_files", "NO")

        # store expected results file names
        extension = self.sampler_kwargs.get("save", "hdf5")
        gzip = self.sampler_kwargs.get("gzip", False)
        self.resultsfiles = []

        for i in range(self.n_parallel):
            curconfig = copy.deepcopy(config)
            vardict = {}

            label = f"{self.submit_options.get('name', 'cwinpy_pe')}_{''.join(self.dets)}_{self.psrname}"

            if self.n_parallel > 1:
                configfile = os.path.join(
                    configlocation,
                    "{}_{}_{}.ini".format("".join(self.dets), self.psrname, i),
                )

                label += f"_{i}"
            else:
                configfile = os.path.join(
                    configlocation, "{}_{}.ini".format("".join(self.dets), self.psrname)
                )

            self.resultsfiles.append(
                bilby.core.result.result_file_name(
                    os.path.abspath(self.resdir), label, extension=extension, gzip=gzip
                )
            )

            curconfig["label"] = label

            # add files for transfer
            if transfer_files == "YES":
                transfer_input = []

                transfer_input.append(relative_topdir(configfile, self.resdir))

                for key in [
                    "par_file",
                    "inj_par",
                    "data_file_1f",
                    "data_file_2f",
                    "prior",
                ]:
                    if key in list(config.keys()):
                        if key in ["data_file_1f", "data_file_2f"]:
                            for detkey in config[key]:
                                transfer_input.append(
                                    relative_topdir(config[key][detkey], self.resdir)
                                )

                                # exclude full path as the transfer directory is flat
                                curconfig[key][detkey] = os.path.basename(
                                    config[key][detkey]
                                )
                        else:
                            transfer_input.append(
                                relative_topdir(config[key], self.resdir)
                            )

                            # exclude full path as the transfer directory is flat
                            curconfig[key] = os.path.basename(config[key])

                # transfer ephemeris files
                for ephem in ["earth", "sun"]:
                    key = f"{ephem}ephemeris"
                    if key in config:
                        if isinstance(config[key], dict):
                            for etype in copy.deepcopy(config[key]):
                                transfer_input.append(
                                    relative_topdir(config[key][etype], self.resdir)
                                )
                                curconfig[key][etype] = os.path.basename(
                                    config[key][etype]
                                )
                        else:
                            transfer_input.append(
                                relative_topdir(config[key], self.resdir)
                            )
                            curconfig[key] = os.path.basename(config[key])

                curconfig["outdir"] = "results/"

                # add output directory to inputs in case resume file exists
                transfer_input.append(".")

                vardict["ARGS"] = f"--config {os.path.basename(configfile)}"
                vardict["INITIALDIR"] = self.resdir
                vardict["TRANSFERINPUT"] = ",".join(transfer_input)
                vardict["TRANSFEROUTPUT"] = curconfig["outdir"]
            else:
                vardict["ARGS"] = f"--config {os.path.basename(configfile)}"

            # set log files
            vardict["LOGFILE"] = os.path.join(
                self.log_directories["log"], f"{label}.log"
            )
            vardict["OUTPUTFILE"] = os.path.join(
                self.log_directories["out"], f"{label}.out"
            )
            vardict["ERRORFILE"] = os.path.join(
                self.log_directories["error"], f"{label}.err"
            )

            # write out configuration file
            parseobj = DefaultConfigFileParser()
            if not os.path.isfile(configfile):
                with open(configfile, "w") as fp:
                    fp.write(parseobj.serialize(curconfig))

            self.vars.append(vardict)

            # output the DAG configuration to a file
            if not os.path.isfile(dagconfigfile):
                # make sure pulsar files in DAG config are full paths
                with open(dagconfigfile, "w") as fp:
                    self.cf.write(fp)


class MergePELayer(CondorLayer):
    def __init__(self, pelayer, **kwargs):
        """
        Class to create submit file for merging parameter estimation outputs.

        Parameters
        ----------
        dag: :class:`PulsarPELayer`
            The parent PulsarPELayer.
        """

        super().__init__(
            pelayer.dag,
            pelayer.cf,
            default_executable="bilby_result",
            layer_name=kwargs.pop("layer_name", "cwinpy_pe_merge"),
            **kwargs,
        )

        self.parent_layer_class = pelayer

        # set parent layer
        self.parent_layer = pelayer.layer

        self.outdir = self.get_option("basedir", section="run", default=os.getcwd())

        self.submit_options["request_memory"] = "8 GB"
        self.submit_options["request_cpus"] = 1
        self.submit_options["request_disk"] = "500 MB"
        self.submit_options["universe"] = "local"

        # generate the node variables
        self.generate_node_vars()

        # generate layer
        self.generate_layer(self.vars)

    def generate_node_vars(self):
        """
        Generate the node variables for this layer.
        """

        arglist = []

        arglist.append("--results")
        arglist.extend(self.parent_layer_class.resultsfiles)

        # set results directory
        arglist.append(f"--outdir {os.path.abspath(self.parent_layer_class.resdir)}")

        label = (
            f"{self.submit_options.get('name', 'cwinpy_pe')}_"
            f"{''.join(self.parent_layer_class.dets)}_"
            f"{self.parent_layer_class.psrname}"
        )
        arglist.append(f"--label {label}")

        self.resultsfile = os.path.join(
            os.path.abspath(self.parent_layer_class.resdir), label + "*"
        )

        # set merge flag
        arglist.append("--merge")

        extension = self.parent_layer_class.sampler_kwargs.get("save", "hdf5")
        gzip = self.parent_layer_class.sampler_kwargs.get("gzip", False)
        arglist.append(f"--extension {extension}")
        if gzip and extension == "json":
            arglist.append("--gzip")

        vardict = {"ARGS": " ".join(arglist)}

        # set log files
        vardict["LOGFILE"] = os.path.join(
            self.parent_layer_class.log_directories["log"], f"{label}.log"
        )
        vardict["OUTPUTFILE"] = os.path.join(
            self.parent_layer_class.log_directories["out"], f"{label}.out"
        )
        vardict["ERRORFILE"] = os.path.join(
            self.parent_layer_class.log_directories["error"], f"{label}.err"
        )

        self.vars = [vardict]
