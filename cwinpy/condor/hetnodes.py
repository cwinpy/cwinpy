import copy
import os
import re

from configargparse import DefaultConfigFileParser

from ..heterodyne.base import Heterodyne
from ..utils import relative_topdir
from . import CondorLayer


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

        # check for use of OSG
        self.osg = self.get_option("osg", default=False)
        self.outdir = self.get_option("basedir", section="run", default=os.getcwd())

        # check for use of tempo2
        self.usetempo2 = self.get_option("usetempo2", default=False)

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
        self.set_option("request_memory", default="16 GB")
        self.set_option("request_cpus", otype=int, default=1)
        self.set_option("request_disk", default="2 GB")

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

        environment = []
        if self.usetempo2:
            tempo2 = os.environ.get("TEMPO2", None)

            if tempo2 is None:
                raise ValueError("No TEMPO2 environment variable exists")

            # add TEMPO2 environment variable to the submit file
            environment.append(f"TEMPO2={tempo2}")

        # check whether GWOSC is required
        self.require_gwosc = kwargs.get("require_gwosc", False)

        # set scitokens to access proprietary data
        if not self.require_gwosc:
            self.submit_options["use_oauth_services"] = "igwn"
            self.submit_options[
                "igwn_oauth_permissions"
            ] = "read:/ligo read:/virgo read:/kagra"
            environment.append(
                "BEARER_TOKEN_FILE=$$(CondorScratchDir)/.condor_creds/igwn.use"
            )

        if environment:
            self.submit_options["environment"] = f'"{" ".join(environment)}"'

        # additional options
        additional_options = {}
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

        if self.osg:
            ligojob = self.submit_options.get("accounting_group", "").startswith(
                "ligo."
            )

            if ligojob:
                # set to check that proprietary LIGO frames are available
                self.requirements.append("(HAS_CVMFS_IGWN_PRIVATE_DATA =?= True)")

            # allow use of local pool (https://computing.docs.ligo.org/guide/htcondor/access/#local-access-points)
            additional_options["MY.flock_local"] = "True"

            if self.submit_options.get("desired_sites", ""):
                # allow specific OSG sites to be requested
                additional_options["MY.DESIRED_Sites"] = self.submit_options[
                    "desired_sites"
                ]
                self.requirements.append("(IS_GLIDEIN =?= True)")
            elif ligojob:
                # if desired_sites are not explicitly specified, default any
                # "ligo" tagged jobs to only run on the local pool
                # (heterodyning is not suited to running on the OSG due to the
                # large amounts of frame data that must be transferred)
                additional_options["MY.DESIRED_Sites"] = '"none"'

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
                ] = "/opt/conda/envs/python38/bin/cwinpy_heterodyne"
                additional_options["MY.SingularityImage"] = (
                    '"/cvmfs/singularity.opensciencegrid.org/matthew-pitkin/'
                    'cwinpy-containers/cwinpy-dev-python38:latest"'
                )
                self.requirements.append("(HAS_SINGULARITY =?= True)")
                self.submit_options["transfer_executable"] = False

            # NOTE: the next two statements are currently only required for OSG running,
            # but at the moment not all local pools advertise the CVMFS repo flags
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
                    "environment or the CWInPy development singularity "
                    "container."
                )

            # check if using GWOSC frames from CVMFS
            if self.require_gwosc:
                self.requirements.append("(HAS_CVMFS_gwosc_osgstorage_org =?= TRUE)")

        # generate the node variables
        self.generate_node_vars(configurations)

        # generate layer
        self.generate_layer(
            self.vars,
            parentname=kwargs.get("parentname", None),
            submitoptions=additional_options,
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

        dagconfigfile = os.path.join(configlocation, "heterodyne_pipeline_config.ini")

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

            # output the DAG configuration to a file that will
            # later be read and stored in the HeterodynedData objects
            if not os.path.isfile(dagconfigfile):
                # make sure pulsar files in DAG config are full paths
                if self.cf.has_option("ephemerides", "pulsarfiles"):
                    self.cf.set(
                        "ephemerides", "pulsarfiles", str(config["pulsarfiles"])
                    )

                with open(dagconfigfile, "w") as fp:
                    self.cf.write(fp)

            config["cwinpy_heterodyne_pipeline_config_file"] = dagconfigfile

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

                        # create empty dummy files, so Condor doesn't complain about
                        # files not existing see https://stackoverflow.com/a/12654798/1862861
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
                for ephem in ["earthephemeris", "sunephemeris"]:
                    if ephem in config:
                        for etype in copy.deepcopy(config[ephem]):
                            transfer_input.append(
                                relative_topdir(config[ephem][etype], resultsdir)
                            )

                            config[ephem][etype] = os.path.basename(
                                config[ephem][etype]
                            )

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
            vardict["LOGFILE"] = os.path.join(
                self.log_directories["log"], f"{logstr}.log"
            )
            vardict["OUTPUTFILE"] = os.path.join(
                self.log_directories["out"], f"{logstr}.out"
            )
            vardict["ERRORFILE"] = os.path.join(
                self.log_directories["error"], f"{logstr}.err"
            )

            # write out the configuration files for each job
            parseobj = DefaultConfigFileParser()
            with open(configfile, "w") as fp:
                fp.write(parseobj.serialize(config))

            self.vars.append(vardict)


class MergeLayer(CondorLayer):
    def __init__(self, dag, cf, configurations, **kwargs):
        """
        Class to create submit file for jobs that merge heterodyne outputs.

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
            layer_name=kwargs.pop("layer_name", "cwinpy_heterodyne_merge"),
            **kwargs,
        )

        self.outdir = self.get_option("basedir", section="run", default=os.getcwd())

        self.log_directories = {}
        for logtype in ["log", "error", "out"]:
            self.log_directories[logtype] = self.get_option(
                logtype, default=os.path.join(os.path.abspath(self.outdir), "log")
            )
            if not os.path.exists(self.log_directories[logtype]):
                os.makedirs(self.log_directories[logtype])

        self.submit_options["request_memory"] = "2 GB"
        self.submit_options["request_disk"] = "2 GB"
        self.submit_options["request_cpus"] = 1
        self.submit_options["universe"] = "local"

        additional_options = {}
        additional_options["log"] = "$(LOGFILE)"
        additional_options["output"] = "$(OUTPUTFILE)"
        additional_options["error"] = "$(ERRORFILE)"

        # generate the node variables
        self.generate_node_vars(configurations)

        # generate layer
        self.generate_layer(
            self.vars,
            parentname=kwargs.get("parentname", None),
            submitoptions=additional_options,
        )

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
        configfile = os.path.abspath(
            os.path.join(
                configlocation,
                "{}{}_{}_merge.ini".format(psrstring, detector, int(freqfactor)),
            )
        )

        # output merge job configuration
        configs = {}
        configs["remove"] = self.get_option(
            "remove", section="merge", otype=bool, default=False
        )
        configs["heterodynedfiles"] = heterodynedfiles
        configs["output"] = output
        configs["overwrite"] = self.get_option(
            "overwrite", section="merge", otype=bool, default=True
        )

        parseobj = DefaultConfigFileParser()
        with open(configfile, "w") as fp:
            fp.write(parseobj.serialize(configs))

        vardict["ARGS"] = f"--config {configfile}"

        # set log files
        logstr = f"{self.layer_name}"
        vardict["LOGFILE"] = os.path.join(self.log_directories["log"], f"{logstr}.log")
        vardict["OUTPUTFILE"] = os.path.join(
            self.log_directories["out"], f"{logstr}.out"
        )
        vardict["ERRORFILE"] = os.path.join(
            self.log_directories["error"], f"{logstr}.err"
        )

        self.vars.append(vardict)
