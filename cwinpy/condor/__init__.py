import copy
import fnmatch
import os
import shutil

from htcondor import Submit


class CondorLayer:
    # general options (name: (type, default))
    OPTIONS = {
        "universe": (str, "vanilla"),
        "getenv": (bool, False),
        "accounting_group": (str, None),
        "accounting_group_user": (str, None),
        "notification": (str, "Never"),
    }

    def __init__(self, dag, cf, **kwargs):
        """
        A class for a generic Condor layer.

        Parameters
        ----------
        dag: :class:`hcondor.dags.DAG`
            The HTCondor DAG associated with this layer.
        cf: :class:`configparser.ConfigParser`
            The configuration file for the DAG set up.
        section_prefix: str
            A potential prefix for any section names specific to this layer.
        default_executable: str
            The default executable used by the submit files in this layer.
        layer_name: str
            The default name for this layer within the DAG. This will also be
            used to label the submit files and DAG jobs.
        """

        self.dag = dag
        self.cf = cf

        # check for a section prefix to use for the configuration file
        self.section_prefix = kwargs.get("section_prefix", "")

        # dictionary to contain generic submit options for all jobs
        self.submit_options = {}

        # set executable
        self.executable = kwargs.get("default_executable", None)

        # set layer name
        self.layer_name = self.get_option(
            "name", default=kwargs.get("layer_name", None)
        )

        # set general options
        self.set_general_options()

    @staticmethod
    def conda_prefix():
        """
        Detector whether in a conda environment or not and return the path
        prefix if within a conda environment otherwise return None.

        Returns
        -------
        prefix: str
            The conda environment path prefix.
        """

        return os.environ.get("CONDA_PREFIX", None)

    @property
    def executable(self):
        return self.submit_options["executable"]

    @executable.setter
    def executable(self, exec):
        """
        Set the executable for the Condor submit file.

        Parameters
        ----------
        exec: str
            The default executable name to use if it cannot be found.
        """

        # set the executable
        exec = self.get_option("executable", default=exec)

        # make sure we get the full path (check whether in a conda environment)
        if self.conda_prefix() is not None:
            # set executable in conda path
            exec = os.path.join(self.conda_prefix(), "bin", f"{exec}")

        exe = shutil.which(exec)
        if exe is not None:
            # reset executable to have the full path
            self.submit_options["executable"] = exe
        else:
            raise OSError(f"{exec} not installed on this system, unable to proceed")

    def get_option(self, valuename, section=None, otype=None, default=None):
        # first try adding section prefix on to section name
        sectionname = None
        if section is not None:
            sectionname = self.section_prefix + "_" + section
            if not self.cf.has_section(sectionname):
                sectionname = section
        else:
            for section in self.cf.sections():
                if self.section_prefix:
                    # check section starts with the section prefix
                    if not section.startswith(self.section_prefix):
                        continue

                if valuename in self.cf.items(section):
                    sectionname = section
                    break
            else:
                raise IOError(
                    f"Configfile does not contain a section with option {valuename}"
                )

        if otype is None or otype is str or otype == "str":
            getfunc = self.cf.get
        elif otype is int or otype == "int":
            getfunc = self.cf.getint
        elif otype is bool or otype in ["bool", "boolean"]:
            getfunc = self.cf.getboolean
        elif otype is float or otype == "float":
            getfunc = self.cf.getfloat
        else:
            raise TypeError(f"Attempting to get unknown type {otype} section value")

        return getfunc(sectionname, valuename, fallback=default)

    def set_option(
        self, valuename, section=None, optionname=None, otype=None, default=None
    ):
        value = self.get_option(section, valuename, otype=otype, default=default)

        if optionname is None:
            optionname = valuename

        if value is not None:
            self.submit_options[optionname] = value

    def set_general_options(self):
        """
        Set all the generic options for all layers.
        """

        for option, td in self.OPTIONS.items():
            self.set_option(option, otype=td[0], default=td[1])

    def generate_submit_job(self, **kwargs):
        """
        Generate a submit object.
        """

        # dictionary to contain specific submit options
        submit = {}

        submit.update(copy.deepcopy(self.submit_options))
        submit.update(copy.deepcopy(kwargs))

        # add arguments
        submit["arguments"] = "$(ARGS)"

        # add requirements
        if hasattr(self, "requirements"):
            if isinstance(self.requirements, list):
                submit["requirements"] = " && ".join(self.requirements)
            else:
                submit["requirements"] = self.requirements

        return Submit(submit)

    def generate_layer(self, vars, parentname=None, **kwargs):
        """
        Generate a new layer in the DAG.

        Parameters
        ----------
        vars: list
            A list of dictionaries containing the variables for each job in the
            layer.
        parentname: str
            A string containing the name that any parent jobs have, which can
            have the "*" wildcard.
        """

        layer = self.dag.layer(
            name=self.layer_name,
            submit_description=self.generate_submit_job(**kwargs),
            retries=self.get_option("retry", default=2),
            vars=vars,
        )

        if parentname is not None:
            # find parent nodes and add them
            selector = lambda x: fnmatch.fnmatch(x.name, parentname)
            layer.add_parents(self.dag.select(selector))
