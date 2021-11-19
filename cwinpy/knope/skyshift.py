def skyshift_pipeline(**kwargs):
    """
    Run skyshift pipeline within Python. This will create a `HTCondor <https://research.cs.wisc.edu/htcondor/>`_
    DAG for consecutively running a ``cwinpy_heterodyne`` and multiple ``cwinpy_pe`` instances on a
    computer cluster. Optional parameters that can be used instead of a configuration file (for
    "quick setup") are given in the "Other parameters" section.

    Parameters
    ----------
    config: str
        A configuration file, or :class:`configparser:ConfigParser` object,
        for the analysis.

    Other parameters
    ----------------
    run: str
        The name of an observing run for which open data exists, which will be
        heterodyned, e.g., "O1".
    detector: str, list
        The detector, or list of detectors, for which the data will be
        heterodyned. If not set then all detectors available for a given run
        will be used.
    samplerate: str:
        Select the sample rate of the data to use. This can either be 4k or
        16k for data sampled at 4096 or 16384 Hz, respectively. The default
        is 4k, except if running on hardware injections for O1 or later, for
        which 16k will be used due to being required for the highest frequency
        source. For the S5 and S6 runs only 4k data is available from GWOSC,
        so if 16k is chosen it will be ignored.
    pulsar: str, list
        The path to a TEMPO(2)-style pulsar parameter file to heterodyne. If a
        pulsar name is given instead of a parameter file then an attempt will
        be made to find the pulsar's ephemeris from the ATNF pulsar catalogue,
        which will then be used.
    osg: bool
        Set this to True to run on the Open Science Grid rather than a local
        computer cluster.
    output: str,
        The location for outputting the heterodyned data. By default the
        current directory will be used. Within this directory, subdirectories
        for each detector will be created.
    joblength: int
        The length of data (in seconds) into which to split the individual
        heterodyne jobs. By default this is set to 86400, i.e., one day. If
        this is set to 0, then the whole dataset is treated as a single job.
    accounting_group_tag: str
        For LVK users this sets the computing accounting group tag.
    usetempo2: bool
        Set this flag to use Tempo2 (if installed) for calculating the signal
        phase evolution for the heterodyne rather than the default LALSuite
        functions.

    Returns
    -------
    dag:
        An object containing a pycondor :class:`pycondor.Dagman` object.
    """

    # if "config" in kwargs:
    #    hetconfigfile = kwargs.pop("config")
    #    peconfigfile = hetconfigfile
    # else:   pragma: no cover
    #    parser = ArgumentParser(
    #        description=(
    #            "A script to create a HTCondor DAG to process GW strain data "
    #            "by heterodyning it based on the expected phase evolution for "
    #            "a selection of pulsars, and then perform parameter "
    #            "estimation for the unknown signal parameters of those "
    #            "sources."
    #        )
    #    )


def skyshift_pipeline_cli(**kwargs):  # pragma: no cover
    """
    Entry point to ``cwinpy_skyshift_pipeline`` script. This just calls
    :func:`cwinpy.knope.skyshift_pipeline`, but does not return any objects.
    """

    kwargs["cli"] = True  # set to show use of CLI
    _ = skyshift_pipeline(**kwargs)
