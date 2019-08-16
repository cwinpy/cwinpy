"""
Run P-P plot testing for cwinpy_knope.
"""

import os
import glob
import bilby
from .knope import knope_dag


def generate_pp_plots(**kwargs):
    """
    Script entry point, or function, to generate P-P plots (see, e.g., [1]_): a
    frequentist-style evaluation to test that the true value of a parameter
    falls with a given Bayesian credible interval the "correct" amount of
    times, provided the trues values are drawn from the same prior as used when
    evaluating the posteriors.

    Parameters
    ----------
    path: str
        Glob-able directory path pattern to a set of JSON format bilby results
        files.
    output: str
        Output filename for the PP plot.
    parameters: list
        A list of the parameters to include in the PP plot.

    References
    ----------

        .. [1] `T. Sidery et al. <https://arxiv.org/abs/1312.6013>`_,
           *PRD*, **89**, 084060 (2014)
    """

    if "path" in kwargs:
        path = kwargs["path"]

        # default output file is a PNG image in the current directory
        outfile = kwargs.get("output", os.path.join(os.getcwd(), "ppplot.png"))

        # default parameters are "h0", "phi0", "psi" and "iota"
        parameters = kwargs.get("parameters", ["h0", "phi0", "psi", "iota"])
    elif "cwinpy_knope_generate_pp_plots" == os.path.split(sys.argv[0])[-1]:
        try:
            from argparse import ArgumentParser
        except ImportError:
            raise ImportError("To run 'cwinpy_knope_generate_pp_plots'")

        parser = ArgumentParser(
            description=(
                "A script to create a PP plot of CW signal parameters"
            )
        )
        parser.add_argument(
            "--path", "-p",
            required=True,
            help=(
                "A glob-able path pattern to a set of bilby JSON results "
                "files for the set of simulations."
            ),
            dest="path",
        )
        parser.add_argument(
            "--output", "-o",
            help=(
                "The output plot file name [default: %(default)s]."
            ),
            default=os.path.join(os.getcwd(), "ppplot.png"),
            dest="outfile",
        )
        parser.add_argument(
            "--parameter",
            action="append",
            default=["h0", "phi0", "psi", "iota"],
            help=(
                "The parameters with which to create the PP plots. "
                "[default: %(default)s]."
            ),
        )

        args = parser.parse_args()
        path = args.path
        outfile = args.outfile
        parameters = args.parameters
    else:
        raise KeyError("A 'path' keyword must be supplied.")

    # get results files
    try:
        resfiles = [
            rfile
            for rfile in glob.glob(path)
            if os.path.splitext(rfile)[1] == ".json"
        ]
    except Exception as e:
        raise IOError("Problem finding results files: {}".format(e))

    if len(resfiles) == 0:
        raise IOError(
            "Problem finding results files. Probably an invalid path!"
        )

    credints = {param: [] for param in parameters}

    # read in results files and get injection credible interval
    for rfile in resfiles:
        try:
            results = bilby.core.result.read_in_result(rfile)
        except Exception as e:
            raise IOError(
                "Could not read in results from file '{}'\n{}".format(rfile, e)
            )

        # check parameters are in results file

        # get truths from results

        # get credible interval containing truth

    # make plots!


class KnopePPPlotsDAG(object):
    """
    This class will generate a HTCondor Dagman job to create a number of
    simulated gravitational-wave signals from pulsars in Gaussian noise. These
    will be analysed using the ``cwinpy_knope`` script to sample the posterior
    probability distributions of the required parameter space. For each
    simulation and parameter combination the credible interval (symmetric in
    probability about the median) in which the known true signal value lies
    """