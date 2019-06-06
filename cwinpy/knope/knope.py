"""
Run known pulsar parameter estimation using bilby.
"""

import os
import sys
import ast
import signal
import numpy as np

from cwinpy import __version__
from ..data import MultiHeterodynedData
from ..likelihood import TargetedPulsarLikelihood

import bilby
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import (parse_args, BilbyPipeError)


description="""\
A script to use Bayesian inference to estimate the parameters of a \
continuous gravitational-wave signal from a known pulsar."""


def sighandler(signum, frame):
    # perform periodic eviction with exit code 130
    # see https://git.ligo.org/lscsoft/bilby_pipe/blob/0b5ca550e3a92494ef3e04801e79a2f9cd902b44/bilby_pipe/parser.py#L270
    sys.exit(130)


def create_parser():
    """
    Create the argument parser.
    """

    parser = BilbyArgParser(
        prog=sys.argv[0],
        description=description,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False
    )
    parser.add("ini", type=str, is_config_file=True, help="Configuration ini file")
    parser.add(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    parser.add(
        "--periodic-restart-time",
        default=10800,
        type=int,
        help=(
            'Time after which the job will be self-evicted with code 130. '
            'After this, condor will restart the job. Default is 10800s. '
            'This is used to decrease the chance of HTCondor hard evictions.'
        ),
    )

    pulsarparser = parser.add_argument_group('Pulsar inputs')
    pulsarparser.add(
        '-p', '--pulsar',
        default=None,
        help='The name of the pulsar',
    )
    pulsarparser.add(
        '--par-file',
        required=True,
        default=None,
        help=(
            'The path to a TEMPO(2) style file containing the pulsar '
            'parameters'
        ),
    )

    dataparser = parser.add_argument_group('Data inputs')
    dataparser.add('-d', '--detector',
                   action='append',
                   help=(
                       'The abbreviated name of a detector to analyse. '
                       'Multiple detectors can be passed with multiple '
                       'arguments, e.g., --detector H1 --detector L1.'
                   ),
                   default=None,
    )
    dataparser.add('-f', '--data-file',
                   action='append',
                   required=True,
                   help=(
                       'The path to a data file for a given detector. The '
                       'format should be of the form "DET:PATH", where DET is '
                       'the detector name. Multiple files can be passed with '
                       'multiple arguments, e.g., --data-file H1:H1data.txt '
                       '--data-file L1:L1data.txt.'
                    ),
    )

    outputparser = parser.add_argument_group('Output')
    outputparser.add(
        '-o', '--outdir',
        default=None,
        help='The output directory for the results',
    )
    outputparser.add('-l', '--label',
                     default=None,
                     help='The output filename label for the results',
    )

    samplerparser = parser.add_argument_group('Sampler inputs')
    samplerparser.add('-s', '--sampler',
                      default='dynesty',
                      help=(
                          'The sampling algorithm to use bilby (default: '
                          '%(default)s)'
                      ),
    )
    samplerparser.add(
        '--sampler-kwargs',
        default=None,
        help=(
            'The keyword arguments for running the sampler. This should be in '
            'the format of a standard Python dictionary.'
        ),
    )
    samplerparser.add(
        '--likelihood',
        default='studentst',
        help=(
            'The name of the likelihood function to use. This can be either '
            '"studentst" or "gaussian".'
        ),
    )
    samplerparser.add(
        '--prior',
        required=True,
        help=(
            'The path to a bilby-style prior file defining the parameters to '
            'be estimated and their prior probability distributions.'
        ),
    )

    return parser
    

class KnopeRunner(object):
    """
    Set up and run the known pulsar parameter estimation.
    
    Parameters
    ----------
    kwargs: dict
        A dictionary of analysis setup parameters.
    """

    def __init__(self, kwargs):
        self.set_parameters(kwargs)
        self.set_data()
        self.set_likelihood()

    def set_parameters(self, kwargs):
        """
        Set the run parameters and their defaults.

        Parameters
        ----------
        kwargs: dict
            Dictionary of run parameters.
        """

        if not isinstance(kwargs, dict):
            raise TypeError("Argument must be a dictionary")

        # pulsar parameters
        self.pulsar = kwargs.get('pulsar', None)

        if 'par_file' not in kwargs:
            raise KeyError('A pulsar parameter file must be provided')
        else:
            self.parfile = kwargs['par_file']

        # data parameters
        if 'detector' in kwargs:
            if isinstance(kwargs['detector'], str):
                self.detectors = [kwargs['detector']]
            elif isinstance(kwargs['detector'], list):
                self.detectors = []
                for det in kwargs['detector']:
                    try:
                        # remove additional quotation marks from string
                        thisdet = ast.literal_eval(det)
                    except ValueError:
                        thisdet = det
                        
                    if isinstance(det, str):
                        self.detectors.append(det)
                    else:
                        raise TypeError("Detector must be a string")
        else:
            self.detectors = None
        
        if 'data_file' in kwargs: 
            self.datafiles = {}
            try:
                data = ast.literal_eval(kwargs['data_file'])
            except ValueError:
                data = kwargs['data_file']

            if isinstance(data, str):
                # make into a list
                data = [data]

            if isinstance(data, dict):
                # make into a list
                if self.detectors is None:
                    self.detectors = list(data.keys())
                else:
                    for det in data.keys():
                        if det not in self.detectors:
                            data.pop(det)

                data = list(data.values())

            if isinstance(data, list):
                # pass through list and check strings
                detlist = []
                for dfile in data:
                    detdata = dfile.split(':')  # split detector and path
                    if self.detectors is None:
                        if len(detdata) == 2:
                            detlist.append(detdata[0])
                            self.datafiles[detdata[0]] = detdata[1]
                        else:
                            raise ValueError("Data string must be of the form "
                                             "'DET:FILEPATH'")
                    else:
                        if len(detdata) == 2:
                            if detdata[0] not in self.detectors:
                                raise ValueError("Data file does not have "
                                                 "consistent detector")
                            self.datafiles[detdata[0]] = detdata[1]
                        elif len(detdata) == 1 and len(self.detectors) == 1:
                            self.datafiles[self.detectors[0]] = dfile
                        else:
                            raise ValueError("Data string must be of the form "
                                             "'DET:FILEPATH'")

                # set detectors
                if self.detectors is None:
                    self.detectors = detlist
            else:
                raise TypeError("Data files are not of a recognised type")

            # remove any detectors than are not required
            if len(self.detectors) < len(self.datafiles):
                for det in list(self.datafiles.keys()):
                    if det not in self.detectors:
                        self.datafiles.pop(det)

            if len(self.datafiles) < len(self.detectors):
                raise ValueError("Fewer data files than specified detectors")
        else:
            raise KeyError("Data files must be given")

        # output parameters
        self.outdir = kwargs.get('outdir', None)
        self.label = kwargs.get('label', None)

        # sampler parameters
        self.sampler = kwargs.get('sampler', 'dynesty')
        self.sampler_kwargs = kwargs.get('sampler_kwargs', {})
        if isinstance(self.sampler_kwargs, str):
            try:
                self.sampler_kwargs = ast.literal_eval(self.sampler_kwargs)
            except ValueError:
                raise ValueError("Unable to parser sampler keyword arguments")
        self.likelihoodtype = kwargs.get('likelihood', 'studentst')
        self.prior = kwargs.get('prior', None)
        if not isinstance(self.prior, (str, bilby.core.prior.PriorDict)):
            raise ValueError('The prior is not defined')

        # default restart time to 1000000 seconds if not running through CLI
        self.periodic_restart_time = kwargs.get('periodic_restart_time',
                                                10000000)

    def set_data(self):
        """
        Set the :class:`cwinpy.data.MultiHeterodynedData` object.
        """

        self.hetdata = MultiheterodynedData(data=self.datafiles,
                                            par=self.parfile)

    def set_likelihood(self):
        """
        Set the likelihood function.
        """

        self.likelihood = TargetedPulsarLikelihood(
            data=self.hetdata,
            priors=bilby.core.prior.PriorDict(self.prior),
            likelihood=self.likelihoodtype
        )

    def run_sampler(self):
        """
        Run bilby to sample the posterior.
        
        Returns
        -------
        res
            A bilby Results object.
        """

        # restart the job after 10800 seconds (for if running on Condor to
        # prevent hard evictions)
        signal.signal(signal.SIGALRM, handler=sighandler)
        signal.alarm(self.periodic_restart_time)

        self.result = bilby.run_sampler(
            sampler=self.sampler,
            prior=self.prior,
            **self.sampler_kwargs)

        return self.result


def knope(**kwargs):
    """
    Entry point to cwinpy_knope script, or for running an analysis directly
    from Python.

    Parameters
    ----------
    pulsar: str
        The name of the pulsar being analysed.
    par_file: str
        The path to a TEMPO(2) style pulsar parameter file for the source.
    detector: str, list
        A string, or list of strings, containing the abbreviated names for
        the detectors being analysed (e.g., "H1", "L1", "V1").
    data_file: str, list, dict
        A string, list, or dictionary contain paths to the heterodyned data
        to be used in the analysis. For a single detector this can be a single
        string. For multiple detectors a list can be passed with the file path
        for each detector given in the same order as the list passed to the
        ``detector`` argument, or as a dictionary with each file path keyed to
        the associated detector. in the latter case the ``detector`` keyword
        argument is not required, unless wanting to analyse fewer detectors
        than passed.
    sampler: str
        The sampling algorithm to use within bilby. The default is "dynesty".
    sampler_kwargs: dict
        A dictionary of keyword arguments to be used by the given sampler
        method.
    outdir: str
        The output directory for the results.
    label: str
        The name of the output file (excluding the '.json' extension) for the
        results.
    likelihood: str
        The likelihood function to use. At the moment this can be either
        'studentst' or 'gaussian', with 'studentst' being the default.
    prior: str, PriorDict
        A string to a bilby-style
        `prior <https://lscsoft.docs.ligo.org/bilby/prior.html>`_ file, or a
        bilby :class:`~bilby.core.prior.PriorDict` object. This defines the
        parameters that are to be estimated and their prior distributions.
    periodic_restart_time: int
        The number of seconds after which the run will be evicted with a
        ``130`` exit code. This prevents hard evictions if running under
        HTCondor. For running via the command line interface, this defaults to
        10800 seconds (3 hours), at which point the job will be stopped (and
        then restarted if running under HTCondor). If running directly within
        Python this defaults to 10000000. 
    """

    if 'cwinpy_knope' == os.path.split(sys.argv[0])[-1]:
        # get command line arguments
        parser = create_parser()

        try:
            args, unknown_args = parse_args(sys.argv[1:], parser)
        except BilbyPipeError as e:
            raise IOError("{}".format(e))

        # convert args to a dictionary
        dargs = vars(args)
    else:
        dargs = kwargs

    # set up the run
    runner = KnopeRunner(dargs)

    # run the sampler
    runner.run_sampler()
