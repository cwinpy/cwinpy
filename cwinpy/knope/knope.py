"""
Run known pulsar parameter estimation using bilby.
"""

import os
import sys
import ast
import numpy as np

import bilby
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import (parse_args, BilbyPipeError)

from cwinpy import __version__

from lalpulsar.PulsarParametersWrapper import PulsarParametersPy

description="""\
A script to use Bayesian inference to estimate the parameters of a \
continuous gravitational-wave signal from a known pulsar."""


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

    pulsarparser = parser.add_group('Pulsar inputs')
    pulsarparser.add('-p', '--pulsar', default=None,
                     help='The name of the pulsar')
    pulsarparser.add('--par-file', help='The path to a TEMPO(2) style '
                     'file containing the pulsar parameters',
                     required=True, default=None)

    dataparser = parser.add_argument_group('Data inputs')
    dataparser.add('-d', '--detector', action='append', help='The abbreviated '
                   'name of a detector to analyse (multiple detectors can be '
                   'passed with multiple arguments)', default=None)
    dataparser.add('-f', '--data-file', action='append', help='The path to '
                   'a data file for a given detector. The format should be of '
                   'the form "DET:PATH", where DET is the detector name '
                   '(multiple files can be passed with multiple arguments).',
                   required=True)

    samplerparser = parser.add_group('Sampler inputs')

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
        self.set_defaults(kwargs)
        self.set_data()


    def set_defaults(self, kwargs):
        """
        Set the default run parameters.

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
                detdata = data.split(':')  # split detector and path
                if self.detectors is None:
                    if len(detdata) == 2:
                        self.detectors = [detdata[0]]
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
                        self.datafiles[self.detectors[0]] = data
                    else:
                        raise ValueError("Data string must be of the form "
                                         "'DET:FILEPATH'")
            elif isinstance(data, list):
                # pass through list and check strings
            elif isinstance(data, dict):
                # pass through dict and check values
            else:
                raise TypeError("Data files are not of a recognised type")
        else:
            raise KeyError("Data files must be given")


    def set_data(self):
        """
        Set the :class:`cwinpy.data.MultiHeterodynedData` object.
        """



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
    """

    if 'cwinpy_knope' in sys.argv:
        # get command line arguments
        parser = create_parser()

        try:
            args, unknown_args = parse_args(parser)
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
