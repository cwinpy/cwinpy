"""
Run known pulsar parameter estimation using bilby.
"""

import os
import sys
import ast
import signal
import numpy as np

from cwinpy import __version__
from ..data import (HeterodynedData,
                    MultiHeterodynedData)
from ..likelihood import TargetedPulsarLikelihood
from .._version import get_versions

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
    parser.add(
        "--config",
        type=str,
        is_config_file=True,
        help="Configuration ini file",
    )
    parser.add(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=get_versions()['version']),
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
            'parameters.'
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
    dataparser.add('--data-file',
                   default=None,
                   action='append',
                   help=(
                       'The path to a heterodyned data file for a given '
                       'detector. The format should be of the form '
                       '"DET:PATH",  where DET is the detector name. '
                       'Multiple files can be passed with multiple '
                       'arguments, e.g., --data-file H1:H1data.txt '
                       '--data-file L1:L1data.txt. This data will be assumed '
                       'to be that in a search for a signal from the l=m=2 '
                       'mass quadrupole and therefore heterodyned at twice '
                       'the source\'s rotation frequency. To add data '
                       'explicitly setting the heterodyned frequency at twice '
                       'the rotation frequency use "--data-file-2f", or for '
                       'data at the rotation frequency use "--data-file-1f".'
                    ),
    )
    dataparser.add('--data-file-2f',
                   default=None,
                   action='append',
                   help=(
                       'The path to a data file for a given detector where '
                       'the data is explicitly given as being heterodyned at '
                       'twice the source\'s rotation frequency. The inputs '
                       'should be in the same format as those given to the '
                       '"--data-file" flag. This flag should generally be '
                       'preferred over the use of "--data-file".'
                    ),
    )
    dataparser.add('--data-file-1f',
                   default=None,
                   action='append',
                   help=(
                       'The path to a data file for a given detector where '
                       'the data is explicitly given as being heterodyned at '
                       'the source\'s rotation freqeuncy. The inputs should '
                       'be in the same format as those given to the '
                       '"--data-file" flag.'
                    ),
    )
    dataparser.add('--data-kwargs',
                   default=None,
                   help=(
                       'A Python dictionary containing keywords to pass to '
                       'the HeterodynedData object.'
                   ),
    )

    simparser = parser.add_argument_group('Simulated data')
    simparser.add('--inj-par',
                  type=str,
                  default=None,
                  help=(
                      'The path to a TEMPO(2) style file containing the '
                      'parameters of a simulated signal to "inject" into the '
                      'data.'
                  ),
    )
    simparser.add('--inj-times',
                  default=None,
                  help=(
                      'A Python list of pairs of times between which to add '
                      'the simulated signal (specified by the "--inj-par" '
                      'flag) to the data. By default the signal is added into '
                      'the whole data set.'
                  ),
    )
    simparser.add('--fake-asd',
                  action='append',
                  default=None,
                  help=(
                      'This flag sets the code to perform the analysis on '
                      'simulated Gaussian noise, with data samples drawn from '
                      'a Gaussian distribution defined by a given amplitude '
                      'spectral density. The flag is set in a similar way to '
                      'the "--data-file" flag. The argument can either be a '
                      'float giving an ASD value, or a string containing a '
                      'detector alias to produce noise from the design curve '
                      'for that detector, or a string containing a path to a '
                      'file with the noise curve for a detector. This can be '
                      'used in conjunction with the "--detector" flag, e.g., '
                      '"--detector H1 --fake-asd 1e-23", or without the '
                      '"--detector" flag, e.g., "--fake-asd H1:1e-23". Values '
                      'for multiple detectors can be passed by repeated use '
                      'of the flag, noting that if used in conjunction with '
                      'the "--detector" flag detectors and ASD values should '
                      'be added in the same order, e.g., "--detector H1 '
                      '--fake-asd H1 --detector L1 --fake-asd L1". This flag '
                      'is ignored is "--data-file" values for the same '
                      'detector have already been passed. The fake data that '
                      'is produced is assumed to be that for a signal at '
                      'twice the source rotation frequency. To explicitly set '
                      'fake data at once or twice the rotation frequency '
                      'use the "--fake-asd-1f" and "--fake-asd-2f" flags '
                      'instead.'
                  ),
    )
    simparser.add('--fake-asd-1f',
                  action='append',
                  default=None,
                  help=(
                      'This flag sets the data to be Gaussian noise '
                      'explicitly for a source emitting at the rotation '
                      'frequency. See the documentation for "--fake-asd" for '
                      'details of its use.'
                  ),
    )
    simparser.add('--fake-asd-2f',
                  action='append',
                  default=None,
                  help=(
                      'This flag sets the data to be Gaussian noise '
                      'explicitly for a source emitting at twice the rotation '
                      'frequency. See the documentation for "--fake-asd" for '
                      'details of its use.'
                  ),
    )
    simparser.add('--fake-sigma',
                  action='append',
                  default=None,
                  help=(
                      'This flag is equivalent to "--fake-asd", but '
                      'instead of taking in an amplitude spectral density '
                      'value it takes in a noise standard deviation.'
                  ),
    )
    simparser.add('--fake-sigma-1f',
                  action='append',
                  default=None,
                  help=(
                      'This flag is equivalent to "--fake-asd-1f", but '
                      'instead of taking in an amplitude spectral density '
                      'value it takes in a noise standard deviation.'
                  ),
    )
    simparser.add('--fake-sigma-2f',
                  action='append',
                  default=None,
                  help=(
                      'This flag is equivalent to "--fake-asd-2f", but '
                      'instead of taking in an amplitude spectral density '
                      'value it takes in a noise standard deviation.'
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

        # keyword arguments for creating the HeterodynedData objects
        self.datakwargs = kwargs.get('data_kwargs', {})
 
        if 'par_file' not in kwargs:
            raise KeyError('A pulsar parameter file must be provided')
        else:
            self.datakwargs['par'] = kwargs['par_file']

        # injection parameters
        self.datakwargs.setdefault('injpar', kwargs.get('inj_par', None))
        self.datakwargs.setdefault('inject', (
            False if self.datakwargs['injpar'] is None else True)
        )

        # get list of times at which to inject the signal
        self.datakwargs.setdefault('injtimes', kwargs.get('inj_times', None))
        try:
            self.datakwargs['injtimes'] = ast.literal_eval(
                self.datakwargs['injtimes']
            )
        except ValueError:
            pass

        if self.datakwargs['injtimes'] is not None:
            if not isinstance(self.datakwargs['injtimes'], (list, np.ndarray)): 
                raise TypeError('Injection times must be a list')

        # data parameters
        if 'detector' in kwargs:
            if isinstance(kwargs['detector'], str):
                detectors = [kwargs['detector']]
            elif isinstance(kwargs['detector'], list):
                detectors = []
                for det in kwargs['detector']:
                    try:
                        # remove additional quotation marks from string
                        thisdet = ast.literal_eval(det)
                    except ValueError:
                        thisdet = det

                    if isinstance(det, str):
                        detectors.append(det)
                    else:
                        raise TypeError("Detector must be a string")
        else:
            detectors = None

        # empty heterodyned data object
        self.hetdata = MultiHeterodynedData()

        # set the heterodyned data structure
        resetdetectors = True if detectors is None else False
        if ('data_file' in kwargs or 'data_file_1f' in kwargs
                or 'data_file_2f' in kwargs): 
            data2f = []
            for kw in ['data_file', 'data_file_2f']:
                if kw in kwargs:
                    try:
                        data2f = ast.literal_eval(kwargs[kw])
                    except ValueError:
                        data2f = kwargs[kw]
                    break

            data1f = []
            if 'data_file_1f' in kwargs:
                try:
                    data1f = ast.literal_eval(kwargs['data_file_1f'])
                except ValueError:
                    data1f = kwargs['data_file_1f']

            if isinstance(data2f, str):
                # make into a list
                data2f = [data2f]

            if isinstance(data1f, str):
                # make into a list
                data1f = [data1f]

            for freq, data in zip([1., 2.], [data1f, data2f]):
                self.datakwargs['freqfactor'] = freq

                if isinstance(data, dict):
                    # make into a list
                    if detectors is None:
                        detectors = list(data.keys())
                    else:
                        for det in data.keys():
                            if det not in detectors:
                                data.pop(det)

                    data = list(data.values())

                if isinstance(data, list):
                    # pass through list and check strings
                    for dfile in data:
                        detdata = dfile.split(':')  # split detector and path
                        if detectors is None:
                            if len(detdata) == 2:
                                self.hetdata.add_data(
                                    HeterodynedData(
                                        data=detdata[1],
                                        detector=detdata[0],
                                        **self.datakwargs
                                    )
                                )
                            else:
                                raise ValueError("Data string must be of the form "
                                                 "'DET:FILEPATH'")
                        else:
                            if len(detdata) == 2:
                                if detdata[0] not in detectors:
                                    raise ValueError("Data file does not have "
                                                     "consistent detector")
                                self.hetdata.add_data(
                                    HeterodynedData(
                                        data=detdata[1],
                                        detector=detdata[0],
                                        **self.datakwargs
                                    )
                                )
                            elif len(detdata) == 1 and len(detectors) == 1:
                                self.hetdata.add_data(
                                    data=dfile,
                                    detector=detectors[0],
                                    **self.datakwargs
                                )
                            else:
                                raise ValueError("Data string must be of the form "
                                                 "'DET:FILEPATH'")
                else:
                    raise TypeError("Data files are not of a recognised type")

                # remove any detectors than are not requested
                if detectors is not None:
                    for det in list(self.hetdata.detectors):
                        if det not in detectors:
                            self.hetdata.pop(det)
            else:
                raise KeyError("Data files must be given")

        # set fake data
        detectors = None if resetdetectors else detectors
        if (('fake_asd' in kwargs
                or 'fake_asd_1f' in kwargs
                or 'fake_asd_2f' in kwargs
                or 'fake_sigma' in kwargs
                or 'fake_sigma_1f' in kwargs
                or 'fake_sigma_2d' in kwargs)):
            fakeasd2f = []
            issigma2f = False
            for kw in ['fake_asd', 'fake_asd_2f', 'fake_sigma',
                       'fake_sigma_2f']:
                if kw in kwargs:
                    try:
                        fakeasd2f = ast.literal_eval(kwargs[kw])
                    except ValueError:
                        fakeasd2f = kwargs[kw]
                    if 'sigma' in kw:
                        issigma2f = True
                        break

            fakeasd1f = []
            issigma1f = False
            for kw in ['fake_asd_1f', 'fake_sigma_1f']:
                if kw in kwargs:
                    try:
                        fakeasd1f = ast.literal_eval(kwargs[kw])
                    except ValueError:
                        fakeasd1f = kwargs[kw]
                    if 'sigma' in kw:
                        issigma1f = True
                        break           

            if isinstance(fakeasd2f, (str, float)):
                # make into a list
                if detectors is None:
                    fakeasd2f = [fakeasd2f]
                else:
                    fakeasd2f = len(detectors) * [fakeasd2f]

            if isinstance(fakeasd1f, (str, float)):
                # make into a list
                if detectors is None:
                    fakeasd1f = [fakeasd1f]
                else:
                    fakeasd1f = len(detectors) * [fakeasd1f]

            for freq, fakedata, issigma in zip([1., 2.],
                                               [fakeasd1f, fakeasd2f],
                                               [issigma1f, issigma2f]):
                self.datakwargs['freqfactor'] = freq
                self.datakwargs['issigma'] = issigma

                if isinstance(fakedata, dict):
                    # make into a list
                    if detectors is None:
                        detectors = list(fakedata.keys())
                    else:
                        for det in fakedata.keys():
                            if det not in detectors:
                                fakedata.pop(det)

                    fakedata = list(fakedata.values())

                if isinstance(fakedata, list):
                    # parse through list
                    for fdata in fakedata:
                        detfdata = fdata.split(':')
                        if detectors is None:
                            if len(detfdata) == 2:
                                try:
                                    asdval = float(detdata[1])
                                except ValueError:
                                    asdval = detdata[1]

                                # check if actual data already exists
                                if detdata[0] in self.hetdata.detectors:
                                    for het in self.hetdata[detdata[0]]:
                                        if het.freq_factor == freq:
                                            # data already exists
                                            continue

                                self.hetdata.add_data(
                                    HeterodynedData(
                                        fakeasd=asdval,
                                        detector=detdata[0],
                                        **self.datakwargs
                                    )
                                )
                            else:
                                raise ValueError("Fake data string must be of "
                                                 "the form 'DET:ASD'")
                        else:
                            if len(detfdata) == 2:
                                if detfdata[0] not in detectors:
                                    raise ValueError("Fake data input does not have "
                                                     "consistent detector")

                                try:
                                    asdval = float(detdata[1])
                                except ValueError:
                                    asdval = detdata[1]

                                # check if actual data already exists
                                if detdata[0] in self.hetdata.detectors:
                                    for het in self.hetdata[detdata[0]]:
                                        if het.freq_factor == freq:
                                            # data already exists
                                            continue

                                self.hetdata.add_data(
                                    HeterodynedData(
                                        fakeasd=asdval,
                                        detector=detdata[0],
                                        **self.datakwargs
                                    )
                                )
                            elif len(detfdata) == 1 and len(detectors) == 1:
                                try:
                                    asdval = float(detfdata)
                                except ValueError:
                                    asdval = detfdata

                                self.hetdata.add_data(
                                    fakeasd=asdval,
                                    detector=detectors[0],
                                    **self.datakwargs
                                )
                            else:
                                raise ValueError("Fake data string must be of the form "
                                                 "'DET:FILEPATH'")
                else:
                    raise TypeError("Fake data not of the correct type")

        if len(self.hetdata) == 0:
            raise ValueError("No data has been supplied!")

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

        # output parameters
        if 'outdir' in kwargs:
            self.sampler_kwargs.setdefault('outdir', kwargs.get('outdir'))
        if 'label' in kwargs:
            self.sampler_kwargs.setdefault('label', kwargs.get('label'))

        # default restart time to 1000000 seconds if not running through CLI
        self.periodic_restart_time = kwargs.get('periodic_restart_time',
                                                10000000)

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
            likelihood=self.likelihood,
            **self.sampler_kwargs
        )

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
    inj_par: str
        The path to a TEMPO(2) style pulsar parameter file containing the
        parameters of a simulated signal to be injected into the data. If
        this is not given a simulated signal will not be added.
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
        than passed. This data is assumed to have been heterodyned at twice the
        source's rotation frequency - to explicitly add data heterodyned at
        once or twice the source's rotation frequency use the ``data_file_1f``
        and ``data_file_2f` arguments.
    data_file_2f: str, list, dict
        Data files that have been heterodyned at twice the source's rotation
        frequency. See the documentation for ``data_file`` above for usage.
    data_file_1f: str, list, dict
        Data files that have been heterodyned at the source's rotation
        frequency. See the documentation for ``data_file`` above for usage.
    fake_asd: float, str, list, dict
        This specifies the creation of fake Gaussian data drawn from a
        distribution with a given noise amplitude spectral density (ASD). If
        passing a float (and set of ``detector``'s are specified), then this
        value is used when generating the noise for all detectors. If this is
        a string then it should give a detector alias, which specifies using
        the noise ASD for the design sensitivity curve of that detector, or it
        should give a path to a file containing a frequency series of the
        ASD. If a list, then this should be a different float or string for
        each supplied detector. If a dictionary, then this should be a set
        of floats or string keyed to detector names. The simulated noise
        produced with this argument is assumed to be at twice the source
        rotation frequency. To explicitly specify fake data generation at once
        or twice the source rotation frequency use the equivalent
        ``fake_asd_1f`` and ``fake_asd_2f`` arguments respectively. If wanting
        to supply noise standard deviations rather than ASD use the
        ``fake_sigma`` arguments instead. This argument is ignored if
        ``data_file``'s are supplied.
    fake_asd_1f: float, str, list, dict
        Set the amplitude spectral density for fake noise generation at the
        source rotation frequency. See the ``fake_asd`` argument for usage.
    fake_asd_2f: float, str, list, dict
        Set the amplitude spectral density for fake noise generation at twice
        the source rotation frequency. See the ``fake_asd`` argument for usage.
    fake_sigma: float, str, list, dict
        Set the standard deviation for generating fake Gaussian noise. See the
        ``fake_asd`` argument for usage. The simulated noise produced with this
        argument is assumed to be at twice the source rotation frequency. To
        explicitly specify fake data generation at once or twice the source
        rotation frequency use the equivalent ``fake_sigma_1f`` and
        ``fake_sigma_2f`` arguments respectively.
    fake_sigma_1f: float, str, list, dict
        Set the noise standard deviation for fake noise generation at the
        source rotation frequency. See the ``fake_sigma`` argument for usage.
    fake_sigma_2f: float, str, list, dict
        Set the noise standard deviation for fake noise generation at twice
        the source rotation frequency. See the ``fake_sigma`` argument for usage.
    data_kwargs: dict
        A dictionary of keyword arguments to pass to the
        :class:`cwinpy.data.HeterodynedData` objects.
    inj_times: list
        A list of pairs of times between which the simulated signal (if given
        with the ``inj_par`` argument) will be added to the data.
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
