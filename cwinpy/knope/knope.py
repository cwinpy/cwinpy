"""
Run known pulsar parameter estimation using bilby.
"""

import os
import sys
import ast
import glob
import signal
import warnings
import numpy as np

import cwinpy
from ..data import HeterodynedData, MultiHeterodynedData
from ..likelihood import TargetedPulsarLikelihood

import bilby
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import parse_args, BilbyPipeError


def sighandler(signum, frame):
    # perform periodic eviction with exit code 130
    # see https://git.ligo.org/lscsoft/bilby_pipe/blob/0b5ca550e3a92494ef3e04801e79a2f9cd902b44/bilby_pipe/parser.py#L270
    sys.exit(130)


def create_knope_parser():
    """
    Create the argument parser.
    """

    description = """\
A script to use Bayesian inference to estimate the parameters of a \
continuous gravitational-wave signal from a known pulsar."""

    parser = BilbyArgParser(
        prog=sys.argv[0],
        description=description,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
    )
    parser.add("--config", type=str, is_config_file=True, help="Configuration ini file")
    parser.add(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=cwinpy.__version__),
    )
    parser.add(
        "--periodic-restart-time",
        default=10800,
        type=int,
        help=(
            "Time after which the job will be self-evicted with code 130. "
            "After this, condor will restart the job. Default is 10800s. "
            "This is used to decrease the chance of HTCondor hard evictions."
        ),
    )

    pulsarparser = parser.add_argument_group("Pulsar inputs")
    pulsarparser.add(
        "--par-file",
        required=True,
        type=str,
        help=("The path to a TEMPO(2) style file containing the pulsar parameters."),
    )

    dataparser = parser.add_argument_group("Data inputs")
    dataparser.add(
        "-d",
        "--detector",
        action="append",
        help=(
            "The abbreviated name of a detector to analyse. "
            "Multiple detectors can be passed with multiple "
            "arguments, e.g., --detector H1 --detector L1."
        ),
    )
    dataparser.add(
        "--data-file",
        action="append",
        help=(
            "The path to a heterodyned data file for a given "
            "detector. The format should be of the form "
            '"DET:PATH",  where DET is the detector name. '
            "Multiple files can be passed with multiple "
            "arguments, e.g., --data-file H1:H1data.txt "
            "--data-file L1:L1data.txt. This data will be assumed "
            "to be that in a search for a signal from the l=m=2 "
            "mass quadrupole and therefore heterodyned at twice "
            "the source's rotation frequency. To add data "
            "explicitly setting the heterodyned frequency at twice "
            'the rotation frequency use "--data-file-2f", or for '
            'data at the rotation frequency use "--data-file-1f".'
        ),
    )
    dataparser.add(
        "--data-file-2f",
        action="append",
        help=(
            "The path to a data file for a given detector where "
            "the data is explicitly given as being heterodyned at "
            "twice the source's rotation frequency. The inputs "
            "should be in the same format as those given to the "
            '"--data-file" flag. This flag should generally be '
            'preferred over the use of "--data-file".'
        ),
    )
    dataparser.add(
        "--data-file-1f",
        action="append",
        help=(
            "The path to a data file for a given detector where "
            "the data is explicitly given as being heterodyned at "
            "the source's rotation frequency. The inputs should "
            "be in the same format as those given to the "
            '"--data-file" flag.'
        ),
    )
    dataparser.add(
        "--data-kwargs",
        help=(
            "A Python dictionary containing keywords to pass to "
            "the HeterodynedData object."
        ),
    )

    simparser = parser.add_argument_group("Simulated data")
    simparser.add(
        "--inj-par",
        type=str,
        help=(
            "The path to a TEMPO(2) style file containing the "
            'parameters of a simulated signal to "inject" into the '
            "data."
        ),
    )
    simparser.add(
        "--inj-times",
        help=(
            "A Python list of pairs of times between which to add "
            'the simulated signal (specified by the "--inj-par" '
            "flag) to the data. By default the signal is added into "
            "the whole data set."
        ),
    )
    simparser.add(
        "--show-truths",
        action="store_true",
        default=False,
        help=(
            "If plotting the results, setting this flag will "
            'overplot the "true" signal values. If adding a '
            "simulated signal then these parameter values will be "
            'taken from the file specified by the "--inj-par" flag, '
            "otherwise the values will be taken from the file "
            'specified by the "--par-file" flag.'
        ),
    )
    simparser.add(
        "--fake-asd",
        action="append",
        help=(
            "This flag sets the code to perform the analysis on "
            "simulated Gaussian noise, with data samples drawn from "
            "a Gaussian distribution defined by a given amplitude "
            "spectral density. The flag is set in a similar way to "
            'the "--data-file" flag. The argument can either be a '
            "float giving an ASD value, or a string containing a "
            "detector alias to produce noise from the design curve "
            "for that detector, or a string containing a path to a "
            "file with the noise curve for a detector. This can be "
            'used in conjunction with the "--detector" flag, e.g., '
            '"--detector H1 --fake-asd 1e-23", or without the '
            '"--detector" flag, e.g., "--fake-asd H1:1e-23". Values '
            "for multiple detectors can be passed by repeated use "
            "of the flag, noting that if used in conjunction with "
            'the "--detector" flag detectors and ASD values should '
            'be added in the same order, e.g., "--detector H1 '
            '--fake-asd H1 --detector L1 --fake-asd L1". This flag '
            'is ignored if "--data-file" values for the same '
            "detector have already been passed. The fake data that "
            "is produced is assumed to be that for a signal at "
            "twice the source rotation frequency. To explicitly set "
            "fake data at once or twice the rotation frequency "
            'use the "--fake-asd-1f" and "--fake-asd-2f" flags '
            "instead."
        ),
    )
    simparser.add(
        "--fake-asd-1f",
        action="append",
        help=(
            "This flag sets the data to be Gaussian noise "
            "explicitly for a source emitting at the rotation "
            'frequency. See the documentation for "--fake-asd" for '
            "details of its use."
        ),
    )
    simparser.add(
        "--fake-asd-2f",
        action="append",
        help=(
            "This flag sets the data to be Gaussian noise "
            "explicitly for a source emitting at twice the rotation "
            'frequency. See the documentation for "--fake-asd" for '
            "details of its use."
        ),
    )
    simparser.add(
        "--fake-sigma",
        action="append",
        help=(
            'This flag is equivalent to "--fake-asd", but '
            "instead of taking in an amplitude spectral density "
            "value it takes in a noise standard deviation."
        ),
    )
    simparser.add(
        "--fake-sigma-1f",
        action="append",
        help=(
            'This flag is equivalent to "--fake-asd-1f", but '
            "instead of taking in an amplitude spectral density "
            "value it takes in a noise standard deviation."
        ),
    )
    simparser.add(
        "--fake-sigma-2f",
        action="append",
        help=(
            'This flag is equivalent to "--fake-asd-2f", but '
            "instead of taking in an amplitude spectral density "
            "value it takes in a noise standard deviation."
        ),
    )
    simparser.add(
        "--fake-start",
        action="append",
        help=(
            "The GPS start time for generating simulated noise "
            "data. This be added for each detector in the same way "
            'as used in the "--fake-asd" command (default: '
            "1000000000)."
        ),
    )
    simparser.add(
        "--fake-end",
        action="append",
        help=(
            "The GPS end time for generating simulated noise data. "
            "This be added for each detector in the same way as "
            'used in the "--fake-asd" command (default: '
            "1000086400)"
        ),
    )
    simparser.add(
        "--fake-dt",
        action="append",
        help=(
            "The time step for generating simulated noise data. "
            "This be added for each detector in the same way as "
            'used in the "--fake-asd" command (default: 60)'
        ),
    )
    simparser.add(
        "--fake-seed",
        type=int,
        help=(
            "A positive integer random number generator seed used "
            "when generating the simulated data noise."
        ),
    )

    outputparser = parser.add_argument_group("Output")
    outputparser.add("-o", "--outdir", help="The output directory for the results")
    outputparser.add("-l", "--label", help="The output filename label for the results")

    samplerparser = parser.add_argument_group("Sampler inputs")
    samplerparser.add(
        "-s",
        "--sampler",
        default="dynesty",
        help=("The sampling algorithm to use bilby (default: " "%(default)s)"),
    )
    samplerparser.add(
        "--sampler-kwargs",
        help=(
            "The keyword arguments for running the sampler. This should be in "
            "the format of a standard Python dictionary and must be given "
            "within quotation marks, e.g., \"{'Nlive':1000}\"."
        ),
    )
    samplerparser.add(
        "--likelihood",
        default="studentst",
        help=(
            "The name of the likelihood function to use. This can be either "
            '"studentst" or "gaussian".'
        ),
    )
    samplerparser.add(
        "--prior",
        type=str,
        required=True,
        help=(
            "The path to a bilby-style prior file defining the parameters to "
            "be estimated and their prior probability distributions."
        ),
    )
    samplerparser.add(
        "--grid",
        action="store_true",
        default=False,
        help=(
            "Set this flag to evaluate the posterior over a grid rather than "
            "using a stochastic sampling method."
        ),
    )
    samplerparser.add(
        "--grid-kwargs",
        help=(
            "The keyword arguments for running the posterior evaluation over "
            "a grid. This should be a the format of a standard Python "
            "dictionary, and must be given within quotation marks, "
            "e.g., \"{'grid_size':100}\"."
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

        # remove any None keyword values
        for key in list(kwargs.keys()):
            if kwargs[key] is None:
                kwargs.pop(key)

        # keyword arguments for creating the HeterodynedData objects
        self.datakwargs = kwargs.get("data_kwargs", {})

        if "par_file" not in kwargs:
            raise KeyError("A pulsar parameter file must be provided")
        else:
            self.datakwargs["par"] = kwargs["par_file"]

        # injection parameters
        self.datakwargs.setdefault("injpar", kwargs.get("inj_par", None))
        self.datakwargs.setdefault(
            "inject", (False if self.datakwargs["injpar"] is None else True)
        )

        # get list of times at which to inject the signal
        self.datakwargs.setdefault("injtimes", kwargs.get("inj_times", None))
        try:
            self.datakwargs["injtimes"] = ast.literal_eval(self.datakwargs["injtimes"])
        except (ValueError, SyntaxError):
            pass

        if self.datakwargs["injtimes"] is not None:
            if not isinstance(self.datakwargs["injtimes"], (list, np.ndarray)):
                raise TypeError("Injection times must be a list")

        # data parameters
        if "detector" in kwargs:
            if isinstance(kwargs["detector"], str):
                detectors = [kwargs["detector"]]
            elif isinstance(kwargs["detector"], list):
                detectors = []
                for det in kwargs["detector"]:
                    try:
                        # remove additional quotation marks from string
                        thisdet = ast.literal_eval(det)
                    except (ValueError, SyntaxError):
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
        if (
            "data_file" in kwargs
            or "data_file_1f" in kwargs
            or "data_file_2f" in kwargs
        ):
            data2f = []
            for kw in ["data_file", "data_file_2f"]:
                if kw in kwargs:
                    try:
                        data2f = ast.literal_eval(kwargs[kw])
                    except (ValueError, SyntaxError):
                        data2f = kwargs[kw]
                    break

            data1f = []
            if "data_file_1f" in kwargs:
                try:
                    data1f = ast.literal_eval(kwargs["data_file_1f"])
                except (ValueError, SyntaxError):
                    data1f = kwargs["data_file_1f"]

            if isinstance(data2f, str):
                # make into a list
                data2f = [data2f]

            if isinstance(data1f, str):
                # make into a list
                data1f = [data1f]

            for freq, data in zip([1.0, 2.0], [data1f, data2f]):
                self.datakwargs["freqfactor"] = freq

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
                    for i, dfile in enumerate(data):
                        detdata = dfile.split(":")  # split detector and path

                        if len(detdata) == 2:
                            if detectors is not None:
                                if detdata[0] not in detectors:
                                    raise ValueError(
                                        "Data file does not have consistent detector"
                                    )
                            thisdet = detdata[0]
                            thisdata = detdata[1]
                        elif len(detdata) == 1 and detectors is not None:
                            try:
                                thisdet = detectors[i]
                            except Exception as e:
                                raise ValueError(
                                    "Detectors is not a list: {}".format(e)
                                )
                            thisdata = dfile
                        else:
                            raise ValueError(
                                "Data string must be of the form 'DET:FILEPATH'"
                            )

                        self.hetdata.add_data(
                            HeterodynedData(
                                data=thisdata, detector=thisdet, **self.datakwargs
                            )
                        )
                else:
                    raise TypeError("Data files are not of a recognised type")

                # remove any detectors than are not requested
                if detectors is not None:
                    for det in list(self.hetdata.detectors):
                        if det not in detectors:
                            self.hetdata.pop(det)

        # set fake data
        detectors = None if resetdetectors else detectors
        if (
            "fake_asd" in kwargs
            or "fake_asd_1f" in kwargs
            or "fake_asd_2f" in kwargs
            or "fake_sigma" in kwargs
            or "fake_sigma_1f" in kwargs
            or "fake_sigma_2d" in kwargs
        ):

            starts = kwargs.get("fake_start", 1000000000)  # default start time
            ends = kwargs.get("fake_end", 1000086400)  # default end time
            dts = kwargs.get("fake_dt", 60)  # default time step
            ftimes = kwargs.get("fake_times", None)  # time array(s)
            fseed = kwargs.get("fake_seed", None)  # data seed

            fakeasd2f = []
            issigma2f = False
            for kw in ["fake_asd", "fake_asd_2f", "fake_sigma", "fake_sigma_2f"]:
                if kw in kwargs:
                    try:
                        fakeasd2f = ast.literal_eval(kwargs[kw])
                    except (ValueError, SyntaxError):
                        fakeasd2f = kwargs[kw]
                    if "sigma" in kw:
                        issigma2f = True
                        break

            fakeasd1f = []
            issigma1f = False
            for kw in ["fake_asd_1f", "fake_sigma_1f"]:
                if kw in kwargs:
                    try:
                        fakeasd1f = ast.literal_eval(kwargs[kw])
                    except (ValueError, SyntaxError):
                        fakeasd1f = kwargs[kw]
                    if "sigma" in kw:
                        issigma1f = True
                        break

            if isinstance(starts, str):
                try:
                    starts = ast.literal_eval(starts)
                except (ValueError, SyntaxError):
                    pass

            if isinstance(ends, str):
                try:
                    ends = ast.literal_eval(ends)
                except (ValueError, SyntaxError):
                    pass

            if isinstance(dts, str):
                try:
                    dts = ast.literal_eval(dts)
                except (ValueError, SyntaxError):
                    pass

            if isinstance(starts, int):
                if detectors is None:
                    starts = [starts]
                else:
                    starts = len(detectors) * [starts]

            if isinstance(ends, int):
                if detectors is None:
                    ends = [ends]
                else:
                    ends = len(detectors) * [ends]

            if isinstance(dts, int):
                if detectors is None:
                    dts = [dts]
                else:
                    dts = len(detectors) * [dts]

            if isinstance(ftimes, (np.ndarray, list)) or ftimes is None:
                if len(np.shape(ftimes)) == 1 or ftimes is None:
                    if detectors is None:
                        ftimes = [ftimes]
                    else:
                        ftimes = len(detectors) * [ftimes]

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

            # set random seed
            rstate = None
            if fseed is not None:
                rstate = np.random.RandomState(fseed)

            for freq, fakedata, issigma in zip(
                [1.0, 2.0], [fakeasd1f, fakeasd2f], [issigma1f, issigma2f]
            ):
                self.datakwargs["freqfactor"] = freq
                self.datakwargs["issigma"] = issigma
                self.datakwargs["fakeseed"] = rstate

                if isinstance(fakedata, dict):
                    # make into a list
                    if detectors is None:
                        detectors = list(fakedata.keys())
                    else:
                        for det in fakedata.keys():
                            if det not in detectors:
                                fakedata.pop(det)

                    fakedata = list(fakedata.values())

                if isinstance(starts, dict):
                    # make into a list
                    if list(starts.keys()) != detectors:
                        raise ValueError(
                            "Fake data start times do not "
                            "contain consistent detectors"
                        )
                    else:
                        starts = list(starts.values())

                if isinstance(ends, dict):
                    # make into a list
                    if list(ends.keys()) != detectors:
                        raise ValueError(
                            "Fake data end times do not contain consistent detectors"
                        )
                    else:
                        ends = list(ends.values())

                if isinstance(dts, dict):
                    # make into a list
                    if list(dts.keys()) != detectors:
                        raise ValueError(
                            "Fake data time steps do not "
                            "contain consistent detectors"
                        )
                    else:
                        dts = list(dts.values())

                if isinstance(ftimes, dict):
                    # make into a list
                    if list(ftimes.keys()) != detectors:
                        raise ValueError(
                            "Fake data time arrays do not "
                            "contain consistent detectors"
                        )
                    else:
                        ftimes = list(ftimes.values())

                if len(fakedata) > 0:
                    if len(starts) == 1 and len(fakedata) > 1:
                        starts = len(fakedata) * starts

                    if len(ends) == 1 and len(fakedata) > 1:
                        ends = len(fakedata) * ends

                    if len(dts) == 1 and len(fakedata) > 1:
                        dts = len(fakedata) * dts

                    if len(ftimes) == 1 and len(fakedata) > 1:
                        ftimes = len(fakedata) * ftimes

                    if (
                        len(fakedata) != len(starts)
                        or len(fakedata) != len(ends)
                        or len(fakedata) != len(dts)
                        or len(fakedata) != len(ftimes)
                    ):
                        raise ValueError(
                            "Fake data values and times are not consistent"
                        )

                if isinstance(fakedata, list):
                    # parse through list
                    detidx = 0
                    for fdata, start, end, dt, ftime in zip(
                        fakedata, starts, ends, dts, ftimes
                    ):
                        try:
                            # make sure value is a string so it can be "split"
                            detfdata = str(fdata).split(":")
                        except Exception as e:
                            raise TypeError(
                                "Fake time value is the wrong type: {}".format(e)
                            )

                        if ftime is None:
                            try:
                                # make sure values are strings so they can be
                                # "split"
                                detstart = str(start).split(":")
                                detend = str(end).split(":")
                                detdt = str(dt).split(":")
                            except Exception as e:
                                raise TypeError(
                                    "Fake time value is the wrong type: {}".format(e)
                                )

                        if len(detfdata) == 2:
                            if detectors is not None:
                                if detfdata[0] not in detectors:
                                    raise ValueError(
                                        "Fake data input does not have "
                                        "consistent detector"
                                    )

                            try:
                                asdval = float(detfdata[1])
                            except ValueError:
                                asdval = detfdata[1]

                            thisdet = detfdata[0]

                            # check if actual data already exists
                            if thisdet in self.hetdata.detectors:
                                for het in self.hetdata[thisdet]:
                                    if het.freq_factor == freq:
                                        # data already exists
                                        continue

                            if ftime is None:
                                for detcheck in [detstart, detend, detdt]:
                                    if len(detcheck) == 2:
                                        if detcheck[0] != thisdet:
                                            raise ValueError("Inconsistent detectors!")

                                try:
                                    int(detcheck[-1])
                                except ValueError:
                                    raise TypeError("Problematic type!")

                                ftime = np.arange(
                                    int(detstart[-1]), int(detend[-1]), int(detdt[-1])
                                )
                        elif len(detfdata) == 1 and detectors is not None:
                            try:
                                asdval = float(detfdata[-1])
                            except ValueError:
                                asdval = detfdata[-1]

                            try:
                                thisdet = detectors[detidx]
                                detidx += 1
                            except Exception as e:
                                raise ValueError(
                                    "Detectors is not a list: {}".format(e)
                                )

                            # check if actual data already exists
                            if thisdet in self.hetdata.detectors:
                                for het in self.hetdata[thisdet]:
                                    if het.freq_factor == freq:
                                        # data already exists
                                        continue

                            if ftime is None:
                                for detcheck in [detstart, detend, detdt]:
                                    if len(detcheck) == 2:
                                        if detcheck[0] != thisdet:
                                            raise ValueError("Inconsistent detectors!")

                                try:
                                    int(detcheck[-1])
                                except ValueError:
                                    raise TypeError("Problematic type!")

                                ftime = np.arange(
                                    int(detstart[-1]), int(detend[-1]), int(detdt[-1])
                                )
                        else:
                            raise ValueError(
                                "Fake data string must be of the form 'DET:ASD'"
                            )

                        self.hetdata.add_data(
                            HeterodynedData(
                                fakeasd=asdval,
                                detector=thisdet,
                                times=ftime,
                                **self.datakwargs
                            )
                        )
                else:
                    raise TypeError("Fake data not of the correct type")

        if len(self.hetdata) == 0:
            raise ValueError("No data has been supplied!")

        # sampler parameters
        self.sampler = kwargs.get("sampler", "dynesty")
        self.sampler_kwargs = kwargs.get("sampler_kwargs", {})
        if isinstance(self.sampler_kwargs, str):
            try:
                self.sampler_kwargs = ast.literal_eval(self.sampler_kwargs)
            except (ValueError, SyntaxError):
                raise ValueError("Unable to parse sampler keyword arguments")
        self.use_grid = kwargs.get("grid", False)
        self.grid_kwargs = kwargs.get("grid_kwargs", {})
        if isinstance(self.grid_kwargs, str):
            try:
                self.grid_kwargs = ast.literal_eval(self.grid_kwargs)
            except (ValueError, SyntaxError):
                raise ValueError("Unable to parse grid keyword arguments")
        self.likelihoodtype = kwargs.get("likelihood", "studentst")
        self.prior = kwargs.get("prior", None)
        if not isinstance(self.prior, (str, dict, bilby.core.prior.PriorDict)):
            raise ValueError("The prior is not defined")
        else:
            self.prior = bilby.core.prior.PriorDict(self.prior)

        # output parameters
        if "outdir" in kwargs:
            self.sampler_kwargs.setdefault("outdir", kwargs.get("outdir"))
        if "label" in kwargs:
            self.sampler_kwargs.setdefault("label", kwargs.get("label"))

        show_truths = kwargs.get("show_truths", False)
        if show_truths:
            # don't overwrite 'injection_parameters' is they are already defined
            if "injection_parameters" not in self.sampler_kwargs:
                if self.datakwargs["injpar"] is not None:
                    injpartmp = self.hetdata.to_list[0].injpar
                else:
                    injpartmp = self.hetdata.to_list[0].par

                # get "true" values of any parameters in the prior
                injtruths = {}
                for key in self.prior:
                    injtruths[key] = injpartmp[key.upper()]

                    # check iota and theta
                    if key.lower() in ["iota", "theta"]:
                        if (
                            injtruths[key] is None
                            and injpartmp["COS{}".format(key.upper())] is not None
                        ):
                            injtruths[key] = np.arccos(
                                injpartmp["COS{}".format(key.upper())]
                            )
                    elif key.lower() in ["cosiota", "costheta"]:
                        if (
                            injtruths[key] is None
                            and injpartmp[key[3:].upper()] is not None
                        ):
                            injtruths[key] = np.cos(injpartmp[key[3:].upper()])

                self.sampler_kwargs.update({"injection_parameters": injtruths})

        # set use_ratio to False by default
        if "use_ratio" not in self.sampler_kwargs:
            self.sampler_kwargs["use_ratio"] = False

        # default restart time to 1000000 seconds if not running through CLI
        self.periodic_restart_time = kwargs.get("periodic_restart_time", 10000000)

    def set_likelihood(self):
        """
        Set the likelihood function.
        """

        self.likelihood = TargetedPulsarLikelihood(
            data=self.hetdata, priors=self.prior, likelihood=self.likelihoodtype
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
            priors=self.prior,
            likelihood=self.likelihood,
            **self.sampler_kwargs
        )

        return self.result

    def run_grid(self):
        """
        Run the sampling over a grid in parameter space.

        Returns
        -------
        grid
            A bilby Grid object.
        """

        self.grid = bilby.core.grid.Grid(
            likelihood=self.likelihood, priors=self.prior, **self.grid_kwargs
        )

        return self.grid


def knope(**kwargs):
    """
    Run knope within Python.

    Parameters
    ----------
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
        and ``data_file_2f`` arguments.
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
    fake_start: int, list, dict
        The GPS start time for generating fake data. If requiring data at once
        and twice the rotation frequency for the same detector, then the same
        start time will be used for both frequencies.
    fake_end: int, list, dict
        The GPS end time for generating fake data. If requiring data at once
        and twice the rotation frequency for the same detector, then the same
        end time will be used for both frequencies.
    fake_dt: int, list, dict:
        The time step in seconds for generating fake data. If requiring data at
        once and twice the rotation frequency for the same detector, then the
        same time step will be used for both frequencies.
    fake_times: dict, array_like
        Instead of passing start times, end times and time steps for the fake
        data generation, an array of GPS times (or a dictionary of arrays keyed
        to the detector) can be passed instead.
    fake_seed: int, :class:`numpy.random.RandomState`
        A seed for random number generation for the creation of fake data.
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
    grid: bool
        If True then the posterior will be evaluated on a grid.
    grid_kwargs: dict
        A dictionary of keyword arguments to be used by the grid sampler.
    outdir: str
        The output directory for the results.
    label: str
        The name of the output file (excluding the '.json' extension) for the
        results.
    likelihood: str
        The likelihood function to use. At the moment this can be either
        'studentst' or 'gaussian', with 'studentst' being the default.
    show_truths: bool
        If plotting the results, setting this argument will overplot the
        "true" signal values. If adding a simulated signal then these parameter
        values will be taken from the file specified by the "inj_par" argument,
        otherwise the values will be taken from the file specified by the
        "par_file" argument.
    prior: str, PriorDict
        A string to a bilby-style
        `prior <https://lscsoft.docs.ligo.org/bilby/prior.html>`_ file, or a
        bilby :class:`~bilby.core.prior.PriorDict` object. This defines the
        parameters that are to be estimated and their prior distributions.
    config: str
        The path to a configuration file containing the analysis arguments.
    periodic_restart_time: int
        The number of seconds after which the run will be evicted with a
        ``130`` exit code. This prevents hard evictions if running under
        HTCondor. For running via the command line interface, this defaults to
        10800 seconds (3 hours), at which point the job will be stopped (and
        then restarted if running under HTCondor). If running directly within
        Python this defaults to 10000000. 
    """

    if "cwinpy_knope" == os.path.split(sys.argv[0])[-1] or "config" in kwargs:
        # get command line arguments
        parser = create_knope_parser()

        # parse config file or command line arguments
        if "config" in kwargs:
            cliargs = ["--config", kwargs["config"]]
        else:
            cliargs = sys.argv[1:]

        try:
            args, unknown_args = parse_args(cliargs, parser)
        except BilbyPipeError as e:
            raise IOError("{}".format(e))

        # convert args to a dictionary
        dargs = vars(args)

        if "config" in kwargs:
            # update with other keyword arguments
            dargs.update(kwargs)
    else:
        dargs = kwargs

    # set up the run
    runner = KnopeRunner(dargs)

    # run the sampler (expect in testing)
    if runner.use_grid:
        runner.run_grid()
    elif not hasattr(cwinpy, "_called_from_test"):
        runner.run_sampler()

    return runner


def knope_cli(**kwargs):
    """
    Entry point to ``cwinpy_knope script``. This just calls :func:`cwinpy.knope.knope`,
    but does not return any objects.
    """

    _ = knope(**kwargs)


class KnopeDAGRunner(object):
    """
    Set up and run the known pulsar parameter estimation DAG.

    Parameters
    ----------
    config: :class:`configparser.ConfigParser`
          A :class:`configparser.ConfigParser` object with the analysis setup
          parameters.
    """

    def __init__(self, config):
        # create and build the dag
        self.create_dag(config)

        if self.submitdag:
            self.dag.submit_dag(self.submit_options)

    def create_dag(self, config):
        """
        Create the HTCondor DAG from the configuration parameters.

        Parameters
        ----------
        config: :class:`configparser.ConfigParser`
            A :class:`configparser.ConfigParser` object with the analysis setup
            parameters.
        """

        import configparser

        if not isinstance(config, configparser.ConfigParser):
            raise TypeError("'config' must be a ConfigParser object")

        try:
            from pycondor import Job, Dagman
        except ImportError:
            raise ImportError("To run 'cwinpy_knope_dag' you must install pycondor")

        # get DAG arguments
        if config.has_section("dag"):
            # submit directory location
            submit = config.get(
                "dag", "submit", fallback=os.path.join(os.getcwd(), "submit")
            )

            # DAG name prefix
            name = config.get("dag", "name", fallback="cwinpy_knope")

            # get whether to automatically submit the dag
            self.submitdag = config.getboolean("dag", "submitdag", fallback=False)

            # get any additional submission options
            self.submit_options = config.get("dag", "submit_options", fallback=None)
        else:
            raise IOError("Configuration file must have a [condor] section.")

        # create the Dagman
        self.dag = Dagman(name=name, submit=submit)

        # get output directory base from the [run] section
        basedir = config.get("run", "basedir", fallback=None)
        if basedir is None:
            raise IOError("No output directory set.")
        else:
            try:
                os.makedirs(basedir, exist_ok=True)
            except Exception as e:
                raise IOError(
                    "Could not create output directory " "location: '{}'".format(e)
                )

        # get the cwinpy_knope job arguments
        if config.has_section("job"):
            # executable
            from shutil import which

            jobexec = which(config.get("job", "executable", fallback="cwinpy_knope"))

            if jobexec is None:
                raise ValueError("cwinpy_knope executable is not specified")
            elif os.path.basename(jobexec) != "cwinpy_knope":
                raise ValueError(
                    "Executable '{}' is not 'cwinpy_knope'!".format(jobexec)
                )

            # job name prefix
            jobname = config.get("job", "name", fallback="cwinpy_knope")

            # condor universe
            universe = config.get("job", "universe", fallback="vanilla")

            # stdout directory location
            output = config.get("job", "out", fallback=os.path.join(basedir, "out"))

            # stderr directory location#
            error = config.get("job", "error", fallback=os.path.join(basedir, "error"))

            # log directory
            log = config.get("job", "log", fallback=os.path.join(basedir, "log"))

            # submit directory location
            jobsubmit = config.get(
                "job", "submit", fallback=os.path.join(basedir, "submit")
            )

            # get local environment variables
            getenv = config.getboolean("job", "getenv", fallback=False)

            # request memory
            reqmem = config.get("job", "request_memory", fallback="4 GB")

            # request CPUs
            reqcpus = config.getint("job", "request_cpus", fallback=1)

            # requirements
            requirements = config.get("job", "requirements", fallback=None)

            # retries of job on failure
            retry = config.getint("job", "retry", fallback=0)
        else:
            raise IOError("Configuration file must have a [job] section.")

        # create cwinpy_knope Job
        self.job = Job(
            jobname,
            jobexec,
            error=error,
            log=log,
            output=output,
            submit=jobsubmit,
            universe=universe,
            request_memory=reqmem,
            request_cpus=reqcpus,
            getenv=getenv,
            queue=1,
            requirements=requirements,
            retry=retry,
            dag=self.dag,
        )

        # create configurations for each cwinpy_knope job
        if config.has_section("knope"):
            # get the paths to the pulsar parameter files
            parfiles = config.get("knope", "pulsars", fallback=None)

            if parfiles is None:
                raise ValueError(
                    "Configuration must contain a set of pulsar parameter files"
                )

            # the "pulsars" option in the [knope] section can either be:
            #  - the path to a single file
            #  - a list of parameter files
            #  - a directory (or glob-able directory pattern) containing parameter files
            #  - a combination of a list of directories and/or files
            # All files must have the extension '.par'
            parfiles = ast.literal_eval(parfiles)
            if not isinstance(parfiles, list):
                parfiles = [parfiles]

            pulsars = []
            for parfile in parfiles:
                # add "*.par" wildcard to any directories
                if os.path.isdir(parfile):
                    parfile = os.path.join(parfile, "*.par")

                # get all parameter files
                pulsars.extend(
                    [
                        par
                        for par in glob.glob(parfile)
                        if os.path.splitext(par)[1] == ".par"
                    ]
                )

            # get names of all the pulsars
            from lalpulsar.PulsarParametersWrapper import PulsarParametersPy

            pulsarnames = []
            for pulsar in list(pulsars):
                if os.path.isfile(pulsar):
                    psr = PulsarParametersPy(pulsar)
                    if psr["PSRJ"] is not None:
                        pulsarnames.append(psr["PSRJ"])
                    else:
                        warnings.warn(
                            "Parameter file '{}' has no name, so it will be "
                            "ignored".format(pulsar)
                        )
                        pulsars.remove(pulsar)
                else:
                    # remove the pulsar
                    pulsars.remove(pulsar)

            # the "data-file-1f" and "data-file-2f" options in the [knope]
            # section specify the locations of heterodyned data files at the
            # rotation frequency and twice the rotation frequency of each
            # source. It is expected that the heterodyned file names
            # contain the name of the pulsar (based on the "PSRJ" name in the
            # associated parameter file). The option should be a dictionary
            # with keys being the detector name for the data sets.
            datafiles1f = config.get("knope", "data-file-1f", fallback=None)
            datafiles2f = config.get("knope", "data-file-2f", fallback=None)

            detectors1f = None
            if datafiles1f is not None:
                datafiles1f = ast.literal_eval(datafiles1f)

                if not isinstance(datafiles1f, dict):
                    raise TypeError("Data files must be specified in a dictionary")

                detectors1f = sorted(list(datafiles1f.keys()))

            detectors2f = None
            if datafiles2f is not None:
                datafiles2f = ast.literal_eval(datafiles2f)

                if not isinstance(datafiles2f, dict):
                    raise TypeError("Data files must be specified in a dictionary")

                detectors2f = sorted(list(datafiles2f.keys()))

            if datafiles1f is None and datafiles2f is None:
                raise IOError("No data files have been given")

            if detectors1f != detectors2f and datafiles1f and datafiles2f:
                raise IOError("Inconsistent detectors given for data sets")

            detectors = detectors2f if detectors2f else detectors1f

            # get lists of data files. For each detector the passed files could
            # be a single file, a list of files, or a glob-able directory path
            # containing the files.
            datadict = {pname: {} for pname in pulsarnames}

            for datafilesf, freqfactor in zip([datafiles1f, datafiles2f], ["1f", "2f"]):
                if datafilesf is None:
                    continue
                else:
                    for pname in pulsarnames:
                        datadict[pname][freqfactor] = {}

                dff = {det: [] for det in detectors}

                for det in detectors:
                    datafiles = datafilesf[det]
                    if not isinstance(datafiles, list):
                        datafiles = [datafiles]

                    for datafile in datafiles:
                        # add "*" wildcard to any directories
                        if os.path.isdir(datafile):
                            datafile = os.path.join(datafile, "*")

                        # get all data files
                        dff[det].extend(
                            [dat for dat in glob.glob(datafile) if os.path.isfile(dat)]
                        )

                    # check file name contains the name of a supplied pulsar
                    for datafile in list(dff[det]):
                        for pname in pulsarnames:
                            if pname in datafile:
                                if pname not in datadict:
                                    datadict[pname][freqfactor][det] = datafile
                                else:
                                    warnings.warn(
                                        "Duplicate pulsar '{}' data. Ignoring "
                                        "duplicate.".format(pname)
                                    )

            # set some default bilby-style priors
            DEFAULTPRIORS2F = (
                "h0 = Uniform(minimum=0.0, maximum=1.0e-22, name='h0', latex_label='$h_0$')\n"
                "phi0 = Uniform(minimum=0.0, maximum={pi}, name='phi0', latex_label='$\\phi_0$', unit='rad')\n"
                "iota = Sine(minimum=0.0, maximum={pi}, name='iota', latex_label='$\\iota$, unit='rad')\n"
                "psi = Uniform(minimum=0.0, maximum={pi_2}, name='psi', latex_label='$\\psi$, unit='rad')\n"
            ).format(**{"pi": np.pi, "pi_2": (np.pi / 2.0)})

            DEFAULTPRIORS1F = (
                "c21 = Uniform(minimum=0.0, maximum=1.0e-22, name='c21', latex_label='$C_{{21}}$')\n"
                "phi21 = Uniform(minimum=0.0, maximum={2pi}, name='phi21', latex_label='$\\Phi_{{21}}$', unit='rad')\n"
                "iota = Sine(minimum=0.0, maximum={pi}, name='iota', latex_label='$\\iota$, unit='rad')\n"
                "psi = Uniform(minimum=0.0, maximum={pi_2}, name='psi', latex_label='$\\psi$, unit='rad')\n"
            ).format(**{"2pi": 2.0 * np.pi, "pi": np.pi, "pi_2": (np.pi / 2.0)})

            DEFAULTPRIORS1F2F = (
                "c21 = Uniform(minimum=0.0, maximum=1.0e-22, name='c21', latex_label='$C_{{21}}$')\n"
                "c22 = Uniform(minimum=0.0, maximum=1.0e-22, name='c22', latex_label='$C_{{22}}$')\n"
                "phi21 = Uniform(minimum=0.0, maximum={2pi}, name='phi21', latex_label='$\\Phi_{{21}}$', unit='rad')\n"
                "phi22 = Uniform(minimum=0.0, maximum={2pi}, name='phi22', latex_label='$\\Phi_{{22}}$', unit='rad')\n"
                "iota = Sine(minimum=0.0, maximum={pi}, name='iota', latex_label='$\\iota$, unit='rad')\n"
                "psi = Uniform(minimum=0.0, maximum={pi_2}, name='psi', latex_label='$\\psi$, unit='rad')\n"
            ).format(**{"2pi": 2.0 * np.pi, "pi": np.pi, "pi_2": (np.pi / 2.0)})

            # get priors (if none are specified use the defaults)
            priors = config.get("knope", "priors", fallback=None)

            # "priors" can be a file, list of files, directory containing files,
            # glob-able path pattern to a set of files, or a dictionary of files
            # keyed to pulsar names. If all case bar a single file, or a
            # dictionary of keyed files, it is expected that the prior file name
            # contains the PSRJ name of the associated pulsar.
            if priors is not None:
                priors = ast.literal_eval(priors)

                if isinstance(priors, dict):
                    priorfiles = priors
                else:
                    if isinstance(priors, list):
                        allpriors = []
                        for priorfile in priors:
                            # add "*" wildcard to directories
                            if os.path.isdir(priorfile):
                                priorfile = os.path.join(priorfile, "*")

                            # get all prior files
                            allpriors.extend([pf for pf in glob.glob(priorfile)])

                        priorfiles = {}
                        for pname in pulsarnames:
                            for priorfile in allpriors:
                                if pname in priorfile:
                                    if pname not in priorfiles:
                                        priorfiles[pname] = priorfile
                                    else:
                                        warnings.warn(
                                            "Duplicate prior '{}' data. Ignoring "
                                            "duplicate.".format(pname)
                                        )
                    elif isinstance(priors, str):
                        if os.path.isfile(priors):
                            priorfiles = {psr: priorfile for psr in pulsarnames}
                        else:
                            raise ValueError(
                                "Prior file '{}' does not exist".format(priors)
                            )
                    else:
                        raise TypeError("Prior type is no recognised")
            else:
                # use default priors
                priorfile = os.path.join(basedir, "prior.txt")

                with open(priorfile, "w") as fp:
                    if datafiles1f is not None and datafiles2f is not None:
                        fp.write(DEFAULTPRIORS1F2F)
                    elif datafiles2f is not None:
                        fp.write(DEFAULTPRIORS2F)
                    else:
                        fp.write(DEFAULTPRIORS1F)

                priorfiles = {psr: priorfile for psr in pulsarnames}

            # check prior and data exist for the same pulsar, if not remove
            priornames = list(priorfiles.keys())
            datanames = list(datadict.keys())

            for pname in priornames + datanames:
                if pname in priornames and pname in datanames:
                    continue
                else:
                    warnings.warn(
                        "Removing pulsar '{}' as either no data, or no prior "
                        "is given".format(pname)
                    )
                    if pname in datanames:
                        datadict.pop(pname)
                    if pname in priornames:
                        priorfiles.pop(pname)

                    if pname in pulsarnames:
                        idx = pulsarnames.index(pname)
                        pulsars.pop(idx)
                        pulsarnames.pop(idx)

            # check location to output 'cwinpy_knope' input configuration files.
            configlocation = config.get(
                "knope", "config", fallback=os.path.join(basedir, "configs")
            )

            if configlocation is not None:
                # make directory
                try:
                    os.makedirs(configlocation, exist_ok=True)
                except Exception as e:
                    raise IOError(
                        "Could not create configuration directory "
                        "location: '{}'".format(e)
                    )

            # get the sampler (default is dynesty)
            sampler = config.get("knope", "sampler", fallback="dynesty")

            # get the sampler keyword arguments
            samplerkwargs = config.get("knope", "sampler_kwargs", fallback=None)
        else:
            raise IOError("Configuration file must have a [knope] section.")

        # create jobs (output and label set using pulsar name)
        for pname, parfile in zip(pulsarnames, pulsars):
            # create output directory
            psrbase = os.path.join(basedir, pname)

            try:
                os.makedirs(psrbase, exist_ok=True)
            except Exception as e:
                raise IOError(
                    "Could not created output directory '{}': {}".format(psrbase, e)
                )

            # create dictionary of configuration outputs
            configdict = {}

            configdict["par-file"] = parfile

            # data file(s)
            for freqfactor in ["1f", "2f"]:
                try:
                    configdict["data-file-{}".format(freqfactor)] = str(
                        datadict[pname][freqfactor]
                    )
                except KeyError:
                    pass

            configdict["outdir"] = psrbase
            configdict["label"] = pname
            configdict["prior"] = priorfiles[pname]
            configdict["sampler"] = sampler
            if samplerkwargs is not None:
                configdict["sampler-kwargs"] = samplerkwargs

            # output the configuration file
            configfile = os.path.join(configlocation, "{}.ini".format(pname))
            from configargparse import DefaultConfigFileParser

            parseobj = DefaultConfigFileParser()
            with open(configfile, "w") as fp:
                fp.write(parseobj.serialize(configdict))

            # add arguments to a job
            self.job.add_arg(configfile)

        self.dag.build()


def knope_dag(**kwargs):
    """
    Run knope_dag within Python. This will create a `HTCondor <https://research.cs.wisc.edu/htcondor/>`_
    DAG for running multiple ``cwinpy_knope`` instances on a computer cluster.

    Parameters
    ----------
    config: str
        A configuration file for the analysis.

    Returns
    -------
    dag:
        The pycondor :class:`pycondor.Dagman` object.
    """

    try:
        import configparser
    except ImportError:
        raise ImportError("To run 'cwinpy_knope_dag' you must install configparser")

    if "config" in kwargs:
        configfile = kwargs["config"]
    else:
        try:
            from argparse import ArgumentParser
        except ImportError:
            raise ImportError("To run 'cwinpy_knope_dag'")

        parser = ArgumentParser(
            description=(
                "A script to create a HTCondor DAG to run Bayesian inference to "
                "estimate the parameters of continuous gravitational-wave signals "
                "from a selection of known pulsars."
            )
        )
        parser.add_argument("config", help=("The configuation file for the analysis"))

        args = parser.parse_args()
        configfile = args.config

    config = configparser.ConfigParser()

    try:
        config.read_file(open(configfile, "r"))
    except Exception as e:
        raise IOError(
            "Problem reading configuation file '{}'\n: {}".format(configfile, e)
        )

    return KnopeDAGRunner(config).dag


def knope_dag_cli(**kwargs):
    """
    Entry point to the cwinpy_knope_dag script. This just calls :func:`cwinpy.knope.knope_dag`,
    but does not return any objects. 
    """

    _ = knope_dag(**kwargs)
