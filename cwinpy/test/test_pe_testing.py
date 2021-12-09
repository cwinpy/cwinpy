"""
Test script for testing class.
"""

import os
import shutil

import bilby
import numpy as np
import pycondor
import pytest
from cwinpy.pe.testing import PEPPPlotsDAG


class TestPEPP(object):
    @classmethod
    def setup_class(cls):
        """
        Create directory for tests and set default values.
        """

        # set the base directory
        cls.basedir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "base")

        cls.ninj = 50  # number of simulated signals
        cls.maxamp = 5e-23  # maximum amplitude
        cls.freqrange = (10.0, 100.0)  # frequency range

        # default prior dictionary
        cls.priors = {}
        cls.priors["h0"] = bilby.core.prior.Uniform(
            name="h0", minimum=0.0, maximum=1e-22
        )

    @classmethod
    def teardown_class(cls):
        """
        Remove test directory.
        """

        shutil.rmtree(cls.basedir)

    def test_failures(self):
        with pytest.raises(TypeError):
            PEPPPlotsDAG(1)

        with pytest.raises(ValueError):
            PEPPPlotsDAG(self.priors, ninj=-1)

        with pytest.raises(ValueError):
            PEPPPlotsDAG(self.priors, maxamp=-1.0)

        with pytest.raises(IOError):
            PEPPPlotsDAG(self.priors, basedir=1)

        with pytest.raises(TypeError):
            PEPPPlotsDAG(self.priors, basedir=self.basedir, freqrange=1)

        with pytest.raises(ValueError):
            PEPPPlotsDAG(self.priors, basedir=self.basedir, freqrange=[1, 2, 3])

    def test_run(self):
        run = PEPPPlotsDAG(
            self.priors,
            basedir=self.basedir,
            ninj=self.ninj,
            maxamp=self.maxamp,
            freqrange=self.freqrange,
        )

        assert len(run.pulsars) == self.ninj
        assert np.all(
            np.array([run.pulsars[psr]["parameters"]["H0"] for psr in run.pulsars])
            < self.maxamp
        )
        assert np.all(
            np.array([run.pulsars[psr]["parameters"]["F0"] for psr in run.pulsars])
            > self.freqrange[0]
        ) and np.all(
            np.array([run.pulsars[psr]["parameters"]["F0"] for psr in run.pulsars])
            < self.freqrange[1]
        )

        # check output prior
        for prior1, prior2 in zip(
            bilby.core.prior.PriorDict(filename=run.priorfile),
            bilby.core.prior.PriorDict(dictionary=self.priors),
        ):
            assert prior1 == prior2

        # check for the correct number of pulsars
        assert len(run.pulsars) == self.ninj

        # check for the correct number of output parameter files
        assert len(os.listdir(run.pulsardir)) == self.ninj

        # check output is a DAG
        assert isinstance(run.runner.dag.pycondor_dag, pycondor.Dagman)

        # checkout correct number of DAG jobs
        assert len(run.runner.dag.pycondor_dag.nodes) == (self.ninj + 1)

        # check config files are present
        configexists = 0
        for psr in run.pulsars:
            for job in run.runner.dag.pycondor_dag.nodes[:-1]:
                if psr in str(job.args):
                    if os.path.isfile(
                        os.path.join(
                            self.basedir, "configs", job.args[0].arg.split()[-1]
                        )
                    ):
                        configexists += 1
                        break

        assert configexists == self.ninj
