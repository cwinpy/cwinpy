"""
Test script for simulations.
"""

import glob
import os
import shutil

import bilby
import numpy as np
import pytest
from astropy import units as u
from cwinpy.data import HeterodynedData
from cwinpy.pe.simulation import PEMassQuadrupoleSimulationDAG
from lalpulsar.PulsarParametersWrapper import PulsarParametersPy


class TestPESimulation(object):
    @classmethod
    def setup_class(cls):
        """
        Create directory for simulations.
        """

        # set the base directory
        cls.basedir = os.path.join(
            os.path.split(os.path.realpath(__file__))[0], "simulation"
        )
        os.makedirs(cls.basedir, exist_ok=True)

        # create pulsar parameter files for testing
        cls.pardir = os.path.join(
            os.path.split(os.path.realpath(__file__))[0], "pardir"
        )
        os.makedirs(cls.pardir, exist_ok=True)

        for name, ra, dist in zip(
            ["J0000+0000", "J0100+0000"], [0.0, (1 / 24) * 2 * np.pi], [1.0, 2.0]
        ):
            par = PulsarParametersPy()
            par["PSRJ"] = name
            par["F"] = [100.0]  # set frequency to 100 Hz
            par["RAJ"] = ra
            par["DECJ"] = 0.0
            par["DIST"] = (dist * u.kpc).to("m").value
            with open(os.path.join(cls.pardir, "{}.par".format(name)), "w") as fp:
                fp.write(str(par))

        # create heterodyned data for testing
        cls.hetdir = os.path.join(
            os.path.split(os.path.realpath(__file__))[0], "hetdir"
        )
        os.makedirs(cls.hetdir, exist_ok=True)

        cls.hetfiles = {}
        for det, name in zip(["H1", "L1"], ["J0000+0000", "J0100+0000"]):
            het = HeterodynedData(
                times=np.linspace(1000000000.0, 1000086340.0, 1440),
                fakeasd=det,
                par=os.path.join(cls.pardir, "{}.par".format(name)),
            )
            cls.hetfiles[det] = os.path.join(cls.hetdir, "{}.hdf".format(name))
            het.write(cls.hetfiles[det])

    @classmethod
    def teardown_class(cls):
        """
        Remove test simulation directory.
        """

        shutil.rmtree(cls.basedir)
        shutil.rmtree(cls.pardir)
        shutil.rmtree(cls.hetdir)

    def test_failures(self):
        with pytest.raises(TypeError):
            # no positional argument
            PEMassQuadrupoleSimulationDAG()

        with pytest.raises(TypeError):
            # wrong type for parfile
            PEMassQuadrupoleSimulationDAG(None, parfiles=1.0)

        with pytest.raises(IOError):
            # non-existent par file
            PEMassQuadrupoleSimulationDAG(None, parfiles={"J0000+0000": "no.par"})

        with pytest.raises(TypeError):
            # wrong type for amplitude prior
            PEMassQuadrupoleSimulationDAG(None)

        ampprior = bilby.core.prior.Uniform(0.0, 1e40, name="q22")

        with pytest.raises(TypeError):
            # wrong type for distance error
            PEMassQuadrupoleSimulationDAG(ampprior, distance_err=1)

        with pytest.raises(TypeError):
            # wrong type for prior
            PEMassQuadrupoleSimulationDAG(ampprior, prior=1)

        with pytest.raises(TypeError):
            # wrong type for number of pulsars
            PEMassQuadrupoleSimulationDAG(ampprior, npulsars=2.3)

        with pytest.raises(ValueError):
            # wrong number of pulsars
            PEMassQuadrupoleSimulationDAG(ampprior, npulsars=0)

        with pytest.raises(TypeError):
            # wrong type for position distribution
            PEMassQuadrupoleSimulationDAG(ampprior, posdist=1)

        with pytest.raises(KeyError):
            # wrong key for position distribution
            PEMassQuadrupoleSimulationDAG(
                ampprior,
                parfiles=self.pardir,
                posdist=bilby.core.prior.PriorDict(
                    {"blah": bilby.core.prior.Uniform(1, 2, name="blah")}
                ),
            )

        with pytest.raises(ValueError):
            # position distribution with unknown values
            PEMassQuadrupoleSimulationDAG(
                ampprior,
                npulsars=1,
                posdist=bilby.core.prior.PriorDict(
                    {"blah": bilby.core.prior.Uniform(1, 2, name="blah")}
                ),
            )

        with pytest.raises(TypeError):
            # wrong type for frequency distribution
            PEMassQuadrupoleSimulationDAG(ampprior, fdist=1, npulsars=1)

        with pytest.raises(TypeError):
            # wrong type for frequency distribution
            PEMassQuadrupoleSimulationDAG(ampprior, fdist=1, npulsars=1)

        with pytest.raises(TypeError):
            # wrong type for orientation distribution
            PEMassQuadrupoleSimulationDAG(ampprior, oridist=1, npulsars=1)

        with pytest.raises(TypeError):
            # wrong type for detector
            PEMassQuadrupoleSimulationDAG(ampprior, oridist=1, npulsars=1, detector=1)

    def test_sim_pulsars(self):
        """
        Test the code for when generating fake pulsars.
        """

        # use delta function priors to check values are the same
        q22 = 1.1e32
        ra = 1.0
        dec = 0.5
        dist = 2.5
        iota = 0.1
        phi0 = 2.3
        psi = 0.5
        f0 = 100.0
        ampprior = bilby.core.prior.DeltaFunction(q22, name="q22")
        posdist = bilby.core.prior.PriorDict(
            {
                "ra": bilby.core.prior.DeltaFunction(ra, name="ra"),
                "dec": bilby.core.prior.DeltaFunction(dec, name="dec"),
                "dist": bilby.core.prior.DeltaFunction(dist, name="dist"),
            }
        )
        oridist = bilby.core.prior.PriorDict(
            {
                "psi": bilby.core.prior.DeltaFunction(psi, name="psi"),
                "phi0": bilby.core.prior.DeltaFunction(phi0, name="phi0"),
                "iota": bilby.core.prior.DeltaFunction(iota, name="iota"),
            }
        )
        fdist = bilby.core.prior.DeltaFunction(f0, name="f0")

        # the expected default PE prior
        expectedprior = bilby.core.prior.PriorDict(
            {
                "h0": bilby.core.prior.Uniform(
                    0.0, 1e-22, name="h0", latex_label="$h_0$"
                ),
                "iota": bilby.core.prior.Sine(name="iota", latex_label=r"$\iota$"),
                "phi0": bilby.core.prior.Uniform(
                    0.0, np.pi, name="phi0", latex_label=r"$\phi_0$"
                ),
                "psi": bilby.core.prior.Uniform(
                    0.0, np.pi / 2, name="psi", latex_label=r"$\psi$"
                ),
            }
        )

        npulsars = 2
        detectors = "H1"
        sim = PEMassQuadrupoleSimulationDAG(
            ampprior,
            oridist=oridist,
            posdist=posdist,
            npulsars=npulsars,
            detector=detectors,
            basedir=self.basedir,
            fdist=fdist,
        )

        # check directories
        for dir in glob.glob(os.path.join(self.basedir, "*")):
            assert os.path.basename(dir) in [
                "configs",
                "error",
                "log",
                "out",
                "priors",
                "pulsars",
                "results",
                "submit",
            ]

        for dir in ["configs", "priors", "pulsars"]:
            assert len(glob.glob(os.path.join(self.basedir, dir, "*"))) == npulsars

        # check priors are the same as the default
        for pname in sim.priors:
            assert sim.priors[pname] == expectedprior

            # check fake pulsars contain the same values
            psr = PulsarParametersPy(
                os.path.join(self.basedir, "pulsars", "{}.par".format(pname))
            )

            assert psr["PSRJ"] == pname
            assert psr["RAJ"] == ra
            assert psr["DECJ"] == dec
            assert psr["F0"] == f0
            assert psr["Q22"] == q22
            assert psr["PSI"] == psi
            assert psr["IOTA"] == iota
            assert psr["PHI0"] == phi0
            assert np.allclose(psr["DIST"], (dist * u.kpc).to("m").value)
