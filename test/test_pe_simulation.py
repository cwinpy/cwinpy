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
from cwinpy.hierarchical import DeltaFunctionDistribution
from cwinpy.pe.simulation import PEPulsarSimulationDAG
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
            os.path.split(os.path.realpath(__file__))[0], "test_pardir"
        )
        os.makedirs(cls.pardir, exist_ok=True)

        cls.names = ["J0000+0000", "J0100+0000"]
        cls.ras = [0.0, (1 / 24) * 2 * np.pi]
        cls.decs = [0.0, 0.0]
        cls.dists = [1.0, None]
        cls.pardict = {}
        for name, ra, dec, dist in zip(cls.names, cls.ras, cls.decs, cls.dists):
            par = PulsarParametersPy()
            par["PSRJ"] = name
            par["F"] = [100.0]  # set frequency to 100 Hz
            par["RAJ"] = ra
            par["DECJ"] = dec
            if dist is not None:
                par["DIST"] = (dist * u.kpc).to("m").value
            with open(os.path.join(cls.pardir, "{}.par".format(name)), "w") as fp:
                fp.write(str(par))
            cls.pardict[name] = os.path.join(cls.pardir, "{}.par".format(name))

        # create heterodyned data for testing
        cls.hetdir = os.path.join(
            os.path.split(os.path.realpath(__file__))[0], "test_hetdir"
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
            het.write(cls.hetfiles[det], overwrite=True)

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
            # wrong type for parfile
            PEPulsarSimulationDAG(ampdist=None, parfiles=1.0)

        with pytest.raises(ValueError):
            # wrong type for parfile
            PEPulsarSimulationDAG(ampdist=None, parfiles="blah_blah_blah")

        with pytest.raises(IOError):
            # non-existent par file
            PEPulsarSimulationDAG(ampdist=None, parfiles={"J0000+0000": "no.par"})

        with pytest.raises(ValueError):
            # no amplitude distribution and no parameter files
            PEPulsarSimulationDAG(ampdist=None, parfiles=None)

        with pytest.raises(ValueError):
            # directory clash for pulsar parameter files
            PEPulsarSimulationDAG(
                ampdist=None, parfiles=os.path.join(self.basedir, "pulsars")
            )

        with pytest.raises(TypeError):
            # wrong type for amplitude prior
            PEPulsarSimulationDAG(ampdist=1, parfiles=self.pardir)

        ampprior = bilby.core.prior.Uniform(0.0, 1e40, name="blah")
        with pytest.raises(KeyError):
            # wrong key for amplitude prior
            PEPulsarSimulationDAG(ampdist=ampprior)

        ampprior = bilby.core.prior.Uniform(0.0, 1e40, name="q22")

        with pytest.raises(TypeError):
            # wrong type for distance error
            PEPulsarSimulationDAG(ampdist=ampprior, distance_err=1)

        with pytest.raises(TypeError):
            # wrong type for prior
            PEPulsarSimulationDAG(ampdist=ampprior, prior=1)

        with pytest.raises(ValueError):
            # empty dictionary for prior
            PEPulsarSimulationDAG(ampdist=ampprior, prior={})

        with pytest.raises(FileNotFoundError):
            # bad prior file name
            PEPulsarSimulationDAG(ampdist=ampprior, prior="ksdkfkhvsad")

        with pytest.raises(TypeError):
            # wrong type for number of pulsars
            PEPulsarSimulationDAG(ampdist=ampprior, npulsars=2.3)

        with pytest.raises(ValueError):
            # wrong number of pulsars
            PEPulsarSimulationDAG(ampdist=ampprior, npulsars=0)

        with pytest.raises(TypeError):
            # wrong type for position distribution
            PEPulsarSimulationDAG(ampdist=ampprior, posdist=1, npulsars=1)

        with pytest.raises(KeyError):
            # wrong key for position distribution
            PEPulsarSimulationDAG(
                ampdist=ampprior,
                parfiles=self.pardir,
                posdist=bilby.core.prior.PriorDict(
                    {"blah": bilby.core.prior.Uniform(1, 2, name="blah")}
                ),
            )

        with pytest.raises(ValueError):
            # position distribution with unknown values
            PEPulsarSimulationDAG(
                ampdist=ampprior,
                npulsars=1,
                posdist=bilby.core.prior.PriorDict(
                    {"blah": bilby.core.prior.Uniform(1, 2, name="blah")}
                ),
            )

        with pytest.raises(TypeError):
            # wrong type for frequency distribution
            PEPulsarSimulationDAG(ampdist=ampprior, fdist=1, npulsars=1)

        with pytest.raises(TypeError):
            # wrong type for frequency distribution
            PEPulsarSimulationDAG(ampdist=ampprior, fdist=1, npulsars=1)

        with pytest.raises(TypeError):
            # wrong type for orientation distribution
            PEPulsarSimulationDAG(ampdist=ampprior, oridist=1, npulsars=1)

        with pytest.raises(TypeError):
            # wrong type for detector
            PEPulsarSimulationDAG(ampdist=ampprior, npulsars=1, detector=1)

        with pytest.raises(ValueError):
            # no detectors or data files
            PEPulsarSimulationDAG(
                ampdist=ampprior,
                npulsars=1,
                basedir=os.path.join(self.basedir, "nodet"),
                detector=None,
            )

        shutil.rmtree(os.path.join(self.basedir, "nodet"))

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
        testdir = os.path.join(self.basedir, "test_sim_pulsar")
        sim = PEPulsarSimulationDAG(
            ampdist=ampprior,
            oridist=oridist,
            posdist=posdist,
            npulsars=npulsars,
            detector=detectors,
            basedir=testdir,
            fdist=fdist,
        )

        # test error for start/end time setting
        with pytest.raises(ValueError):
            PEPulsarSimulationDAG(
                ampdist=ampprior,
                oridist=oridist,
                posdist=posdist,
                npulsars=npulsars,
                detector=detectors,
                basedir=testdir,
                fdist=fdist,
                starttime=1020304050,
            )

        shutil.rmtree(testdir)

        sim = PEPulsarSimulationDAG(
            ampdist=ampprior,
            oridist=oridist,
            posdist=posdist,
            npulsars=npulsars,
            detector=detectors,
            basedir=testdir,
            fdist=fdist,
        )

        # check directories
        for dir in glob.glob(os.path.join(testdir, "*")):
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
            assert len(glob.glob(os.path.join(testdir, dir, "*"))) == npulsars

        # check priors are the same as the default
        for pname in sim.priors:
            assert sim.priors[pname] == expectedprior

            # check fake pulsars contain the same values
            psr = PulsarParametersPy(
                os.path.join(testdir, "pulsars", "{}.par".format(pname))
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

        shutil.rmtree(testdir)

        # test galactic and galactocentric position distributions
        l = 0.3  # noqa: E741
        b = 1.1
        posdist = bilby.core.prior.PriorDict(
            {
                "l": bilby.core.prior.DeltaFunction(l, name="l"),
                "b": bilby.core.prior.DeltaFunction(b, name="b"),
                "dist": bilby.core.prior.DeltaFunction(dist, name="dist"),
            }
        )

        sim = PEPulsarSimulationDAG(
            ampdist=ampprior,
            oridist=oridist,
            posdist=posdist,
            npulsars=npulsars,
            detector=detectors,
            basedir=testdir,
            fdist=fdist,
        )

        ras = []
        decs = []
        dists = []
        for pname in sim.priors:
            # check fake pulsars contain the same values
            psr = PulsarParametersPy(
                os.path.join(testdir, "pulsars", "{}.par".format(pname))
            )

            ras.append(psr["RAJ"])
            decs.append(psr["DECJ"])
            dists.append(psr["DIST"])

        for vals in [ras, decs, dists]:
            assert np.all(np.array(vals) == vals[0])

        shutil.rmtree(testdir)

        x = 0.3
        y = 1.1
        z = 0.2
        posdist = bilby.core.prior.PriorDict(
            {
                "x": bilby.core.prior.DeltaFunction(x, name="x"),
                "y": bilby.core.prior.DeltaFunction(y, name="y"),
                "z": bilby.core.prior.DeltaFunction(z, name="z"),
            }
        )

        sim = PEPulsarSimulationDAG(
            ampdist=ampprior,
            oridist=oridist,
            posdist=posdist,
            npulsars=npulsars,
            detector=detectors,
            basedir=testdir,
            fdist=fdist,
        )

        ras = []
        decs = []
        dists = []
        for pname in sim.priors:
            # check fake pulsars contain the same values
            psr = PulsarParametersPy(
                os.path.join(testdir, "pulsars", "{}.par".format(pname))
            )

            ras.append(psr["RAJ"])
            decs.append(psr["DECJ"])
            dists.append(psr["DIST"])

        for vals in [ras, decs, dists]:
            assert np.all(np.array(vals) == vals[0])

        shutil.rmtree(testdir)

    def test_data_pulsars(self):
        """
        Test the code for when using existing pulsar data.
        """

        # use delta function priors to check values are the same
        q22 = 1.1e32
        iota = 0.1
        phi0 = 2.3
        psi = 0.5
        f0 = 100.0
        dist = 2.0
        ampprior = bilby.core.prior.DeltaFunction(q22, name="q22")
        oridist = bilby.core.prior.PriorDict(
            {
                "psi": bilby.core.prior.DeltaFunction(psi, name="psi"),
                "phi0": bilby.core.prior.DeltaFunction(phi0, name="phi0"),
                "iota": bilby.core.prior.DeltaFunction(iota, name="iota"),
            }
        )
        posdist = bilby.core.prior.PriorDict(
            {"dist": bilby.core.prior.DeltaFunction(dist, name="dist")}
        )
        fdist = bilby.core.prior.DeltaFunction(f0, name="f0")

        testdir = os.path.join(self.basedir, "test_data_pulsar")

        # set different priors for the pulsars
        priors = {}
        disterrs = {}
        for i, name in enumerate(self.names):
            priors[name] = bilby.core.prior.PriorDict(
                {"q22": bilby.core.prior.Uniform(0.0, (i + 1) * 1e40, name="q22")}
            )
            disterrs[name] = (i + 1) * 0.2

        # pass directory of par files and data files
        sim = PEPulsarSimulationDAG(
            ampdist=ampprior,
            prior=priors,
            distance_err=disterrs,
            oridist=oridist,
            posdist=posdist,
            parfiles=self.pardir,
            datafiles=self.hetfiles,
            basedir=testdir,
            fdist=fdist,
            overwrite_parameters=False,
        )

        for dir in glob.glob(os.path.join(testdir, "*")):
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
            assert len(glob.glob(os.path.join(testdir, dir, "*"))) == len(self.ras)

        # check signal values are correct
        for i, pname in enumerate(self.names):
            # check fake pulsars contain the same values
            psr = PulsarParametersPy(
                os.path.join(testdir, "pulsars", "{}.par".format(pname))
            )

            assert psr["PSRJ"] == pname
            assert psr["RAJ"] == self.ras[i]
            assert psr["DECJ"] == self.decs[i]
            assert psr["F0"] == f0
            assert psr["Q22"] == q22
            assert psr["PSI"] == psi
            assert psr["IOTA"] == iota
            assert psr["PHI0"] == phi0

            if self.dists[i] is None:
                assert np.allclose(psr["DIST"], (dist * u.kpc).to("m").value)
            else:
                assert np.allclose(psr["DIST"], (self.dists[i] * u.kpc).to("m").value)

            # check the priors
            assert sim.priors[pname]["q22"] == priors[pname]["q22"]
            assert "dist" in sim.priors[pname]
            if self.dists[i] is None:
                assert np.allclose(
                    sim.priors[pname]["dist"].sigma,
                    (dist * (i + 1) * 0.2 * u.kpc).to("m").value,
                )
                assert np.allclose(
                    sim.priors[pname]["dist"].mu, (dist * u.kpc).to("m").value
                )
            else:
                assert np.allclose(
                    sim.priors[pname]["dist"].sigma,
                    (self.dists[i] * (i + 1) * 0.2 * u.kpc).to("m").value,
                )
                assert np.allclose(
                    sim.priors[pname]["dist"].mu, (self.dists[i] * u.kpc).to("m").value
                )
            assert sim.priors[pname]["dist"].minimum == 0.0
            assert not np.isfinite(sim.priors[pname]["dist"].maximum)

        shutil.rmtree(testdir)

        # now add distance errors with a single float and with individual distributions
        disterrs = 0.3
        sim = PEPulsarSimulationDAG(
            ampdist=ampprior,
            prior=priors,
            distance_err=disterrs,
            oridist=oridist,
            posdist=posdist,
            parfiles=self.pardict,  # pass dictionary or par files this time
            datafiles=self.hetfiles,
            basedir=testdir,
            fdist=fdist,
            overwrite_parameters=True,
        )

        # check signal values are correct
        for i, pname in enumerate(self.names):
            # check fake pulsars contain the same values
            psr = PulsarParametersPy(
                os.path.join(testdir, "pulsars", "{}.par".format(pname))
            )

            assert psr["PSRJ"] == pname
            assert psr["RAJ"] == self.ras[i]
            assert psr["DECJ"] == self.decs[i]
            assert psr["F0"] == f0
            assert psr["Q22"] == q22
            assert psr["PSI"] == psi
            assert psr["IOTA"] == iota
            assert psr["PHI0"] == phi0

            # both distances should be the same as ones in the par file will be
            # overwritten
            assert np.allclose(psr["DIST"], (dist * u.kpc).to("m").value)

            # check the priors
            assert sim.priors[pname]["q22"] == priors[pname]["q22"]
            assert "dist" in sim.priors[pname]
            assert np.allclose(
                sim.priors[pname]["dist"].sigma,
                (dist * disterrs * u.kpc).to("m").value,
            )
            assert np.allclose(
                sim.priors[pname]["dist"].mu, (dist * u.kpc).to("m").value
            )
            assert sim.priors[pname]["dist"].minimum == 0.0
            assert not np.isfinite(sim.priors[pname]["dist"].maximum)

        shutil.rmtree(testdir)

        disterrs = {
            self.names[0]: bilby.core.prior.Uniform(
                (self.dists[0] * 0.9 * u.kpc).to("m").value,
                (self.dists[0] * 1.1 * u.kpc).to("m").value,
                name="dist",
            )
        }
        sim = PEPulsarSimulationDAG(
            ampdist=ampprior,
            prior=priors,
            distance_err=disterrs,
            oridist=oridist,
            posdist=posdist,
            parfiles=self.pardict,  # pass dictionary or par files this time
            datafiles=self.hetfiles,
            basedir=testdir,
            fdist=fdist,
            sampler_kwargs={"nlive": 2000},
            overwrite_parameters=False,
        )

        # check signal values are correct
        for i, pname in enumerate(self.names):
            # check fake pulsars contain the same values
            psr = PulsarParametersPy(
                os.path.join(testdir, "pulsars", "{}.par".format(pname))
            )

            assert psr["PSRJ"] == pname
            assert psr["RAJ"] == self.ras[i]
            assert psr["DECJ"] == self.decs[i]
            assert psr["F0"] == f0
            assert psr["Q22"] == q22
            assert psr["PSI"] == psi
            assert psr["IOTA"] == iota
            assert psr["PHI0"] == phi0

            if self.dists[i] is None:
                assert np.allclose(psr["DIST"], (dist * u.kpc).to("m").value)
            else:
                assert np.allclose(psr["DIST"], (self.dists[i] * u.kpc).to("m").value)

            # check the priors
            assert sim.priors[pname]["q22"] == priors[pname]["q22"]

            if self.dists[i] is not None:
                assert "dist" in sim.priors[pname]
                assert sim.priors[pname]["dist"] == disterrs[pname]
            else:
                assert "dist" not in sim.priors[pname]

        shutil.rmtree(testdir)

        # use a single prior for every pulsar
        priors = bilby.core.prior.PriorDict(
            {"h0": bilby.core.prior.Uniform(0.0, 1e-22, name="h0")}
        )

        # use h0 to set amplitude
        h0 = 1e-24
        ampprior = bilby.core.prior.DeltaFunction(h0, name="h0")

        # pass directory of par files and data files
        sim = PEPulsarSimulationDAG(
            ampdist=ampprior,
            prior=priors,
            oridist={},  # set default orientation
            posdist=posdist,
            parfiles=self.pardir,
            datafiles=self.hetfiles,
            basedir=testdir,
            fdist=fdist,
        )

        # check signal values are correct
        for i, pname in enumerate(self.names):
            # check fake pulsars contain the same values
            psr = PulsarParametersPy(
                os.path.join(testdir, "pulsars", "{}.par".format(pname))
            )

            assert psr["H0"] == h0
            assert 0.0 <= psr["PSI"] <= np.pi / 2.0
            assert 0.0 <= psr["IOTA"] <= np.pi
            assert 0.0 <= psr["PHI0"] <= np.pi

            # check the priors
            assert sim.priors[pname] == priors

        shutil.rmtree(testdir)

        # use a single prior file for each pulsar
        os.makedirs(os.path.join(testdir, "test_prior"), exist_ok=True)
        priors = bilby.core.prior.PriorDict(
            {"q22": bilby.core.prior.Uniform(0.0, 1e40, name="q22")}
        )
        priors.to_file(outdir=os.path.join(testdir, "test_prior"), label="test")
        priorfile = os.path.join(testdir, "test_prior", "test.prior")

        # use ellipticity to set amplitude distribution
        epsilon = 1e-7
        ampprior = DeltaFunctionDistribution(name="epsilon", peak=epsilon)

        # pass directory of par files and data files
        sim = PEPulsarSimulationDAG(
            ampdist=ampprior,
            prior=priorfile,
            oridist={},  # set default orientation
            posdist=posdist,
            parfiles=self.pardir,
            datafiles=self.hetfiles,
            basedir=testdir,
            fdist=fdist,
        )

        # check signal values are correct
        for i, pname in enumerate(self.names):
            # check fake pulsars contain the same values
            psr = PulsarParametersPy(
                os.path.join(testdir, "pulsars", "{}.par".format(pname))
            )

            assert psr["Q22"] == epsilon * 1e38 * np.sqrt(15 / (8 * np.pi))
            assert 0.0 <= psr["PSI"] <= np.pi / 2.0
            assert 0.0 <= psr["IOTA"] <= np.pi
            assert 0.0 <= psr["PHI0"] <= np.pi

            # check the priors
            assert sim.priors[pname] == priors

        shutil.rmtree(testdir)
