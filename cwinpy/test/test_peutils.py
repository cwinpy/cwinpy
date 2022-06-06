"""
Test script for peutils.

The data files used for this come from three different runs:

1. running the cwinpy_skyshift pipeline on the PULSAR03 CW hardware injection
from O1. Only two posterior files have been included.

2. running the cwinpy_knope_pipeline on O2 data for two pulsars (J0737-3039A,
J1843-1113) for a single harmonic search with results for individual detectors
and a multi-detector analysis.

3. running the cwinpy_knope_pipeline on O3 data for two pulsars (J0437-4715,
J0711-6830) for a dual harmonic search.

To make files a bit smaller these have been cut down to only include 1000
samples and only include posterior samples using the bilby_result script (with
the --max-samples 1000 and --lightweight arguments).
"""

import os
import pathlib

import lalsimulation as lalsim
import matplotlib as mpl
import numpy as np
import pytest
from astropy import units as u
from bilby.core.result import read_in_result
from cwinpy import HeterodynedData, MultiHeterodynedData
from cwinpy.data import PSDwrapper
from cwinpy.pe.peutils import (
    UpperLimitTable,
    find_heterodyned_files,
    find_results_files,
    optimal_snr,
    results_odds,
)


class TestPEUtils:
    @classmethod
    def setup_class(cls):
        cls.basedatadir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "peutils"
        )

        # set data directories
        cls.resdir = os.path.join(cls.basedatadir, "results")
        cls.hetdir = os.path.join(cls.basedatadir, "data")
        cls.pardir = os.path.join(cls.basedatadir, "pulsars")

        cls.pnames = ["JPULSAR03", "JPULSAR03A"]
        cls.dets = ["H1", "L1", "H1L1"]

    def test_find_heterodyned_files(self):
        with pytest.raises(TypeError):
            # invalid input type
            find_heterodyned_files(6.5)

        with pytest.raises(ValueError):
            # input is not a directory
            find_heterodyned_files("blah")

        # check nothing is found if passing invalid extension
        assert len(find_heterodyned_files(self.hetdir, ext="json")) == 0

        # check nothing is found if passing directory with no heterodyned files
        assert len(find_heterodyned_files(self.resdir)) == 0

        # check all files are found
        hf = find_heterodyned_files(self.hetdir)

        # check dictionary is keyed to pulsar names
        assert sorted(hf.keys()) == sorted(self.pnames)

        for k in hf.keys():
            # check subdictionaries are keyed on detector names
            assert sorted(hf[k].keys()) == sorted(self.dets[:2])

            for d in hf[k]:
                # check each file is a path
                assert isinstance(hf[k][d], pathlib.Path)
                assert hf[k][d].is_file()

    def test_find_results_files(self):
        with pytest.raises(TypeError):
            # invalid input type
            find_results_files(1.2)

        with pytest.raises(ValueError):
            # input is no a directory
            find_results_files("blah")

        # check no files returned if passing directory with no PE results files
        assert len(find_results_files(self.hetdir)) == 0

        # check all files are found
        hf = find_results_files(self.resdir)

        # check dictionary is keyed to pulsar names
        assert sorted(hf.keys()) == self.pnames

        for k in hf.keys():
            # check subdictionaries are keyed on detector names
            assert sorted(hf[k].keys()) == sorted(self.dets)

            for d in hf[k]:
                # check each file is a path
                assert isinstance(hf[k][d], pathlib.Path)
                assert hf[k][d].is_file()

    def test_optimal_snr(self):
        with pytest.raises(TypeError):
            # invalid input type for results directory
            optimal_snr(9.8, self.hetdir)

        with pytest.raises(TypeError):
            # invalid input type for heterodyned data directory
            optimal_snr(self.resdir, 1.6)

        with pytest.raises(ValueError):
            # invalid "which" value
            optimal_snr(self.resdir, self.hetdir, which="blah")

        resfiles = find_results_files(self.resdir)
        hetfiles = find_heterodyned_files(self.hetdir)

        # get single detector, single source SNR
        snr = optimal_snr(
            resfiles[self.pnames[0]]["H1"], hetfiles[self.pnames[0]]["H1"]
        )
        assert isinstance(snr, float)

        # use a dictionary instead
        snr = optimal_snr(
            resfiles[self.pnames[0]]["H1"], {"H1": hetfiles[self.pnames[0]]["H1"]}
        )
        assert isinstance(snr, float)

        # check using likelihood gives same value as posterior (a flat prior was used to produce the files)
        snrl = optimal_snr(
            resfiles[self.pnames[0]]["H1"],
            hetfiles[self.pnames[0]]["H1"],
            which="likelihood",
        )
        assert snr == snrl

        # pass remove outliers flag
        snr = optimal_snr(
            resfiles[self.pnames[0]]["H1"],
            {"H1": hetfiles[self.pnames[0]]["H1"]},
            remove_outliers=True,
        )
        assert isinstance(snr, float)

        # pass result as Result object
        snr = optimal_snr(
            read_in_result(resfiles[self.pnames[0]]["H1"]),
            hetfiles[self.pnames[0]]["H1"],
        )
        assert isinstance(snr, float) and snr == snrl

        # pass heterodyned data as HeterodynedData object
        snr = optimal_snr(
            resfiles[self.pnames[0]]["H1"],
            HeterodynedData.read(hetfiles[self.pnames[0]]["H1"]),
        )
        assert isinstance(snr, float)

        # get single joint multi-detector result
        snr = optimal_snr(resfiles[self.pnames[0]]["H1L1"], hetfiles[self.pnames[0]])
        assert isinstance(snr, float)

        # do the same, but with MultiHeterodynedData object
        snr = optimal_snr(
            resfiles[self.pnames[0]]["H1L1"],
            MultiHeterodynedData(hetfiles[self.pnames[0]]),
        )
        assert isinstance(snr, float)

        # get results for all pulsars and all detectors combination
        snr = optimal_snr(self.resdir, self.hetdir)
        assert isinstance(snr, dict)
        assert sorted(snr.keys()) == sorted(self.pnames)
        for k in snr:
            assert sorted(snr[k].keys()) == sorted(self.dets)
            assert all([isinstance(v, float) for v in snr[k].values()])

        # pass in par files directory
        snr = optimal_snr(self.resdir, self.hetdir, par=self.pardir)
        assert isinstance(snr, dict)
        assert sorted(snr.keys()) == sorted(self.pnames)
        for k in snr:
            assert sorted(snr[k].keys()) == sorted(self.dets)
            assert all([isinstance(v, float) for v in snr[k].values()])

        # get results for a single detector
        snr = optimal_snr(self.resdir, self.hetdir, det="H1")
        assert isinstance(snr, dict)
        assert sorted(snr.keys()) == sorted(self.pnames)
        assert all([isinstance(v, float) for v in snr.values()])

    def test_results_odds(self):
        with pytest.raises(TypeError):
            # invalid results type
            results_odds(4.1)

        resfiles = find_results_files(self.resdir)

        # pass Result object
        for scale in ["log10", "ln"]:
            lo = results_odds(
                read_in_result(resfiles[self.pnames[0]]["H1"]), scale=scale
            )
            assert isinstance(lo, float)

        # pass single file
        for scale in ["log10", "ln"]:
            lo = results_odds(resfiles[self.pnames[0]]["H1"], scale=scale)
            assert isinstance(lo, float)

        # pass invalid directory
        with pytest.raises(ValueError):
            results_odds(self.hetdir)

        # pass dictionary of files for single source
        losvn = results_odds(resfiles[self.pnames[0]], oddstype="svn")
        locvi = results_odds(resfiles[self.pnames[0]], oddstype="cvi")

        assert isinstance(losvn, float) and isinstance(locvi, float) and losvn > locvi

        # pass without giving coherent multidetector result
        with pytest.raises(RuntimeError):
            results_odds({det: resfiles[self.pnames[0]][det] for det in self.dets[:2]})

        with pytest.raises(KeyError):
            rd = {det: resfiles[self.pnames[0]][det] for det in self.dets[:2]}
            rd.update({"H1L": resfiles[self.pnames[0]]["H1L1"]})
            results_odds(rd)

        # use all pulsars
        for oddstype in ["svn", "cvi"]:
            for scale in ["log10", "ln"]:
                lo = results_odds(self.resdir, oddstype=oddstype, scale=scale)
                assert isinstance(lo, dict)
                assert sorted(lo.keys()) == sorted(self.pnames)
                assert all([isinstance(v, float) for v in lo.values()])

        # get results for one detector
        with pytest.raises(KeyError):
            results_odds(self.resdir, oddstype="svn", det="V1")

        lo = results_odds(self.resdir, oddstype="svn", det="H1")
        assert isinstance(lo, dict)
        assert sorted(lo.keys()) == sorted(self.pnames)
        assert all([isinstance(v, float) for v in lo.values()])


class TestUpperLimitTable:
    @classmethod
    def setup_class(cls):
        cls.basedatadir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "peutils"
        )

        # set data directories
        cls.resdirO2 = os.path.join(cls.basedatadir, "O2results")

        cls.pnamesO2 = ["J0737-3039A", "J1843-1113"]

        # rough rotation frequencies for pulsars
        cls.pfreqsO2 = [44.05 * u.Hz, 541.8 * u.Hz]

        # rough rotational frequency derivatives for pulsars
        cls.pfdotsO2 = [-3.4e-15 * u.Hz / u.s, -2.8e-15 * u.Hz / u.s]

        # rough distances for pulsars
        cls.pdistsO2 = [1.1 * u.kpc, 1.3 * u.kpc]

        cls.dets = ["H1", "L1", "V1", "H1L1V1"]

        # directory that does not contain any results
        cls.invaliddir = os.path.join(cls.basedatadir, "pulsars")

        # create ASD file for plot testing
        freqs = np.linspace(5, 1500, 1000)
        asd = np.zeros((len(freqs), 2))
        asd[:, 0] = freqs
        psdfunc = PSDwrapper(lalsim.SimNoisePSDaLIGOaLIGODesignSensitivityT1800044)
        asd[:, 1] = np.array([np.sqrt(psdfunc(f)) for f in freqs])
        cls.asdfile = "ALIGO_ASD.txt"
        np.savetxt(cls.asdfile, asd)

    @classmethod
    def teardown_class(cls):
        """
        Remove ASD file.
        """

        os.remove(cls.asdfile)

    def test_empty_table(self):
        """
        Test that an empty table is returned in certain circumstances.
        """

        t = UpperLimitTable()

        assert len(t) == 0
        assert isinstance(t, UpperLimitTable)

        # pass invalid path
        t = UpperLimitTable(resdir=self.invaliddir)

        assert len(t) == 0
        assert isinstance(t, UpperLimitTable)

        # ask for pulsar that does not exist in results
        t = UpperLimitTable(resdir=self.resdirO2, pulsars="J0534+2200")

        assert len(t) == 0
        assert isinstance(t, UpperLimitTable)

        # ask for results from detector that does not exist in results
        t = UpperLimitTable(resdir=self.resdirO2, detector="K1")

        assert len(t) == 0
        assert isinstance(t, UpperLimitTable)

    def test_errors(self):
        # pass invalid path
        with pytest.raises(ValueError):
            UpperLimitTable(resdir="Blah")

        # ask for invalid amplitude parameter
        with pytest.raises(ValueError):
            UpperLimitTable(resdir=self.resdirO2, ampparam="Z0")

        # give invalid upper limit quantile
        with pytest.raises(TypeError):
            UpperLimitTable(resdir=self.resdirO2, upperlimit="Z0")

        with pytest.raises(ValueError):
            UpperLimitTable(resdir=self.resdirO2, upperlimit=1.1)

        with pytest.raises(ValueError):
            UpperLimitTable(resdir=self.resdirO2, upperlimit=-0.1)

    def test_O2(self):
        """
        Test the O2 results.
        """

        # just get h0 90% credible upper limit for H1 detector and one pulsar
        t1p = UpperLimitTable(
            resdir=self.resdirO2,
            ampparam="h0",
            detector="H1",
            upperlimit=0.9,
            pulsars=self.pnamesO2[0],
        )

        for colname in ["PSRJ", "F0ROT", "F1ROT", "DIST", "H0_90%UL"]:
            assert colname in t1p.columns
        assert t1p["PSRJ"] == self.pnamesO2[0]

        # just get h0 90% credible upper limits for H1 detector
        t = UpperLimitTable(
            resdir=self.resdirO2, ampparam="h0", detector="H1", upperlimit=0.9
        )

        for colname in ["PSRJ", "F0ROT", "F1ROT", "DIST", "H0_90%UL"]:
            assert colname in t.columns
        assert sorted(t["PSRJ"].value.tolist()) == sorted(self.pnamesO2)
        assert t1p["H0_90%UL"][0] == t[t["PSRJ"] == self.pnamesO2[0]]["H0_90%UL"][0]

        # get 95% credible upper limits for all detectors and include
        # ellipticity, mass quadrupole and spin-down ratio limits
        t = UpperLimitTable(
            resdir=self.resdirO2,
            upperlimit=0.95,
            includeell=True,
            includeq22=True,
            includesdlim=True,
        )

        assert sorted(t["PSRJ"].value.tolist()) == sorted(self.pnamesO2)

        # check various values are close to expected values
        for i, psr in enumerate(self.pnamesO2):
            assert np.isclose(t.loc[psr]["F0ROT"], self.pfreqsO2[i], rtol=0.025)
            assert np.isclose(t.loc[psr]["F1ROT"], self.pfdotsO2[i], rtol=0.025)
            assert np.isclose(t.loc[psr]["DIST"], self.pdistsO2[i], rtol=0.05)

        # check all results columns are present
        for colname in ["PSRJ", "F0ROT", "F1ROT", "DIST", "SDLIM"]:
            assert colname in t.columns
        assert "SDLIM" in t.columns
        for ap in ["H0", "ELL", "Q22", "SDRAT"]:
            for det in self.dets:
                colname = f"{ap}_{det}_95%UL"
                assert colname in t.columns

        # try passing freqs, fdots, distances as dictionaries
        td = UpperLimitTable(
            resdir=self.resdirO2,
            upperlimit=0.95,
            includeell=True,
            includeq22=True,
            includesdlim=True,
            f0={psr: self.pfreqsO2[i] for i, psr in enumerate(self.pnamesO2)},
            fdot={psr: self.pfdotsO2[i] for i, psr in enumerate(self.pnamesO2)},
            distances={psr: self.pdistsO2[i] for i, psr in enumerate(self.pnamesO2)},
        )

        # check various values are the same as to expected values
        for i, psr in enumerate(self.pnamesO2):
            assert np.isclose(td.loc[psr]["F0ROT"], self.pfreqsO2[i], rtol=1e-8)
            assert np.isclose(td.loc[psr]["F1ROT"], self.pfdotsO2[i], rtol=1e-8)
            assert np.isclose(td.loc[psr]["DIST"], self.pdistsO2[i], rtol=1e-8)

    def test_table_string(self):
        """
        Test generating an rst table string.
        """

        t = UpperLimitTable(
            resdir=self.resdirO2, ampparam="h0", detector="H1", upperlimit=0.9
        )

        ts = t.table_string(format="rst", scinot=False)

        lines = ts.strip().split("\n")

        assert len(lines) == 6
        assert len(lines[0].split()) == 5  # five columns
        # check header
        for i, colname in enumerate(["PSRJ", "F0ROT", "F1ROT", "DIST", "H0_90%UL"]):
            assert colname == lines[1].split()[i]  # header
        assert self.pnamesO2[0] == lines[3].split()[0]
        assert self.pnamesO2[1] == lines[4].split()[0]
        assert float(lines[3].split()[-1]) > 0
        assert float(lines[4].split()[-1]) > 0

        for i in range(len(self.pnamesO2)):
            assert np.isclose(
                float(lines[3 + i].split()[1]), self.pfreqsO2[i].value, rtol=0.025
            )
            assert np.isclose(
                float(lines[3 + i].split()[2]), self.pfdotsO2[i].value, rtol=0.025
            )
            assert np.isclose(
                float(lines[3 + i].split()[3]), self.pdistsO2[i].value, rtol=0.05
            )

        latextab = t.table_string(format="latex")
        assert r"\begin{table}" in latextab and r"\end{table}" in latextab

        htmltab = t.table_string(format="html")
        assert "<table>" in htmltab and "</table>" in htmltab

    def test_plot(self):
        """
        Test the plotting function
        """

        # create table
        t = UpperLimitTable(
            resdir=self.resdirO2,
            upperlimit=0.95,
            includeell=True,
            includeq22=True,
            includesdlim=True,
        )

        with pytest.raises(TypeError):
            t.plot(4.5)

        # try plotting h0 (with histogram)
        fig = t.plot(
            column="H0",
            histogram=True,
            showsdlim=True,
            highlightpsrs=[self.pnamesO2[0]],
            asds=[self.asdfile, self.asdfile],
            tobs=[0.5 * 365.25 * 86400, 0.3 * 365.25 * 86400],
        )

        assert isinstance(fig, mpl.figure.Figure)

        # try plotting ellipticity (with histogram)
        fig = t.plot(
            column="ELL",
            histogram=True,
            showsdlim=True,
            highlightpsrs=[self.pnamesO2[0]],
            showq22=True,
            showtau=[1e4, 1e7],
        )

        assert isinstance(fig, mpl.figure.Figure)

        # try plotting joint plot
        fig = t.plot(
            column=["Q22_H1L1V1_95%UL", "SDRAT_H1_95%UL"],
            jointplot=True,
            yscale="linear",
            highlightpsrs=[self.pnamesO2[1]],
        )

        assert isinstance(fig, mpl.figure.Figure)
