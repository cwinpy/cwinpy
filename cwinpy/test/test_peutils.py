"""
Test script for peutils.

The data files used for this come from running the cwinpy_skyshift pipeline
on the PULSAR03 CW hardware injection from O1. To make file a bit smaller, only
two of posterior files from that analysis (one on-source and one sky-shifted)
have been included (and these have been cut down to only include 1000 samples
using the bilby_result script).
"""

import os
import pathlib

import pytest
from bilby.core.result import read_in_result
from cwinpy import HeterodynedData, MultiHeterodynedData
from cwinpy.peutils import (
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
