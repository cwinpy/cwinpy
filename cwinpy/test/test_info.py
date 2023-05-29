import os
from urllib.error import HTTPError

import pytest

from cwinpy.heterodyne import remote_frame_cache
from cwinpy.info import (
    CVMFS_GWOSC_DATA_SERVER,
    CVMFS_GWOSC_DATA_TYPES,
    CVMFS_GWOSC_FRAME_CHANNELS,
    HW_INJ,
    RUNTIMES,
    is_hwinj,
)


def check_connection():
    from socket import create_connection

    from cwinpy.info import CVMFS_GWOSC_DATA_SERVER

    try:
        create_connection((CVMFS_GWOSC_DATA_SERVER, 80))
    except (TimeoutError, ConnectionRefusedError):
        return False

    return True


@pytest.mark.skipif(
    not check_connection(),
    reason="Could not establish connection to GWOSC frame server",
)
def test_frame_cache():
    """
    Test that CVMFS frame caches can be found.
    """

    host = CVMFS_GWOSC_DATA_SERVER

    for run in RUNTIMES:
        if run != "O3":
            # exclude full O3 as CVMFS_GWOSC_DATA_TYPES contains O3a and O3b separately
            for det in RUNTIMES[run]:
                start, end = RUNTIMES[run][det]
                for rate in ["4k", "16k"]:
                    if rate in CVMFS_GWOSC_DATA_TYPES[run]:
                        channel = CVMFS_GWOSC_FRAME_CHANNELS[run][rate][det]
                        frtype = CVMFS_GWOSC_DATA_TYPES[run][rate][det]

                        # get frame file cache
                        try:
                            cache = remote_frame_cache(
                                start,
                                end,
                                channel,
                                frametype=frtype,
                                host=host,
                            )

                            # make sure a list of frame files has been produced
                            assert len(cache[det]) > 1
                        except HTTPError:
                            # ignore if no connection available
                            pass


def test_is_hwinj():
    """
    Test function that checks whether something is a hardware injection file.
    """

    # test incompatible type
    with pytest.raises(TypeError):
        is_hwinj(3.4)

    # test with random string
    assert not is_hwinj("blah")

    # test with current file
    assert not is_hwinj(__file__)

    # test actual HW injection pulsar name
    assert is_hwinj("PULSAR00")

    def idxS5(psr):
        return [
            i
            for i in range(len(HW_INJ["S5"]["hw_inj_files"]))
            if psr in HW_INJ["S5"]["hw_inj_files"][i]
        ][0]

    assert os.path.abspath(is_hwinj("PULSAR00", return_file=True)) == os.path.abspath(
        HW_INJ["S5"]["hw_inj_files"][idxS5("PULSAR00")]
    )

    # test with name of pulsar
    assert is_hwinj("JPULSAR14")

    def idxO1(psr):
        return [
            i
            for i in range(len(HW_INJ["O1"]["hw_inj_files"]))
            if psr in HW_INJ["O1"]["hw_inj_files"][i]
        ][0]

    assert os.path.abspath(is_hwinj("JPULSAR14", return_file=True)) == os.path.abspath(
        HW_INJ["O1"]["hw_inj_files"][idxO1("PULSAR14")]
    )

    # test with PulsarParameters object
    psr = HW_INJ["O2"]["hw_inj_parameters"][4]
    assert is_hwinj(psr)
    assert os.path.abspath(is_hwinj(psr, return_file=True)) == os.path.abspath(
        HW_INJ["O2"]["hw_inj_files"][4]
    )
