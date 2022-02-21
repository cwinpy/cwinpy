from cwinpy.heterodyne import remote_frame_cache
from cwinpy.info import (
    CVMFS_GWOSC_DATA_SERVER,
    CVMFS_GWOSC_DATA_TYPES,
    CVMFS_GWOSC_FRAME_CHANNELS,
    RUNTIMES,
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
                        cache = remote_frame_cache(
                            start,
                            end,
                            channel,
                            frametype=frtype,
                            host=host,
                        )

                        # make sure a list of frame files has been produced
                        assert len(cache[det]) > 1
