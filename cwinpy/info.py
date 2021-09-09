import glob
import os

import pkg_resources
from astropy.time import Time

from .parfile import PulsarParameters


class Runtimes(dict):
    """
    Dictionary-like class to hold start and end times of gravitational-wave
    detector observing runs.
    """

    @property
    def iso(self):
        """
        Convert and return run start and end times to ISO UTC format.
        """

        return Runtimes(
            {
                run: {
                    det: Time(self[run][det], format="gps", scale="utc").iso.tolist()
                    for det in self[run]
                }
                for run in self
            }
        )

    @property
    def mjd(self):
        """
        Convert and return run start and end times in MJD format.
        """

        return Runtimes(
            {
                run: {
                    det: Time(self[run][det], format="gps", scale="utc").mjd.tolist()
                    for det in self[run]
                }
                for run in self
            }
        )

    def run(self, run):
        """
        Return start and end times for all detectors in a given observing run.

        Parameters
        ----------
        run: str
            The name of the required observing run.

        Returns
        -------
        dict:
            A :class:`~cwinpy.observation_data.Runtimes` dictionary.
        """

        if run not in self:
            return None
        else:
            return Runtimes({run: self[run]})

    def detector(self, det):
        """
        Return start end end time for all runs for a given detector.

        Parameters
        ----------
        det: str
            The short name of the required detector, e.g., "H1" for the LIGO
            Hanford detector.

        Returns
        -------
        dict:
            A :class:`~cwinpy.observation_data.Runtimes` dictionary.
        """

        d = Runtimes()

        for run in self:
            if det in self[run]:
                d[run] = {det: self[run][det]}

        return d


#: Start and end times (GPS seconds) of open data observing runs
RUNTIMES = Runtimes()
RUNTIMES["S5"] = {
    "H1": [815155213, 875232014],
    "H2": [815155213, 875232014],
    "L1": [815155213, 875232014],
}
RUNTIMES["S6"] = {
    "H1": [931035615, 971622015],
    "L1": [931035615, 971622015],
}
RUNTIMES["O1"] = {
    "H1": [1126051217, 1137254417],
    "L1": [1126051217, 1137254417],
}
RUNTIMES["O2"] = {
    "H1": [1164556817, 1187733618],
    "L1": [1164556817, 1187733618],
    "V1": [1164556817, 1187733618],
}
RUNTIMES["O3a"] = {
    "H1": [1238166018, 1253977218],
    "L1": [1238166018, 1253977218],
    "V1": [1238166018, 1253977218],
}
# RUNTIMES["O3b"] = {
#    "H1": [1253977218, 1269363618],
#    "L1": [1253977218, 1269363618],
#    "V1": [1253977218, 1269363618],
# }
# RUNTIMES["O3"] = {
#    "H1": [1238166018, 1269363618],
#    "L1": [1238166018, 1269363618],
#    "V1": [1238166018, 1269363618],
# }


#: Start and end run times for continuous wave hardware injections (no injections for Virgo)
HW_INJ_RUNTIMES = Runtimes()
HW_INJ_RUNTIMES["S5"] = {
    "H1": [829412600, 875232014],
    "H2": [829412616, 875232014],
    "L1": [829413522, 875232014],
}
HW_INJ_RUNTIMES["S6"] = {
    "H1": [931035615, 971622015],
    "L1": [931035615, 971622015],
}
HW_INJ_RUNTIMES["O1"] = {
    "H1": [1126051217, 1137254417],
    "L1": [1126051217, 1137254417],
}
HW_INJ_RUNTIMES["O2"] = {
    "H1": [1164556817, 1187733618],
    "L1": [1164556817, 1187733618],
}
HW_INJ_RUNTIMES["O3a"] = {
    "H1": [1238166018, 1253977218],
    "L1": [1238166018, 1253977218],
}
# HW_INJ_RUNTIMES["O3b"] = {
#    "H1": [1253977218, 1269363618],
#    "L1": [1253977218, 1269363618],
# }
# HW_INJ_RUNTIMES["O3"] = {
#    "H1": [1238166018, 1269363618],
#    "L1": [1238166018, 1269363618],
# }


HW_INJ_RUNS = ["S5", "S6", "O1", "O2", "O3a"]
HW_INJ_BASE_PATH = pkg_resources.resource_filename("cwinpy", "data/")
#: locations of hardware injection parameter files
HW_INJ = {
    run: {
        "hw_inj_files": sorted(
            glob.glob(os.path.join(HW_INJ_BASE_PATH, run[0:2], "hw_inj", "*.par"))
        )
    }
    for run in HW_INJ_RUNS
}
for run in HW_INJ:
    HW_INJ[run]["hw_inj_parameters"] = [
        PulsarParameters(par) for par in HW_INJ[run]["hw_inj_files"]
    ]

#: Analysis segment flags for continuous-wave injections from GWOSC
HW_INJ_SEGMENTS = {
    "S5": {
        det: {"includesegments": f"{det}_CW_CAT1", "excludesegments": None}
        for det in HW_INJ_RUNTIMES["S5"]
    },
    "S6": {
        det: {"includesegments": f"{det}_CW_CAT1", "excludesegments": None}
        for det in HW_INJ_RUNTIMES["S6"]
    },
    "O1": {
        det: {
            "includesegments": f"{det}_CBC_CAT1",
            "excludesegments": f"{det}_NO_CW_HW_INJ",
        }
        for det in HW_INJ_RUNTIMES["O1"]
    },
    "O2": {
        det: {
            "includesegments": f"{det}_CBC_CAT1",
            "excludesegments": f"{det}_NO_CW_HW_INJ",
        }
        for det in HW_INJ_RUNTIMES["O2"]
    },
    "O3a": {
        det: {
            "includesegments": f"{det}_CBC_CAT1",
            "excludesegments": f"{det}_NO_CW_HW_INJ",
        }
        for det in HW_INJ_RUNTIMES["O3a"]
    },
}


#: Analysis segment flags for use GWOSC open data
ANALYSIS_SEGMENTS = {
    "S5": {det: f"{det}_CW_CAT1" for det in RUNTIMES["S5"]},
    "S6": {det: f"{det}_CW_CAT1" for det in RUNTIMES["S6"]},
    "O1": {det: f"{det}_CBC_CAT1" for det in RUNTIMES["O1"]},
    "O2": {det: f"{det}_CBC_CAT1" for det in RUNTIMES["O2"]},
    "O3a": {det: f"{det}_CBC_CAT1" for det in HW_INJ_RUNTIMES["O3a"]},
}


#: Base CVMFS directory for open GWOSC frame data
CVMFS_GWOSC_BASE = "/cvmfs/gwosc.osgstorage.org/gwdata"

#: GWOSC data server URL
CVMFS_GWOSC_DATA_SERVER = "datafind.gw-openscience.org"

#: GWOSC data types for different runs
CVMFS_GWOSC_DATA_TYPES = {
    "S5": {
        "4k": {
            "H1": "H1_LOSC_4_V1",
            "H2": "H2_LOSC_4_V1",
            "L1": "L1_LOSC_4_V1",
        },
    },
    "S6": {
        "4k": {
            "H1": "H1_LOSC_4_V1",
            "L1": "L1_LOSC_4_V1",
        }
    },
    "O1": {
        "4k": {
            "H1": "H1_LOSC_4_V1",
            "L1": "L1_LOSC_4_V1",
        },
        "16k": {
            "H1": "H1_LOSC_16_V1",
            "L1": "L1_LOSC_16_V1",
        },
    },
    "O2": {
        "4k": {
            "H1": "H1_GWOSC_O2_4KHZ_R1",
            "L1": "L1_GWOSC_O2_4KHZ_R1",
            "V1": "V1_GWOSC_O2_4KHZ_R1",
        },
        "16k": {
            "H1": "H1_GWOSC_O2_16KHZ_R1",
            "L1": "L1_GWOSC_O2_16KHZ_R1",
            "V1": "V1_GWOSC_O2_16KHZ_R1",
        },
    },
    "O3a": {
        "4k": {
            "H1": "H1_GWOSC_O3a_4KHZ_R1",
            "L1": "L1_GWOSC_O3a_4KHZ_R1",
            "V1": "V1_GWOSC_O3a_4KHZ_R1",
        },
        "16k": {
            "H1": "H1_GWOSC_O3a_16KHZ_R1",
            "L1": "L1_GWOSC_O3a_16KHZ_R1",
            "V1": "V1_GWOSC_O3a_16KHZ_R1",
        },
    },
}

#: data channel names in the GWOSC data frames
CVMFS_GWOSC_FRAME_CHANNELS = {
    "S5": {
        "4k": {
            "H1": "H1:LOSC-STRAIN",
            "H2": "H2:LOSC-STRAIN",
            "L1": "L1:LOSC-STRAIN",
        },
    },
    "S6": {
        "4k": {
            "H1": "H1:LOSC-STRAIN",
            "L1": "L1:LOSC-STRAIN",
        },
    },
    "O1": {
        "4k": {
            "H1": "H1:LOSC-STRAIN",
            "L1": "L1:LOSC-STRAIN",
        },
        "16k": {
            "H1": "H1:GWOSC-16KHZ_R1_STRAIN",
            "L1": "L1:GWOSC-16KHZ_R1_STRAIN",
        },
    },
    "O2": {
        "4k": {
            "H1": "H1:GWOSC-4KHZ_R1_STRAIN",
            "L1": "L1:GWOSC-4KHZ_R1_STRAIN",
            "V1": "V1:GWOSC-4KHZ_R1_STRAIN",
        },
        "16k": {
            "H1": "H1:GWOSC-16KHZ_R1_STRAIN",
            "L1": "L1:GWOSC-16KHZ_R1_STRAIN",
            "V1": "V1:GWOSC-16KHZ_R1_STRAIN",
        },
    },
    "O3a": {
        "4k": {
            "H1": "H1:GWOSC-4KHZ_R1_STRAIN",
            "L1": "L1:GWOSC-4KHZ_R1_STRAIN",
            "V1": "V1:GWOSC-4KHZ_R1_STRAIN",
        },
        "16k": {
            "H1": "H1:GWOSC-16KHZ_R1_STRAIN",
            "L1": "L1:GWOSC-16KHZ_R1_STRAIN",
            "V1": "V1:GWOSC-16KHZ_R1_STRAIN",
        },
    },
}

#: Base CVMFS directory for proprietory LVK frame data
CVMFS_LVK_BASE = "/cvmfs/oasis.opensciencegrid.org/ligo/frames"
