# coding: utf-8

"""
A Python module for Bayesian inferences with continuous gravitational wave
sources.
"""

import subprocess
import numpy as np
from setuptools import setup
from setuptools import Extension


def gsl_config(*args, **kwargs):
    """
    Run gsl-config and return pre-formatted output
    """

    cmd = ["gsl-config"] + list(args)

    return subprocess.check_output(cmd, **kwargs).decode("utf-8").strip()


ext_modules = [
    Extension(
        "cwinpy.heterodyne.fastheterodyne",
        sources=[
            "cwinpy/heterodyne/fastheterodyne.pyx",
        ],
        include_dirs=[
            np.get_include(),
            "cwinpy/heterodyne",
        ],
        libraries=["m"],
        extra_compile_args=[
            "-Wall",
            "-O3",
            "-Wextra",
            "-m64",
            "-ffast-math",
            "-fno-finite-math-only",
            "-funroll-loops",
        ],
    ),
    Extension(
        "cwinpy.cutils",
        sources=[
            "cwinpy/cutils.pyx",
        ],
        include_dirs=[
            np.get_include(),
            gsl_config("--cflags")[2:],
            "cwinpy",
        ],
        library_dirs=[
            gsl_config("--libs").split(" ")[0][2:],
        ],
        libraries=["gsl"],
        extra_compile_args=[
            "-Wall",
            "-O3",
            "-Wextra",
            "-m64",
            "-ffast-math",
            "-fno-finite-math-only",
            "-funroll-loops",
        ],
    ),
]

# add language level = 3
for e in ext_modules:
    e.cython_directives = {"language_level": "3"}

setup(
    use_scm_version=True,
    ext_modules=ext_modules,
)
