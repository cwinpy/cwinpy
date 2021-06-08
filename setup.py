# coding: utf-8

"""
A Python module for Bayesian inferences with continuous gravitational wave
sources.
"""

import numpy as np
from setuptools import setup
from setuptools import Extension


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
            "-march=native",
            "-funroll-loops",
        ],
    )
]

setup(
    use_scm_version=True,
    ext_modules=ext_modules,
)
