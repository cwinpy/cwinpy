# coding: utf-8

"""
A Python module for Bayesian inferences with continuous gravitational wave
sources.
"""

import numpy as np
from setuptools import Extension, setup

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
    )
]

# add language level = 3
for e in ext_modules:
    e.cython_directives = {"language_level": "3"}

setup(
    ext_modules=ext_modules,
)
