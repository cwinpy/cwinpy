# coding: utf-8

"""
A Python module for Bayesian inferences with continuous gravitational wave
sources.
"""

from setuptools import setup
from setuptools import Extension

import numpy


# check whether user has Cython
try:
    import Cython  # noqa: F401
except ImportError:
    have_cython = False
else:
    have_cython = True

extra_compile_args = [
    "-Wall",
    "-O3",
    "-Wextra",
    "-m64",
    "-ffast-math",
    "-fno-finite-math-only",
    "-march=native",
    "-funroll-loops",
]

ext = "pyx" if have_cython else "c"
ext_modules = [
    Extension(
        "cwinpy.heterodyne.fastheterodyne",
        sources=[
            "cwinpy/heterodyne/fastheterodyne.{}".format(ext),
        ],
        include_dirs=[
            numpy.get_include(),
            "cwinpy/heterodyne",
        ],
        libraries=["m"],
        extra_compile_args=extra_compile_args,
    )
]

if have_cython:
    from Cython.Build import cythonize

    ext_modules = cythonize(ext_modules)


setup(
    use_scm_version=True,
    ext_modules=ext_modules,
)
