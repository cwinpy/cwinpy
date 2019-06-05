# coding: utf-8

"""
A Python module for Bayesian inferences with continuous gravitational wave
sources.
"""

import os
import versioneer

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def readfile(filename):
    with open(filename, encoding="utf-8") as fp:
        filecontents = fp.read()
    return filecontents


VERSION = versioneer.get_version()


setup(
    name="cwinpy",
    version=VERSION,
    author="Matthew Pitkin",
    author_email="matthew.pitkin@ligo.org",
    packages=["cwinpy", "cwinpy.knope"],
    url="http://git.ligo.org/CW/software/",
    license="MIT",
    description="A Python module for Bayesian inferences with continuous gravitational wave sources",
    long_description=\
        readfile(os.path.join(os.path.dirname(__file__), "README.md")),
    install_requires=\
        readfile(os.path.join(os.path.dirname(__file__), "requirements.txt")),
    entry_points={
        "console_scripts": [
            "cwinpy_knope=cwinpy.knope.knope:knope",
        ]
    },
    classifiers=["Programming Language :: Python :: 3.7"],
)