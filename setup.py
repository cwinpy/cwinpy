# coding: utf-8

"""
A Python module for Bayesian inferences with continuous gravitational wave
sources.
"""

from setuptools import setup

import os

import versioneer


def readfile(filename):
    with open(filename, encoding="utf-8") as fp:
        filecontents = fp.read()
    return filecontents


VERSION = versioneer.get_version()
CMDCLASS = versioneer.get_cmdclass()

setup(
    name="cwinpy",
    version=VERSION,
    author="Matthew Pitkin",
    author_email="matthew.pitkin@ligo.org",
    packages=["cwinpy", "cwinpy.pe", "cwinpy.iostream"],
    url="http://git.ligo.org/CW/software/cwinpy",
    license="MIT",
    description="A Python module for Bayesian inferences with continuous gravitational-wave sources",
    long_description=readfile(os.path.join(os.path.dirname(__file__), "README.md")),
    long_description_content_type="text/markdown",
    install_requires=readfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt")
    ),
    entry_points={
        "console_scripts": [
            "cwinpy_pe=cwinpy.pe.pe:pe_cli",
            "cwinpy_pe_dag=cwinpy.pe.pe:pe_dag_cli",
            "cwinpy_pe_generate_pp_plots=cwinpy.pe.testing:generate_pp_plots",
        ]
    },
    cmdclass=CMDCLASS,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
