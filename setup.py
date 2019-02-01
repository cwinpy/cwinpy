# coding: utf-8

"""
A Python module for Bayesian inferences with continuous gravitational wave
sources.
"""

import os
import re
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

MAJOR, MINOR1, MINOR2, RELEASE, SERIAL = sys.version_info

READFILE_KWARGS = {"encoding": "utf-8"} if MAJOR >= 3 else {}

def readfile(filename):
    with open(filename, **READFILE_KWARGS) as fp:
        filecontents = fp.read()
    return filecontents

VERSION_REGEX = re.compile("__version__ = \"(.*?)\"")
CONTENTS = readfile(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cwinpy", "__init__.py"))

VERSION = VERSION_REGEX.findall(CONTENTS)[0]


setup(name="cwinpy",
      version=VERSION,
      author="Matthew Pitkin",
      author_email="matthew.pitkin@ligo.org",
      packages=["cwinpy"],
      url="http://git.ligo.org/matthew-pitkin/cwinpy/",
      license="MIT",
      description="A Python module for Bayesian inferences with continuous gravitational wave sources",
      long_description=\
          readfile(os.path.join(os.path.dirname(__file__), "README.md")),
      install_requires=\
          readfile(os.path.join(os.path.dirname(__file__), "requirements.txt")),
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.7"])
