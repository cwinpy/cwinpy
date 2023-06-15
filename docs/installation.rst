############
Installation
############

CWInPy can currently be installed and run on Linux and Mac OS systems with Python 3.8 or higher,
but Windows installation is not supported (although installation though `WSL
<https://docs.microsoft.com/en-us/windows/wsl/install>`__ can be successful). One of the core
package requirements needs the `GNU Scientific Library (GSL) <https://www.gnu.org/software/gsl/>`_
to be installed, so this should either be installed globally (e.g., via the OS's standard package
management system) or could be installed within a `conda environment
<https://anaconda.org/conda-forge/gsl>`_.

.. note::

   If planning to use the `Tempo2 <https://bitbucket.org/psrsoft/tempo2/src/master/>`__ pulsar
   timing package within CWInPy for phases calculations, it must already be installed on you system
   (this will be done by default if installing CWInPy through its conda package). The easiest way to
   install Tempo2 is within a conda environment and installing the `libstempo
   <https://vallis.github.io/libstempo/>`_ package:

   .. code-block:: console

      $ conda install -c conda-forge libstempo

Install CWInPy from source
==========================

CWInPy can be installed from its source `git <https://git-scm.com/>`_ `repository
<https://github.com/cwinpy/cwinpy>`_ using the supplied Python setup script.

First, clone the repository

.. tab-set::

   .. tab-item:: HTTPS

      .. code-block:: console

         $ git clone https://github.com/cwinpy/cwinpy.git

   .. tab-item:: ssh

      .. code-block:: console

         $ git clone git@github.com:cwinpy/cwinpy.git

   .. tab-item:: GitHub CLI

      Using the GitHub `CLI <https://cli.github.com/>`_

      .. code-block:: console

         $ gh repo clone cwinpy/cwinpy

then install CWInPy along with all its requirements using:

.. tab-set::

   .. tab-item:: Standard

      .. code-block:: console

         $ cd cwinpy/
         $ pip install .

      .. note::

         If installing on Mac OS, the required htcondor package is not installable via pip. Either
         install htcondor as described `here
         <https://htcondor.readthedocs.io/en/v9_0/getting-htcondor/install-macos-as-root.html>`__,
         or install it in a conda environment with:

         .. code-block:: console

            $ conda install -c conda-forge htcondor

         before attempting to install CWInPy.

      If wanting to be able to run the CWInPy test suite see the installation instructions in the
      "Developer" tab.

   .. tab-item:: Developer

      For developers, you can either install with the requirements using

      .. code-block:: console

         $ cd cwinpy/
         $ pip install -e .[test,dev,docs]

      or install using your own versions of the required files using

      .. code-block:: console

         $ cd cwinpy/
         $ pip install --no-deps -e .[dev]

      The development installation includes the `pre-commit
      <https://github.com/pre-commit/pre-commit>`_ package. This is used to set up git pre-commit
      hooks that automatically run scripts such as `flake8 <https://pypi.org/project/flake8/>`_,
      `black <https://pypi.org/project/black/>`_, `isort <https://isort.readthedocs.io/>`_ and a
      `spell check <https://github.com/codespell-project/codespell>`_ to ensure that any commits you
      make have a consistent style. Before starting as a developer you must run

      .. code-block:: console

         $ pre-commit install

      within the ``cwinpy`` repository directory, which will add the ``pre-commit`` hook file to
      your ``.git/hooks`` directory. After this, when running ``git commit`` the checks will
      automatically be run and results will be presented to you. In some cases the required fixes
      will be automatically applied, but in cases where there was some failure it will print a
      message about why it failed. In these cases you will have to manually correct the offending
      files before running ``git commit`` again.

      .. note::

         If installing the ``test`` packages you will need to have the
         `Tempo2 <https://bitbucket.org/psrsoft/tempo2/src/master/>`_ pulsar timing package
         installed. The easiest way to install this is by using ``conda`` to install the
         `libstempo <https://vallis.github.io/libstempo/>`_ package:

         .. code-block:: console

            $ conda install -c conda-forge libstempo 

Running parameter estimation via the `bilby <https://lscsoft.docs.ligo.org/bilby/index.html>`_
package with any sampler other than the default of `dynesty
<https://dynesty.readthedocs.io/en/latest/>`_ requires those additional samplers to be `installed
separately <https://lscsoft.docs.ligo.org/bilby/samplers.html#installing-samplers>`_.

Install CWInPy via a package manager
====================================

CWInPy is available through the `PyPI <https://pypi.org/project/cwinpy/>`_ and `Conda
<https://anaconda.org/conda-forge/cwinpy>`_ package management systems and can be installed using:

.. tab-set::

   .. tab-item:: PyPI

      .. code-block:: console

         $ pip install cwinpy

   .. tab-item:: Conda

      Within a conda environment use

      .. code-block:: console

         $ conda install -c conda-forge cwinpy

CWInPy is also available as part of the
`IGWN Conda Distribution <https://computing.docs.ligo.org/conda/>`_ over CVMFS.
To install CVMFS (for Linux and Mac OS only) you can follow the instructions `here
<https://computing.docs.ligo.org/guide/cvmfs/>`__. Once this is installed you can enter a Conda
environment, e.g., ``igwn``, from a terminal using:

   .. code-block:: console

      $ source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
      $ conda activate igwn

If running on many of the `IGWN Computing Grid clusters
<https://computing.docs.ligo.org/guide/dhtc/>`__, the first line can be omitted as the required
Conda distribution should automatically be in your path.

Documentation
=============

The documentation for the project can be built locally from the source code by installing CWInPy
with the additional required dependencies using:

.. code-block:: console

   $ pip install .[docs]

and then running

.. code-block:: console

   $ cd docs
   $ make html

Testing
=======

The package comes with a range of unit tests that can be run from the cloned repository.
To run these tests first install CWInPy with the additional required dependencies using

.. code-block:: bash

   $ pip install -e .[test]

and then run `pytest <https://docs.pytest.org/en/latest/>`_ with:

.. code-block:: console

   $ pytest

from the repository's base directory after the code has been installed.

.. note::

   If installing the ``test`` packages you will need to have the
   `Tempo2 <https://bitbucket.org/psrsoft/tempo2/src/master/>`_ pulsar timing package
   installed. The easiest way to install this is by using ``conda`` to install the
   `libstempo <https://vallis.github.io/libstempo/>`_ package:

   .. code-block:: console

      $ conda install -c conda-forge libstempo