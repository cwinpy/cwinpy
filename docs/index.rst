####################
CWInPy documentation
####################

CWInPy is a Python package designed to perform searches for, and inference on, `continuous
quasi-monochromatic gravitational-wave signals <https://www.ligo.org/science/GW-Continuous.php>`_.
In particular, signals that might be emitted by non-axisymmetric and rapidly rotating `neutron stars
<https://en.wikipedia.org/wiki/Neutron_star>`_, such as `pulsars
<https://en.wikipedia.org/wiki/Pulsar>`_. CWInPy stands for "\ **C**\ ontinuous (gravitational) **W**\ ave
**In**\ ference in **Py**\ thon".

The package provides tools for processing raw gravitational-wave strain time-series data based on
the phase evolution of a given source (using a `TEMPO <https://en.wikipedia.org/wiki/Tempo_(astronomy)>`_-style parameter file); a class to store and
display this data is available. Following data processing, the package provides tools for inferring
the unknown gravitational-wave parameters describing the source and its orientation. An integrated
pipeline combining both these stages (``cwinpy_knope``) is also provided. These tools are all
available through command line exectubles and through a Python API. The pipelines can generate
`HTCondor <https://htcondor.readthedocs.io/en/latest/>`_ `DAGs
<https://htcondor.readthedocs.io/en/latest/users-manual/dagman-workflows.html>`_ to run analyses
over long observing runs for multiple detectors and multiple sources on a computer cluster or via the
Open Science Grid. The pipelines can be used on both open data (as hosted by the `Gravitational-wave
Open Science Center <https://www.gw-openscience.org/data/>`_) and proprietary data from the LIGO and
Virgo detectors.

In addition to these main functions, CWInPy enables users to:

* simulate processed signals from :ref:`individual sources<Simulating a signal>`, or
  :ref:`populations<Pulsar simulations>` of sources;
* perform :ref:`hierarchical inference<Hierarchical analyses>` on the underlying ellipticity/mass
  quadrupole distribution for multiple sources.

Quick links
-----------

To help get started some useful links are:

* :ref:`Installing CWInPy<Installation>`
* :ref:`An example of heterodyning data<Example: two simulated pulsar signals>`
* :ref:`An example of estimating a signal's parameters<Example: single detector data>`

Contributing
------------

CWInPy is open source and anyone is welcome to contribute. The development repository of CWInPy is
currently not public, however the repository is mirrored on `Github
<https://github.com/cwinpy/cwinpy>`_. `Issues <https://github.com/cwinpy/cwinpy/issues>`_,
`discussions <https://github.com/cwinpy/cwinpy/discussions>`_ or `pull requests
<https://github.com/cwinpy/cwinpy/pulls>`_ can be opened in that Github repository, or can be
emailed directly to `contact+cw-software-cwinpy-3315-issue-@support.ligo.org
<mailto:contact+cw-software-cwinpy-3315-issue-@support.ligo.org>`_.

Code of conduct
~~~~~~~~~~~~~~~

Contributors to CWInPy and users of any of the discussion forums are expected to treat each other
with respect and abide by the guidelines of the `Python Community Code of Conduct
<https://www.python.org/psf/conduct/>`_.

.. automodule:: cwinpy
   :members:

.. toctree::
   :maxdepth: 1
   :hidden:

   Installation <installation>
   Analysis pipelines <pipelines>
   API interface <api>
   Validation <comparisons/comparisons>
