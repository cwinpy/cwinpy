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
the phase evolution of a given source (using a `Tempo
<https://en.wikipedia.org/wiki/Tempo_(astronomy)>`_-style parameter file); a class to store and
display this data is available. Following data processing, the package provides tools for inferring
the unknown gravitational-wave parameters describing the source and its orientation. An integrated
pipeline combining both these stages (``cwinpy_knope``) is also provided. These tools are all
available through command line executables and through a Python API. The pipelines can generate
`HTCondor <https://htcondor.readthedocs.io/en/latest/>`__ `DAGs
<https://htcondor.readthedocs.io/en/latest/users-manual/dagman-workflows.html>`_ to run analyses
over long observing runs for multiple detectors and multiple sources on a computer cluster or via
the Open Science Grid. The pipelines can be used on both open data (as hosted by the
`Gravitational-wave Open Science Center <https://gwosc.org/data/>`_) and proprietary
data from the LIGO and Virgo detectors.

In addition to these main functions, CWInPy enables users to:

* simulate processed signals from :ref:`individual sources<Simulating a signal>` or
  :ref:`populations<Pulsar simulations>` of sources;
* perform :ref:`hierarchical inference<Hierarchical analyses>` on the underlying ellipticity/mass
  quadrupole distribution for multiple sources.

Quick links
-----------

To help get started some useful links are:

* :ref:`Installing CWInPy<Installation>`
* :ref:`Running the cwinpy_knope_pipeline on real data<*knope* examples>`
* :ref:`Running the cwinpy_knope_pipeline on open data using "Quick setup"<Quick setup>`
* :ref:`An example of heterodyning data<Example: two simulated pulsar signals>`
* :ref:`An example of estimating a signal's parameters<Example: single detector data>`

Contributing
------------

CWInPy is open source and anyone is welcome to contribute. The development repository of CWInPy is
publicly available at `git.ligo.org/cwinpy/cwinpy <https://git.ligo.org/cwinpy/cwinpy>`_ and is
mirrored on `Github <https://github.com/cwinpy/cwinpy>`_. `Issues
<https://github.com/cwinpy/cwinpy/issues>`_, `discussions
<https://github.com/cwinpy/cwinpy/discussions>`_ or `pull requests
<https://github.com/cwinpy/cwinpy/pulls>`_ can be opened in that Github repository, or can be
emailed directly to `contact+cwinpy-cwinpy-3315-issue-@support.ligo.org
<mailto:contact+cwinpy-cwinpy-3315-issue-@support.ligo.org>`_.

Code of conduct
~~~~~~~~~~~~~~~

Contributors to CWInPy and users of any of the discussion forums are expected to treat each other
with respect and abide by the guidelines of the `Python Community Code of Conduct
<https://www.python.org/psf/conduct/>`_.

Citing CWInPy
-------------

If you use CWInPy for research that leads to a publication, I would be grateful if you cited the
`CWInPy paper <https://doi.org/10.21105/joss.04568>`_ in the
`Journal of Open Source Software <https://joss.theoj.org/>`_:

Pitkin, M., (2022). CWInPy: A Python package for inference with continuous gravitational-wave signals from pulsars. *Journal of Open Source Software*, 7(77), 4568, https://doi.org/10.21105/joss.04568

.. tab-set::

   .. tab-item:: BibTeX

      .. code-block:: bibtex

         @article{cwinpy,
             title = "{CWInPy: A Python package for inference with continuous gravitational-wave signals from pulsars}",
            author = {{Pitkin}, M.},
           journal = {Journal of Open Source Software},
            volume = 7,
            number = 77,
             pages = 4568,
              year = 2022,
               doi = {10.21105/joss.04568},
               url = {https://doi.org/10.21105/joss.04568},
         }

   .. tab-item:: RIS

      .. code-block:: text

         TY  - JOUR
         T1  - CWInPy: A Python package for inference with continuous gravitational-wave signals from pulsars
         AU  - Pitkin, Matthew
         JO  - Journal of Open Source Software
         VL  - 7
         IS  - 77
         SP  - 4568
         PY  - 2022
         DA  - 2022/09/29/
         DO  - 10.21105/joss.04568
         UR  - https://doi.org/10.21105/joss.04568
         ER  - 

   .. tab-item:: EndNode

      .. code-block:: text

         %0 Journal Article
         %T CWInPy: A Python package for inference with continuous gravitational-wave signals from pulsars
         %A Pitkin, Matthew
         %J Journal of Open Source Software
         %V 7
         %N 77
         %P 4568
         %D 2022
         %R 10.21105/joss.04568

   .. tab-item:: RefWorks

      .. code-block:: text

         RT Journal
         T1 CWInPy: A Python package for inference with continuous gravitational-wave signals from pulsars
         A1 Pitkin, Matthew
         JF Journal of Open Source Software
         VO 7
         SP 4568
         YR 2022
         DO DOI: 10.21105/joss.04568

.. automodule:: cwinpy
   :members:

.. toctree::
   :maxdepth: 1
   :hidden:

   Installation <installation>
   Analysis pipelines <pipelines>
   API interface <api>
   Validation <comparisons/comparisons>
