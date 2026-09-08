Welcome to SAJAX's documentation!
=================================

SAJAX is a package that models contamination of exoplanet spectro-photometric time series by active regions (spots, faculae, and flares) on the stellar surface. Its core functionality builds on `SAGE <https://github.com/chakrah/sage>`_ (`Chakraborty et al. 2024 <https://www.aanda.org/articles/aa/abs/2024/05/aa47727-23/aa47727-23.html>`_), and it draws additional inspiration from `ANTARESS <https://gitlab.unige.ch/spice_dune/antaress>`_ (`Bourrier et al. 2024 <https://www.aanda.org/articles/aa/full_html/2024/11/aa49203-24/aa49203-24.html>`_).

The main innovation over SAGE is that SAJAX vectorises the spatial and spectral loops with ``jax.vmap``, making it fast on both CPU and GPU - with no change to the calling code - and fully differentiable, enabling gradient-based inference with tools like NumPyro or Optax.

Key Features
------------

- **Spectroscopic light curves** - provide a spectrum for the quiet star and the active region; SAJAX returns a light curve at every wavelength.
- **Multi-planet systems** - transits from several planets are modeled simultaneously, each with independent orbital and radius parameters.
- **Time-evolving active regions** - spot/faculae latitude, longitude, size, and contrast can vary over the course of an observation instead of staying fixed. This feature can be used to model flares (see Case 9 in :doc:`examples/introduction`).
- **Multiple limb-darkening modes** - provide coefficients for your favorite laws (``linear``, ``quadratic``, ``power2``, ``kipping3``, ``nonlinear4``) or a full intensity profile (``intensity_profile``).
- **Stellar rotation, inclination, and obliquity** - arbitrary rotational phases, stellar-axis inclinations, and a sky-projected spin-orbit angle (``sp_orb``) for aligned or misaligned transits are all supported.
- **JAX-native** - ``jit``, ``vmap``, and ``grad`` work out of the box.
- **pip-installable** - clean, modern packaging.

---------------

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   examples/introduction
   examples/inference
   examples/comparison


.. toctree::
   :maxdepth: 1
   :caption: Reference