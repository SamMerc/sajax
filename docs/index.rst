Welcome to SAJAX's documentation!
=================================

SAJAX is a JAX-accelerated reimplementation of `SAGE <https://github.com/chakrah/sage>`_ 
(`Chakraborty et al. 2024 <https://www.aanda.org/articles/aa/abs/2024/05/aa47727-23/aa47727-23.html>`_),
a code that models stellar contamination of exoplanet transmission spectra from active regions 
(ars, faculae) on the stellar surface.

The key innovation over plain SAGE is that SAJAX vectorises the spectral loop with ``jax.vmap``, 
making it fast on both CPU and GPU without any change to the calling code, and fully differentiable — 
enabling gradient-based inference with tools like NumPyro or Optax.

Key Features
------------

- **Spectroscopic light curves** — provide a spectrum for the quiet star and the active region; SAJAX returns a light curve at every wavelength.
- **Multiple limb-darkening modes** — provide coefficients for your favorite laws (``linear``, ``quadratic``, ``power2``, ``kipping3``, ``nonlinear4``) or a full intensity profile (``intensity_profile``).
- **Stellar rotation + inclination** — arbitrary rotational phases and stellar-axis inclinations are supported.
- **JAX-native** — ``jit``, ``vmap``, and ``grad`` work out of the box.
- **pip-installable** — clean, modern packaging.

Get Started
---------------

.. toctree::
   :maxdepth: 2
   :caption: Get Started:

   installation
   usage
   tutorials/sajax_quickstart

Reference
-------------

.. toctree::
   :maxdepth: 1
   :caption: Reference:

   modules