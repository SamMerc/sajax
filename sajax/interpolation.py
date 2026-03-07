"""
interpolation.py — Spectral interpolation helpers built on JAX.

Replaces ``scipy.interpolate.interp1d`` from the original SAGE code
with ``jnp.interp``, which is differentiable and JIT-compilable.
"""

import jax.numpy as jnp


def interp_spectrum(
    wavelengths: jnp.ndarray,
    fluxes: jnp.ndarray,
    query: jnp.ndarray,
) -> jnp.ndarray:
    """
    Linearly interpolate a stellar spectrum onto new wavelength points.

    Wavelengths outside the tabulated range return 0.0, matching the
    original SAGE behaviour (``fill_value=0.0``).

    Parameters
    ----------
    wavelengths : jnp.ndarray, shape (N,)
        Tabulated wavelength axis — must be sorted in ascending order.
    fluxes : jnp.ndarray, shape (N,)
        Tabulated flux values corresponding to ``wavelengths``.
    query : jnp.ndarray, shape (M,)
        Wavelength points at which to evaluate the spectrum.

    Returns
    -------
    jnp.ndarray, shape (M,)
        Interpolated flux values.  Values outside [wavelengths[0],
        wavelengths[-1]] are set to 0.0.
    """
    return jnp.interp(query, wavelengths, fluxes, left=0.0, right=0.0)