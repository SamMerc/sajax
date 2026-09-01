"""
SAJAX — Stellar Activity Grid for Exoplanets in JAX.

Public API
----------
build_system
    Pre-build all static model arrays once before MCMC sampling. Takes
    ``times``/``P_rot`` to derive the stellar-rotation phase grid, plus an
    optional, all-or-nothing planetary-transit parameter group
    (``t0``/``period``/``a_over_rstar``/``inclination``/``k``), each scalar
    or carrying a trailing ``(nplanet,)`` axis for multiple planets (inferred
    from ``t0``); when occulting a starspot or facula, the planet mask is
    applied at the individual pixel level, so the resulting light-curve
    anomaly is computed correctly.

make_lc
    Pure JAX light curve evaluation, given a model from ``build_system``.
    Accepts JAX tracers. Compatible with jit, vmap, emcee_jax, and gradient-based samplers.

quick_lc
    Convenience wrapper: build_system + make_lc in one call.
    Use for one-off evaluations outside MCMC.

make_rv
    Pure JAX radial-velocity evaluation, given a model from ``build_system``.
    Accepts JAX tracers. Compatible with jit, vmap, emcee_jax, and gradient-based samplers.
    Accepts the same setup as ``make_lc``, plus ``planet_mass``/``stellar_mass``
    (for Keplerian RV semi-amplitude) and ``gamma`` (systemic velocity).

quick_rv
    Convenience wrapper: build_system + make_rv in one call.
    Use for one-off evaluations outside MCMC.

make_lc_and_rv
    Pure JAX light curve + radial-velocity evaluation together, given a
    model from ``build_system``. Accepts the identical parameter set as
    ``make_lc``/``make_rv`` combined. Shares the expensive per-pixel-
    per-wavelength flux computation between the two outputs instead of
    computing it twice, so calling this once is roughly 2x cheaper than
    calling ``make_lc`` and ``make_rv`` separately with the same
    arguments -- useful for joint transit-photometry + radial-velocity
    fits. Returns ``(lc, rv, star_maps)``.

quick_lc_and_rv
    Convenience wrapper: build_system + make_lc_and_rv in one call.
    Use for one-off evaluations outside MCMC.

build_stellar_grid
    Pre-compute the static stellar pixel grid.

rotate_active_region
    Apply stellar rotation and inclination to a Cartesian active region
    position.

_compute_planet_mask
    Compute the mask over stellar disc pixels: ``True`` where the pixel is occulted
    by one planet at this epoch.

_compute_all_planets_mask
    Combine ``_compute_planet_mask`` over ``nplanet`` planets, multiplicatively
    (planets are opaque occulters, unlike active regions' additive contrast).

LdMode
    Type alias for supported limb-darkening laws.
"""

from .core import (
    build_system,
    make_lc,
    quick_lc,
    make_rv,
    quick_rv,
    make_lc_and_rv,
    quick_lc_and_rv,
    build_stellar_grid,
    LdMode,
)
from .geometry import rotate_active_region

from .planet import _compute_planet_mask, _compute_all_planets_mask

from importlib.metadata import version
__version__ = version("sajax")
__all__ = [
    "build_stellar_grid",
    "build_system",
    "make_lc",
    "quick_lc",
    "make_rv",
    "quick_rv",
    "make_lc_and_rv",
    "quick_lc_and_rv",
    "rotate_active_region",
    "_compute_planet_mask",
    "_compute_all_planets_mask",
    "LdMode",
]