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
    Pure JAX evaluation — accepts JAX tracers, compatible with
    jit, vmap, emcee_jax, and gradient-based samplers.

quick_lc
    Convenience wrapper: build_system + make_lc in one call.
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
    quick_lc,
    build_system,
    make_lc,
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
    "rotate_active_region",
    "_compute_planet_mask",
    "_compute_all_planets_mask",
    "LdMode",
]