"""
SAJAX — Stellar Activity Grid for Exoplanets in JAX.

Public API
----------
build_model
    Pre-build all static model arrays once before MCMC sampling.

evaluate_light_curve
    Pure JAX evaluation — accepts JAX tracers, compatible with
    jit, vmap, emcee_jax, and gradient-based samplers.

compute_light_curve
    Convenience wrapper: build_model + evaluate_light_curve in one call.
    Use for one-off evaluations outside MCMC.

build_stellar_grid
    Pre-compute the static stellar pixel grid.

rotate_active_region
    Apply stellar rotation and inclination to a Cartesian active region
    position.

_compute_planet_mask
    Compute the mask over stellar disc pixels: ``True`` where the pixel is occulted
    by the planet at this epoch.

LdcMode
    Type alias for supported limb-darkening laws.

ArOverlapMode
    Type alias for active-region overlap resolution rules.
"""

from .core import (
    compute_light_curve,
    compute_combined_light_curve,
    build_model,
    build_combined_model,
    evaluate_light_curve,
    build_stellar_grid,
    LdcMode,
    ArOverlapMode,
)
from .geometry import rotate_active_region

from .planet import _compute_planet_mask

from importlib.metadata import version
__version__ = version("sajax")
__all__ = [
    "build_stellar_grid",
    "build_model",
    "build_combined_model",
    "evaluate_light_curve",
    "compute_light_curve",
    "compute_combined_light_curve",
    "rotate_active_region",
    "_compute_planet_mask",
    "LdcMode",
    "ArOverlapMode",
]