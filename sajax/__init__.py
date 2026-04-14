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

LdcMode
    Type alias for supported limb-darkening laws.

ArOverlapMode
    Type alias for active-region overlap resolution rules.
"""

from .core import (
    compute_light_curve,
    build_model,
    evaluate_light_curve,
    build_stellar_grid,
    LdcMode,
    ArOverlapMode,
)
from .geometry import rotate_active_region

__version__ = "0.2.2"
__all__ = [
    "build_stellar_grid",
    "build_model",
    "evaluate_light_curve",
    "compute_light_curve",
    "rotate_active_region",
    "LdcMode",
    "ArOverlapMode",
]