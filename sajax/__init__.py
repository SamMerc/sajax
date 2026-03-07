"""
SAJAX — Stellar Activity Grid for Exoplanets in JAX.

Public API
----------
compute_light_curve
    Main entry point: compute spectroscopic stellar-contamination light
    curves for a spotted star over a set of rotational phases.

build_stellar_grid
    Pre-compute the static stellar pixel grid (useful when you want to
    inspect or reuse the grid independently).

rotate_active_region
    Apply stellar rotation and inclination to a Cartesian spot position.
"""

from .core import compute_light_curve, build_stellar_grid
from .geometry import rotate_active_region

__version__ = "0.1.0"
__all__ = [
    "compute_light_curve",
    "build_stellar_grid",
    "rotate_active_region",
]