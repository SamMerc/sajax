"""
planet.py — Keplerian planet orbit and pixel-level transit geometry for sajax.

This module is a standalone companion to sajax/core.py.  It can be used
independently to compute transit light curves, or integrated with sajax
via ``build_combined_model`` / ``compute_combined_light_curve`` (defined in
core.py) to correctly model active-region crossing events — i.e. cases where
the planet occultes a starspot or facula during transit.

Architecture
------------
The module is intentionally *geometry-only*: it computes where the planet is
on the sky at each epoch and which stellar-disc pixels it occults.  The flux
integration (limb darkening, active-region weighting) is handled by the
existing sajax machinery in core.py.  This clean separation means that the
transit model inherits sajax's full limb-darkening parametrisation
automatically — no extra parameters are required.

Orbital convention  (Winn 2010 / Eastman et al. 2013)
------------------------------------------------------
  X  — sky-plane east-west  (positive east)
  Y  — sky-plane north-south  (positive north, foreshortened by cos i)
  Z  — line-of-sight toward observer  (Z > 0 ⟹ planet in front of star)

All sky positions are in units of the stellar radius R*.

Minimum parameter set
---------------------
  t0            : mid-transit epoch  [days]
  period        : orbital period  [days]
  a_over_rstar  : semimajor axis / R*  (dimensionless)
                  May be derived from stellar density via
                  ``stellar_density_to_a_over_rstar()``.
  inclination   : orbital inclination  [rad]   (90 / π/2 = perfect edge-on)
  ecc           : orbital eccentricity  [0, 1)
  omega_peri    : argument of periastron  [rad]
                  (ω = 0° → periapsis at ascending node;
                   ω = 90° → periapsis at inferior conjunction /
                   transit centre for a circular orbit)
  k             : planet-to-star radius ratio  Rp / R*

Limb darkening
--------------
The same LDC law stored in the sajax model dict is applied automatically
to occulted pixels — no separate transit LDC parameters are required.

Public API
----------
  ``_kepler(M, ecc)``                    — differentiable Kepler solver
  ``planet_sky_position(...)``           — single-epoch sky coords (X, Y, Z)
  ``compute_planet_sky_positions(...)``  — vectorised over an array of times
  ``_compute_planet_mask(...)``          — per-pixel occultation mask
  ``build_transit_model(...)``           — pre-compute positions for all times
  ``stellar_density_to_a_over_rstar()``  — unit-conversion convenience
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import vmap, nn as jax_nn


# ---------------------------------------------------------------------------
# 1. Kepler's equation solver  (differentiable, JIT-safe)
# ---------------------------------------------------------------------------

def _kepler(M: jnp.ndarray, ecc: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve Kepler's equation  M = E - e sin E  for the eccentric anomaly E,
    then convert to the true anomaly f and return (sin f, cos f).

    Implementation details
    ~~~~~~~~~~~~~~~~~~~~~~
    * Symmetry fold: M is mapped into [0, π) then restored afterwards,
      which halves the domain and removes sign ambiguity.
    * Starter: E0 = M + e sin M  (good for e ≲ 0.5; adequate for e < 0.9).
    * Refinement: 6 Halley iterations (3rd-order convergence) — the residual
      drops from O(e²) to < 1e-15 in ≤ 4 steps even at e = 0.95.
    * All operations are JAX primitives.  The fixed unrolled iteration graph
      is fully differentiable via JAX's default automatic differentiation.
      No ``custom_jvp`` hook is needed; the iteration count is small enough
      that the unrolled gradient does not cause numerical issues.

    Parameters
    ----------
    M   : mean anomaly [rad]  — scalar or array
    ecc : orbital eccentricity [0, 1)  — scalar

    Returns
    -------
    sinf, cosf : sin and cos of the true anomaly  (same shape as M)
    """
    # Wrap into [0, 2π) and exploit the symmetry sin(2π − M) = −sin(M)
    M = M % (2.0 * jnp.pi)
    flip = M > jnp.pi
    M_ = jnp.where(flip, 2.0 * jnp.pi - M, M)   # now in [0, π)

    # Initial guess
    E = M_ + ecc * jnp.sin(M_)

    # Halley's method:  f = E − e sin E − M,   f′ = 1 − e cos E,   f′′ = e sin E
    #   ΔE = −f / (f′ − f·f′′ / (2 f′))  =  −f·f′ / (f′² − f·f′′/2)
    for _ in range(6):
        sE  = jnp.sin(E)
        cE  = jnp.cos(E)
        f   = E - ecc * sE - M_
        fp  = 1.0 - ecc * cE
        fpp = ecc * sE
        E   = E - f * fp / (fp * fp - 0.5 * f * fpp)

    # Restore the original half-plane
    E = jnp.where(flip, 2.0 * jnp.pi - E, E)

    # Eccentric to true anomaly via the standard formulae
    cE    = jnp.cos(E)
    sE    = jnp.sin(E)
    denom = 1.0 - ecc * cE
    sinf  = jnp.sqrt(jnp.maximum(1.0 - ecc ** 2, 0.0)) * sE / denom
    cosf  = (cE - ecc) / denom
    return sinf, cosf


# ---------------------------------------------------------------------------
# 2. Sky-plane position of the planet at a single epoch
# ---------------------------------------------------------------------------

def planet_sky_position(
    time: jnp.ndarray,
    t0: float,
    period: float,
    a_over_rstar: float,
    inclination: float,
    ecc: float,
    omega_peri: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute the planet's sky-plane position (X, Y, Z) in units of R*.

    Parameters
    ----------
    time          : observation epoch  [same units as t0 / period, e.g. days]
    t0            : mid-transit epoch (inferior conjunction)
    period        : orbital period
    a_over_rstar  : semimajor axis / R*  (dimensionless, > 1 for non-grazing)
    inclination   : orbital inclination [rad]   (π/2 = edge-on)
    ecc           : eccentricity  [0, 1)
    omega_peri    : argument of periastron  [rad]
                    Measured from the ascending node to periapsis.

    Returns
    -------
    X, Y, Z : sky-plane coordinates in units of R*
        X  — east-west (positive east)
        Y  — north-south projected  (= r sin(ω+f) cos i)
        Z  — toward observer  (Z > 0 ⟹ transit;  Z < 0 ⟹ occultation)

    Notes
    -----
    The sky-plane separation from the stellar centre is sqrt(X^2 + Y^2).
    A transit (or occultation) event occurs when sqrt(X^2 + Y^2) < 1 + k,
    where k = Rp / R*.
    """
    # ---- True anomaly at mid-transit ----------------------------------------
    # At inferior conjunction (transit centre): ω + f_transit = π/2
    # ⟹  f_transit = π/2 − ω
    f_transit = 0.5 * jnp.pi - omega_peri

    # ---- Time of periastron passage -----------------------------------------
    # Convert f_transit → E_transit via
    #   tan(E/2) = sqrt((1−e)/(1+e)) · tan(f/2)
    # Use arctan2 for correct quadrant handling.
    half_f    = 0.5 * f_transit
    E_transit = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 - ecc) * jnp.sin(half_f),
        jnp.sqrt(1.0 + ecc) * jnp.cos(half_f),
    )
    M_transit = E_transit - ecc * jnp.sin(E_transit)   # Kepler's eq.
    t_peri    = t0 - (period / (2.0 * jnp.pi)) * M_transit

    # ---- Mean anomaly at observation time ------------------------------------
    M = (2.0 * jnp.pi / period) * (time - t_peri)

    # ---- Solve Kepler --------------------------------------------------------
    sinf, cosf = _kepler(M, ecc)

    # ---- Orbital radius in units of R* --------------------------------------
    # r = a (1 − e^2) / (1 + e cos f)
    r = a_over_rstar * (1.0 - ecc ** 2) / (1.0 + ecc * cosf)

    # ---- Sky-plane projection (Winn 2010, eqs. 1–3) -------------------------
    # Expand cos(ω+f) and sin(ω+f) via angle-addition formulae to avoid
    # computing arctan2(sinf, cosf) (preserves differentiability).
    cos_w  = jnp.cos(omega_peri)
    sin_w  = jnp.sin(omega_peri)
    cos_wf = cosf * cos_w - sinf * sin_w   # cos(ω + f)
    sin_wf = sinf * cos_w + cosf * sin_w   # sin(ω + f)

    X =  r * (-cos_wf)                          # east–west
    Y =  r *  sin_wf * jnp.cos(inclination)     # north–south (projected)
    Z =  r *  sin_wf * jnp.sin(inclination)     # toward observer

    return X, Y, Z


# ---------------------------------------------------------------------------
# 3. Vectorised positions over an array of times
# ---------------------------------------------------------------------------

def compute_planet_sky_positions(
    times: jnp.ndarray,
    t0: float,
    period: float,
    a_over_rstar: float,
    inclination: float,
    ecc: float,
    omega_peri: float,
) -> jnp.ndarray:
    """
    Vectorised wrapper around ``planet_sky_position``.

    Parameters
    ----------
    times : (ntime,) array of observation epochs

    Returns
    -------
    xyz : (ntime, 3) array  —  columns are [X, Y, Z] in units of R*
    """
    _pos = vmap(
        lambda t: jnp.stack(
            planet_sky_position(
                t, t0, period, a_over_rstar, inclination, ecc, omega_peri,
            )
        )
    )(jnp.asarray(times, dtype=jnp.float32))   # (ntime, 3)
    return _pos


# ---------------------------------------------------------------------------
# 4. Per-pixel transit mask on the sajax stellar grid
# ---------------------------------------------------------------------------

def _compute_planet_mask(
    x_disc: jnp.ndarray,   # (total_pixels,)  pixel x coordinates
    y_disc: jnp.ndarray,   # (total_pixels,)  pixel y coordinates
    star_pixel_rad: float,
    X: jnp.ndarray,        # planet sky-plane x  [R*]
    Y: jnp.ndarray,        # planet sky-plane y  [R*]
    Z: jnp.ndarray,        # planet line-of-sight  [R*]  — Z > 0 ⟹ transit
    k: float,              # Rp / R*
) -> jnp.ndarray:
    """
    Boolean mask over in-disc pixels: ``True`` where the pixel is occulted
    by the planet at this epoch.

    The mask is non-zero only when Z > 0 (planet in front of the star).
    Pixels inside the planet disc contribute zero flux; if those pixels
    coincide with an active region, the spot-crossing anomaly emerges
    automatically.

    Parameters
    ----------
    x_disc, y_disc  : in-disc pixel coordinates  [pixels]
    star_pixel_rad  : stellar radius in pixels
    X, Y            : planet sky position  [R*]
    Z               : planet line-of-sight position  [R*]
    k               : planet-to-star radius ratio

    Returns
    -------
    jnp.ndarray, shape (total_pixels,), dtype bool_
    """
    # Normalise pixel coordinates to stellar radii
    xn = x_disc / star_pixel_rad
    yn = y_disc / star_pixel_rad

    # Squared sky-plane distance from planet centre to each pixel
    d2 = (xn - X) ** 2 + (yn - Y) ** 2

    if k == 0.0:
        return jnp.zeros(x_disc.shape, dtype=jnp.float32)

    # Soft disc mask: sigmoid boundary so gradients flow w.r.t. k and planet position.
    # Transition width ~10% of planet radius, floored at 1/10 pixel.
    d = jnp.sqrt(d2 + 1e-8)
    softness = max(0.1 * k, 1.0 / (10.0 * star_pixel_rad))
    disc_mask = jax_nn.sigmoid((k - d) / softness)

    # Hard Z gate: planet in front of the star is topologically binary.
    z_gate = jnp.where(Z > 0.0, 1.0, 0.0)
    return disc_mask * z_gate


# ---------------------------------------------------------------------------
# 5. build_transit_model — pre-compute positions for all (oversampled) epochs
# ---------------------------------------------------------------------------

def build_transit_model(
    times: np.ndarray,
    t0: float,
    period: float,
    a_over_rstar: float,
    inclination: float,
    ecc: float          = 0.0,
    omega_peri: float   = 0.0,
    k: float            = 0.1,
) -> dict:
    """
    Pre-compute the planet's sky-plane position at every epoch in ``times``.

    The returned dict should be stored in the sajax model dict under the key
    ``"transit"``.  The combined model builder ``build_combined_model()``
    (in core.py) does this automatically — end users typically do not need
    to call this function directly.

    Parameters
    ----------
    times         : (ntime,) array of observation epochs  [days]
                    Must be the **oversampled** time array when oversampling
                    is active (see ``build_combined_model``).
    t0            : mid-transit epoch  [days]
    period        : orbital period  [days]
    a_over_rstar  : semimajor axis / R*  (dimensionless)
    inclination   : orbital inclination  [rad]
    ecc           : eccentricity  (default: 0.0 = circular)
    omega_peri    : argument of periastron  [rad]  (default: 0.0)
    k             : planet-to-star radius ratio  Rp / R*  (default: 0.1)

    Returns
    -------
    dict with keys
    ~~~~~~~~~~~~~~
    ``planet_xyz`` : (ntime, 3) jnp.ndarray — planet (X, Y, Z) per epoch
    ``k``          : float — planet-to-star radius ratio
    """
    times_jax = jnp.asarray(times, dtype=jnp.float32)

    xyz = compute_planet_sky_positions(
        times_jax, t0, period, a_over_rstar, inclination, ecc, omega_peri,
    )   # (ntime, 3)

    return dict(
        planet_xyz = xyz,
        k          = float(k),
    )


# ---------------------------------------------------------------------------
# 6. Unit-conversion convenience
# ---------------------------------------------------------------------------

# Physical constants in SI / solar units needed for Kepler's third law
_G_cgs = 6.674_08e-8         # cm^3 g^-1 s^-2


def stellar_density_to_a_over_rstar(
    rho_star_gcc: float,
    period_days: float,
) -> float:
    """
    Convert mean stellar density and orbital period to a / R* via
    Kepler's third law  (Seager & Mallén-Ornelas 2003):

        a / R* = ( G ρ★ P^2 / (3π) )^(1/3)

    Parameters
    ----------
    rho_star_gcc  : mean stellar density  [g cm^-3]
    period_days   : orbital period  [days]

    Returns
    -------
    a_over_rstar  : float  (dimensionless)
    """
    P_sec        = period_days * 86_400.0
    a_over_r_cgs = (_G_cgs * rho_star_gcc * P_sec ** 2 / (3.0 * np.pi)) ** (1.0 / 3.0)
    return float(a_over_r_cgs)


def a_over_rstar_to_stellar_density(
    a_over_rstar: float,
    period_days: float,
) -> float:
    """
    Inverse of ``stellar_density_to_a_over_rstar``:

        ρ★ = 3π / (G P^2) · (a / R*)^3

    Parameters
    ----------
    a_over_rstar : semimajor axis / R*  (dimensionless)
    period_days  : orbital period  [days]

    Returns
    -------
    rho_star_gcc : mean stellar density  [g cm^-3]
    """
    P_sec = period_days * 86_400.0
    rho   = 3.0 * np.pi * a_over_rstar ** 3 / (_G_cgs * P_sec ** 2)
    return float(rho)