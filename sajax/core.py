"""
core.py — JAX-accelerated stellar spot light-curve engine.

This module is a complete rewrite of ``SAGE1/sage.py`` in JAX.

Key differences from the original NumPy/SciPy implementation
-------------------------------------------------------------
1. **No wavelength loop.**
   The original code iterated over wavelengths with a Python ``for`` loop.
   Here the entire spectral axis is handled by ``jax.vmap``, which maps
   the single-channel computation across all wavelengths in parallel.

2. **No scatter-index spot placement.**
   The original code located spot pixels via integer scatter indices
   (fancy indexing with ``.astype(int)``), which is not differentiable
   and incompatible with ``jit``.  SAJAX instead computes an analytic
   angular-distance mask over the full (n, n) pixel grid using
   ``jnp.where``, which is fully vectorised and differentiable.

3. **No class state mutation.**
   The original ``sage_class.rotate_star()`` mutated ``self.phases_rot``
   inside a loop — a latent bug.  SAJAX uses pure functions throughout.

4. **No astropy dependency for geometry.**
   Rotation matrices are implemented directly in JAX (see geometry.py).

5. **Differentiable end-to-end.**
   All operations are JAX-native, so ``jax.grad`` / ``jax.jacobian``
   work on the full pipeline — useful for gradient-based retrieval.
"""

from __future__ import annotations

import functools
from typing import Literal, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap

from .geometry import rotate_active_region

# Type alias
LdcMode = Literal["single", "multi-color", "intensity_profile"]


# ---------------------------------------------------------------------------
# 1. Grid construction  (NumPy — runs once per model configuration)
# ---------------------------------------------------------------------------

def build_stellar_grid(
    planet_pixel_size: int,
    radiusratio: float,
    semimajor: float,
    ve: float,
) -> dict:
    """
    Pre-compute the static stellar pixel grid.

    This is intentionally NumPy — it runs once before any JAX compilation
    and its output is passed as constant data into the jitted functions.

    Parameters
    ----------
    planet_pixel_size : int
        Number of pixels across the planetary disc radius.
    radiusratio : float
        Rp / Rs.
    semimajor : float
        a / Rs  (semi-major axis in stellar radii).
    ve : float
        Stellar equatorial velocity [km/s].

    Returns
    -------
    dict with keys:
        ``n``             — pixel grid side length
        ``star_pixel_rad``— stellar radius in pixels
        ``x_grid``        — (n, n) x pixel coordinates
        ``y_grid``        — (n, n) y pixel coordinates
        ``r_grid``        — (n, n) radial distance from disc centre
        ``starmask``      — (n, n) bool, True inside the stellar disc
        ``vel_grid``      — (n, n) Doppler velocity factor  (delta_v / c)
        ``mu_grid``       — (n, n) limb-darkening angle cos(θ)
        ``total_pixels``  — number of pixels inside the stellar disc
    """
    C = 299_792.458  # speed of light [km/s]

    rs = 1.0 / semimajor
    rp = radiusratio * rs
    star_pixel_rad = (rs / rp) * planet_pixel_size
    n = int(2.0 * planet_pixel_size * (rs / rp) + 2.0 * planet_pixel_size)

    x = np.arange(n) - n / 2.0
    y = np.arange(n) - n / 2.0

    x1, y1 = np.meshgrid(x, y, sparse=True)
    r_grid = np.sqrt(x1 ** 2 + y1 ** 2)

    # x2, y2 are pixel coordinates normalised by the stellar radius
    x2, y2 = np.meshgrid(x / star_pixel_rad, y / star_pixel_rad)
    starmask = (x2 ** 2 + y2 ** 2) <= 1.0

    # Velocity-broadening map:  Δv/c = (y / R_star) * (ve / c)
    vel_grid = np.where(starmask, y2 * (ve / C), 0.0).astype(np.float32)

    # mu = cos θ  for each stellar pixel  (0 outside the disc)
    mu_grid = np.where(
        starmask,
        np.cos(np.arcsin(np.clip(r_grid / star_pixel_rad, 0.0, 1.0))),
        0.0,
    ).astype(np.float32)

    return dict(
        n=n,
        star_pixel_rad=float(star_pixel_rad),
        x_grid=(x2 * star_pixel_rad).astype(np.float32),
        y_grid=(y2 * star_pixel_rad).astype(np.float32),
        r_grid=r_grid.astype(np.float32),
        starmask=starmask,
        vel_grid=vel_grid,
        mu_grid=mu_grid,
        total_pixels=int(starmask.sum()),
    )


# ---------------------------------------------------------------------------
# 2. Spot mask  (JAX — called once per phase, not per wavelength)
# ---------------------------------------------------------------------------

def _compute_spot_mask(
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    star_pixel_rad: float,
    spx: float,
    spy: float,
    spz: float,
    spotsize_rad: float,
) -> jnp.ndarray:
    """
    Boolean mask: True for every grid pixel that is both inside the
    stellar disc and inside the spot.

    This replaces the original SAGE scatter-index approach with a
    vectorised angular-distance computation over the entire (n, n) grid.

    Parameters
    ----------
    x_grid, y_grid : (n, n)
        Pixel coordinate grids in stellar-pixel units.
    star_pixel_rad : float
        Stellar radius in pixels.
    spx, spy, spz : float
        Spot-centre Cartesian coordinates (after rotation + inclination).
    spotsize_rad : float
        Spot angular radius in radians.

    Returns
    -------
    jnp.ndarray, shape (n, n), dtype bool
    """
    r2 = x_grid ** 2 + y_grid ** 2
    in_star = r2 <= star_pixel_rad ** 2

    # z-coordinate of each pixel projected onto the stellar sphere
    z_grid = jnp.sqrt(jnp.maximum(star_pixel_rad ** 2 - r2, 0.0))

    # Spot centre in spherical coordinates (after rotation)
    spotlong_rot = jnp.arctan2(spx, spz)
    spotlat_rot  = jnp.arccos(jnp.clip(spy / star_pixel_rad, -1.0, 1.0))

    # Each grid pixel in spherical coordinates
    longi = jnp.arctan2(x_grid, z_grid)
    lati  = jnp.arccos(jnp.clip(y_grid / star_pixel_rad, -1.0, 1.0))

    # Great-circle angular distance between spot centre and each pixel
    delta_lon = jnp.abs(spotlong_rot - longi)
    d_sigma = jnp.arccos(jnp.clip(
        jnp.cos(spotlat_rot) * jnp.cos(lati)
        + jnp.sin(spotlat_rot) * jnp.sin(lati) * jnp.cos(delta_lon),
        -1.0, 1.0,
    ))

    # Spots whose centre has spz < 0 are on the far side of the star
    visible = spz >= 0.0
    in_spot = (d_sigma <= spotsize_rad) & visible

    return in_star & in_spot


# ---------------------------------------------------------------------------
# 3. Single-wavelength flux  (vmapped over the spectral axis)
# ---------------------------------------------------------------------------

def _flux_at_wavelength(
    # --- vmapped: one scalar/slice per wavelength channel ---
    flux_hot_wl: float,
    flux_cold_wl: float,
    u1_wl: float,
    u2_wl: float,
    I_prof_wl: jnp.ndarray,         # (n_mu_pts,)  intensity profile at λ
    # --- broadcast: shared across all wavelength channels ---
    starmask: jnp.ndarray,           # (n, n)
    mu_grid: jnp.ndarray,            # (n, n)
    vel_grid: jnp.ndarray,           # (n, n)
    total_pixels: int,
    spot_masks: jnp.ndarray,         # (nspot, n, n)
    mu_profile_pts: jnp.ndarray,     # (n_mu_pts,)
    ldc_mode: LdcMode,
) -> tuple[float, float, jnp.ndarray]:
    """
    Compute integrated flux for a single wavelength channel.

    Parameters
    ----------
    (See docstring of ``compute_light_curve`` for parameter descriptions.)

    Returns
    -------
    star_spec   : float  — unspotted disc-integrated flux (normalised)
    total_flux  : float  — spotted  disc-integrated flux (normalised)
    star_grid   : (n, n) — per-pixel flux map at this wavelength
    """
    # ---- unspotted stellar grid ----------------------------------------
    # Apply Doppler broadening:  F_obs = F_rest * (1 + v/c)
    hot_raw = flux_hot_wl * (1.0 + vel_grid)

    # Limb darkening — branch resolved at JAX trace time (not runtime)
    if ldc_mode == "intensity_profile":
        ldc = jnp.interp(mu_grid, mu_profile_pts, I_prof_wl,
                         left=0.0, right=0.0)
    else:
        # Quadratic law:  I(μ) = 1 - u1(1-μ) - u2(1-μ)²
        ldc = 1.0 - u1_wl * (1.0 - mu_grid) - u2_wl * (1.0 - mu_grid) ** 2

    star_grid = jnp.where(starmask, hot_raw * ldc, 0.0)
    star_spec  = jnp.sum(star_grid) / total_pixels

    # ---- cold-region (spot) grid ----------------------------------------
    # The spot uses the same LDC profile as the surrounding photosphere.
    cold_raw  = flux_cold_wl * (1.0 + vel_grid)
    cold_grid = jnp.where(starmask, cold_raw * ldc, 0.0)

    # ---- apply each spot in sequence ------------------------------------
    # For each spot mask we replace the hot-photosphere pixels with cold
    # ones.  lax.scan keeps this differentiable and avoids Python loops.
    def _apply_one_spot(carry_grid: jnp.ndarray,
                         spot_mask: jnp.ndarray) -> tuple:
        updated = jnp.where(spot_mask, cold_grid, carry_grid)
        return updated, None

    star_grid, _ = jax.lax.scan(_apply_one_spot, star_grid, spot_masks)

    total_flux = jnp.sum(star_grid) / total_pixels
    return star_spec, total_flux, star_grid


# ---------------------------------------------------------------------------
# 4. Single-phase computation
# ---------------------------------------------------------------------------

def _compute_single_phase(
    spot_cart_all: jnp.ndarray,      # (nspot, 3) — only argument per phase
    *,
    # everything below is baked in via functools.partial
    wavelength: jnp.ndarray,         # (nwave,)
    flux_hot_interp: jnp.ndarray,    # (nwave,)
    flux_cold_interp: jnp.ndarray,   # (nwave,)
    u1: jnp.ndarray,                 # (nwave,)
    u2: jnp.ndarray,                 # (nwave,)
    I_profile: jnp.ndarray,          # (nwave, n_mu_pts)
    mu_profile_pts: jnp.ndarray,     # (n_mu_pts,)
    x_grid: jnp.ndarray,             # (n, n)
    y_grid: jnp.ndarray,             # (n, n)
    starmask: jnp.ndarray,           # (n, n)
    mu_grid: jnp.ndarray,            # (n, n)
    vel_grid: jnp.ndarray,           # (n, n)
    star_pixel_rad: float,
    total_pixels: int,
    spotsize_rads: jnp.ndarray,      # (nspot,)
    ldc_mode: LdcMode,
    plot_map_wavelength: float,
) -> tuple[float, jnp.ndarray, jnp.ndarray]:
    """
    Full spectral computation for a single rotational phase.

    Returns
    -------
    flux_norm            : float
    contamination_factor : (nwave,) array
    star_map             : (n, n) flux map at ``plot_map_wavelength``
    """
    # ---- Spot masks: (nspot, n, n) — computed once per phase ------------
    # vmap _compute_spot_mask over the spot axis
    spot_masks = vmap(
        lambda cart, sr: _compute_spot_mask(
            x_grid, y_grid, star_pixel_rad,
            cart[0], cart[1], cart[2], sr,
        )
    )(spot_cart_all, spotsize_rads)        # → (nspot, n, n)

    # ---- vmap _flux_at_wavelength over the spectral axis ----------------
    _flux_vmap = vmap(
        functools.partial(
            _flux_at_wavelength,
            starmask=starmask,
            mu_grid=mu_grid,
            vel_grid=vel_grid,
            total_pixels=total_pixels,
            spot_masks=spot_masks,
            mu_profile_pts=mu_profile_pts,
            ldc_mode=ldc_mode,
        ),
        in_axes=(0, 0, 0, 0, 0),           # vmap over first axis (wavelength)
    )

    star_specs, bin_fluxes, star_grids = _flux_vmap(
        flux_hot_interp,   # (nwave,)
        flux_cold_interp,  # (nwave,)
        u1,                # (nwave,)
        u2,                # (nwave,)
        I_profile,         # (nwave, n_mu_pts)
    )

    # ---- Broadband flux normalisation -----------------------------------
    # Sum over wavelengths, then normalise spotted by unspotted
    flux_norm = jnp.sum(bin_fluxes) / jnp.sum(star_specs)

    # ---- Contamination factor per wavelength ----------------------------
    # ε(λ) = F_unspotted(λ) / F_spotted(λ)
    contamination_factor = star_specs / jnp.where(bin_fluxes == 0.0, 1.0,
                                                   bin_fluxes)

    # ---- Stellar map at the requested wavelength ------------------------
    map_idx  = jnp.argmin(jnp.abs(wavelength - plot_map_wavelength))
    star_map = star_grids[map_idx]

    return flux_norm, contamination_factor, star_map


# ---------------------------------------------------------------------------
# 5. Public API
# ---------------------------------------------------------------------------

def compute_light_curve(
    wavelength: np.ndarray,
    flux_hot: np.ndarray,
    flux_cold: np.ndarray,
    params: dict,
    spot_lat: np.ndarray,
    spot_long: np.ndarray,
    spot_size: np.ndarray,
    phases_rot: np.ndarray,
    planet_pixel_size: int,
    ve: float,
    ldc_mode: LdcMode = "multi-color",
    plot_map_wavelength: Optional[float] = None,
) -> dict:
    """
    Compute spectroscopic stellar-contamination light curves for a spotted
    star over an arbitrary set of rotational phases.

    This is the main entry point for SAJAX.  It replaces the original
    ``sage_class.rotate_star()`` method.

    Parameters
    ----------
    wavelength : array_like, shape (nwave,)
        Wavelength array.  Any consistent unit is accepted; the same unit
        must be used for ``plot_map_wavelength``.
    flux_hot : array_like, shape (nwave,)
        Quiet-photosphere flux spectrum sampled at ``wavelength``.
    flux_cold : array_like, shape (nwave,)
        Active-region (spot) flux spectrum sampled at ``wavelength``.
    params : dict
        Stellar and orbital parameters:

        * ``radiusratio``  — Rp / Rs
        * ``semimajor``    — a / Rs
        * ``u1``, ``u2``  — quadratic limb-darkening coefficients
          (scalar, broadcast to all wavelengths for ``multi-color``).
        * ``mu_profile``  — 1-D array of μ sampling points, required for
          ``intensity_profile`` mode.
        * ``I_profile``   — 2-D array, shape (nwave, n_mu), required for
          ``intensity_profile`` mode.
        * ``inc_star``    — stellar inclination in degrees
          (default 90 = equator-on).

    spot_lat : array_like, shape (nspot,)
        Spot latitudes in degrees (−90 to +90).
    spot_long : array_like, shape (nspot,)
        Spot longitudes in degrees.
    spot_size : array_like, shape (nspot,)
        Spot angular radii in degrees.
    phases_rot : array_like, shape (nphase,)
        Rotational phases in degrees at which to evaluate the model.
    planet_pixel_size : int
        Number of pixels across the planetary disc radius.  Controls the
        spatial resolution of the grid.  Values of 15–50 are typical.
    ve : float
        Stellar equatorial velocity in km/s.
    ldc_mode : str
        Limb-darkening mode.  One of:

        * ``"single"``          — no LDC (u1 = u2 = 0).
        * ``"multi-color"``     — single (u1, u2) pair for all wavelengths.
        * ``"intensity_profile"`` — per-wavelength intensity profile.

    plot_map_wavelength : float, optional
        Wavelength at which to return the stellar flux map.  Defaults to
        the midpoint of ``wavelength``.

    Returns
    -------
    dict with keys:

    ``lc``
        ndarray, shape (nphase,).  Broadband integrated flux normalised
        by its median.  1.0 = unspotted level.
    ``epsilon``
        ndarray, shape (nphase, nwave).  Contamination factor
        ε(λ) = F_unspotted(λ) / F_spotted(λ) per phase per wavelength.
    ``star_maps``
        ndarray, shape (nphase, n, n).  Stellar flux map at
        ``plot_map_wavelength`` for each phase.
    """
    # ---- Input coercion -------------------------------------------------
    wavelength  = np.asarray(wavelength,  dtype=np.float32)
    flux_hot    = np.asarray(flux_hot,    dtype=np.float32)
    flux_cold   = np.asarray(flux_cold,   dtype=np.float32)
    phases_rot  = np.atleast_1d(np.asarray(phases_rot, dtype=np.float32))
    spot_lat    = np.atleast_1d(np.asarray(spot_lat,   dtype=np.float32))
    spot_long   = np.atleast_1d(np.asarray(spot_long,  dtype=np.float32))
    spot_size   = np.atleast_1d(np.asarray(spot_size,  dtype=np.float32))

    nwave = len(wavelength)
    nspot = len(spot_lat)

    # ---- Unpack params --------------------------------------------------
    radiusratio    = float(params["radiusratio"])
    semimajor      = float(params["semimajor"])
    inc_star       = float(params.get("inc_star", 90.0))
    u1_in          = params.get("u1", 0.0)
    u2_in          = params.get("u2", 0.0)
    mu_profile_pts = np.asarray(params.get("mu_profile", [0.0, 1.0]),
                                dtype=np.float32)
    # Default I_profile: uniform intensity (no wavelength-dependent LDC)
    I_profile = np.asarray(
        params.get("I_profile",
                   np.ones((nwave, len(mu_profile_pts)), dtype=np.float32)),
        dtype=np.float32,
    )

    # ---- Limb-darkening arrays  (nwave,) --------------------------------
    if ldc_mode == "single":
        u1 = np.zeros(nwave, dtype=np.float32)
        u2 = np.zeros(nwave, dtype=np.float32)
    elif ldc_mode == "multi-color":
        u1 = np.full(nwave, float(u1_in), dtype=np.float32)
        u2 = np.full(nwave, float(u2_in), dtype=np.float32)
    else:  # intensity_profile — u1/u2 unused but kept for uniform signature
        u1 = np.zeros(nwave, dtype=np.float32)
        u2 = np.zeros(nwave, dtype=np.float32)

    # ---- Build static stellar grid (NumPy, runs once) -------------------
    grid = build_stellar_grid(planet_pixel_size, radiusratio, semimajor, ve)

    if plot_map_wavelength is None:
        plot_map_wavelength = float(wavelength[nwave // 2])

    # ---- Convert spot parameters to Cartesian pixel coordinates ---------
    # Co-latitude = 90° - latitude
    spot_colat_rad = np.deg2rad(90.0 - spot_lat)
    spot_long_rad  = np.deg2rad(spot_long)
    spr            = grid["star_pixel_rad"]

    spot_cart_pixel = np.stack([
        spr * np.sin(spot_long_rad) * np.sin(spot_colat_rad),  # x
        spr * np.cos(spot_colat_rad),                           # y
        spr * np.cos(spot_long_rad) * np.sin(spot_colat_rad),  # z
    ], axis=-1).astype(np.float32)  # (nspot, 3)

    spotsize_rads = jnp.asarray(np.deg2rad(spot_size), dtype=jnp.float32)

    # ---- JIT-compile the phase function with all constants baked in -----
    _phase_fn = jit(functools.partial(
        _compute_single_phase,
        wavelength         = jnp.asarray(wavelength),
        flux_hot_interp    = jnp.asarray(flux_hot),
        flux_cold_interp   = jnp.asarray(flux_cold),
        u1                 = jnp.asarray(u1),
        u2                 = jnp.asarray(u2),
        I_profile          = jnp.asarray(I_profile),
        mu_profile_pts     = jnp.asarray(mu_profile_pts),
        x_grid             = jnp.asarray(grid["x_grid"]),
        y_grid             = jnp.asarray(grid["y_grid"]),
        starmask           = jnp.asarray(grid["starmask"]),
        mu_grid            = jnp.asarray(grid["mu_grid"]),
        vel_grid           = jnp.asarray(grid["vel_grid"]),
        star_pixel_rad     = grid["star_pixel_rad"],
        total_pixels       = grid["total_pixels"],
        spotsize_rads      = spotsize_rads,
        ldc_mode           = ldc_mode,
        plot_map_wavelength= float(plot_map_wavelength),
    ))

    # ---- Loop over rotational phases ------------------------------------
    # The first call triggers JIT compilation; subsequent calls are fast.
    lc_list   = []
    eps_list  = []
    maps_list = []

    for phase in phases_rot:
        # Rotate all spot positions for this phase
        rotated = np.stack([
            np.asarray(rotate_active_region(
                jnp.asarray(spot_cart_pixel[s]),
                phase_deg=float(phase),
                inc_deg=inc_star,
            ))
            for s in range(nspot)
        ]).astype(np.float32)  # (nspot, 3)

        flux_norm, contamination, star_map = _phase_fn(
            jnp.asarray(rotated)
        )
        lc_list.append(float(flux_norm))
        eps_list.append(np.array(contamination))
        maps_list.append(np.array(star_map))

    lc = np.array(lc_list)
    # Normalise by the median (same logic as original rotate_star)
    if len(lc) > 1:
        lc = lc / np.median(lc)

    return {
        "lc"        : lc,
        "epsilon"   : np.array(eps_list),
        "star_maps" : np.array(maps_list),
    }