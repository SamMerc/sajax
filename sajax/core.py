"""
core.py -- JAX-accelerated stellar active region light-curve and radial
velocity engine.
"""

from __future__ import annotations

import functools
import warnings
from typing import Literal, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import interpax

from .geometry import rotate_active_region
from .planet import (
    _compute_all_planets_mask,
    compute_multi_planet_sky_positions,
    compute_multi_planet_keplerian_rv,
    keplerian_rv_semi_amplitude,
    _broadcast_orbital_param,
)

# Type alias
LdMode = Literal[
    "linear",          # 1 coeff  : u
    "quadratic",       # 2 coeffs : u1, u2
    "power2",          # 2 coeffs : c, alpha
    "kipping3",        # 3 coeffs : c1, c2, c3
    "nonlinear4",      # 4 coeffs : c1, c2, c3, c4
    "intensity_profile",
]

# Number of LDC coefficients expected per law (used for validation)
_N_COEFFS: dict[str, int] = {
    "linear":     1,
    "quadratic":  2,
    "power2":     2,
    "kipping3":   3,
    "nonlinear4": 4,
}

# Interpolation method for time-varying active-region parameters.
# Maps to interpax.interp1d's `method`:
#   "linear" -> "linear"  (piecewise-linear between the given times)
#   "cubic"  -> "cubic2"  (C2 natural cubic spline)
ArTimeInterp = Literal["linear", "cubic"]
_AR_TIME_INTERP_METHOD: dict[str, str] = {
    "linear": "linear", 
    "cubic": "cubic2"
}

# Numerical safeguards for the super-Gaussian AR shape (see _compute_ar_shape):
# _AR_SHAPE_TINY : floor to avoid log(0) at the AR-centre pixel
# _LOG_ARG_MAX   : cap on the log-space exponent so intermediate powers never
#  literally overflow to inf.
_AR_SHAPE_TINY = 1e-30
_LOG_ARG_MAX   = 50.0

# _FLUX_RATIO_EPS : Floor for the C_a = F_active/F_quiet contrast ratio's denominator
#  (see _flux_at_wavelength). Some limb-darkening laws (e.g. a user-supplied
# intensity_profile with I(mu=0)=0) can vanish at the limb, making the local
# F_quiet exactly 0 there; without a floor this causes NaN which corrupt arted_flux
# even though that pixel's contribution is supposed to be 0.
_FLUX_RATIO_EPS = 1e-12

# Speed of light [km/s], matching build_stellar_grid's own C constant and
# the ve/vel_col unit convention -- used by the RV engine (see
# _compute_single_phase_rv) to convert vel_col's v/c units to km/s.
_C_KMS = 299_792.458

# _RV_WEIGHT_EPS : floor under sum(flux_discs, axis=1) (per wavelength bin)
#  in _compute_single_phase_rv, same rationale as _FLUX_RATIO_EPS -- guards
#  a degenerate fully-dark or fully-occulted disc against a 0/0 -> NaN RV.
_RV_WEIGHT_EPS = 1e-20


# ---------------------------------------------------------------------------
# 1a. Grid construction  (NumPy -- runs once per model configuration)
# ---------------------------------------------------------------------------

def build_stellar_grid(
    stellar_grid_size: int,
    ve: float,
    inc_star: float = 90.0,
) -> dict:
    """
    Pre-compute the static stellar pixel grid, masked to the stellar disc.

    The mask is applied here once so that all downstream JAX functions
    receive 1D arrays containing only the in-disc pixels -- no starmask
    is ever passed around.

    Parameters
    ----------
    stellar_grid_size : int
        Stellar radius in pixels.  This is the single resolution knob:
        higher values give a finer grid at the cost of n^2 memory and
        compute.  Values of 100-300 are typical.
    ve : float
        Stellar **equatorial** velocity [km/s] -- v, not v*sin(i); the
        projected field applies its own ``sin(inc_star)``.
    inc_star : float, optional
        Stellar inclination in degrees (90 = equator-on, 0 = pole-on).
        Enters only through the ``sin(i)`` projection of the velocity field.

    Returns
    -------
    dict with keys
    ~~~~~~~~~~~~~~
    ``n``             - full grid side length (*always odd*)
    ``star_pixel_rad``- stellar radius in pixels (= stellar_grid_size)
    ``total_pixels``  - number of in-disc pixels
    ``flat_indices``  - (total_pixels,) int  indices into the flattened
                        (n, n) grid; used to reconstruct 2D maps at output
    ``x``             - (total_pixels,) x pixel coordinates       [in-disc only]
    ``y``             - (total_pixels,) y pixel coordinates       [in-disc only]
    ``mu``            - (total_pixels,) limb-darkening cos(theta) [in-disc only]
    ``col_idx``       - (total_pixels,) int, 0..n-1 -- which grid column each
                        in-disc pixel belongs to.
    ``vel_col``       - (n,) line-of-sight velocity in units of c (v/c) for
                        each possible column.
                        The sky-projected rotation axis lies in the y-z
                        plane, so the line-of-sight velocity depends only on
                        a pixel's column (its x-coordinate) -- every pixel in
                        a column shares one velocity, so it is never
                        materialised per-pixel.
    """
    C = 299_792.458  # speed of light [km/s]

    star_pixel_rad = float(stellar_grid_size)

    # n = 2 * radius + 1 so the centre falls on a pixel (forces odd grid)
    n = 2 * int(stellar_grid_size) + 1

    coords = np.arange(n) - n // 2   # e.g. -R, ..., -1, 0, 1, ..., R
    xg, yg = np.meshgrid(coords, coords)   # (n, n) each

    r2     = xg ** 2 + yg ** 2
    starmask = r2 <= star_pixel_rad ** 2

    # Apply mask -> 1D in-disc arrays
    flat_indices = np.flatnonzero(starmask)   # (total_pixels,)
    x_disc = xg.ravel()[flat_indices].astype(np.float32)
    y_disc = yg.ravel()[flat_indices].astype(np.float32)
    r_disc = np.sqrt(r2.ravel()[flat_indices]).astype(np.float32)

    # mu = cos(theta) = sqrt(1 - (r/R)^2), clamped for float32 safety
    mu_disc = np.sqrt(
        np.clip(1.0 - (r_disc / star_pixel_rad) ** 2, 0.0, 1.0)
    ).astype(np.float32)

    # coords[0] = -n//2, so col_idx = x + n//2 maps x onto {0, ..., n-1}.
    col_idx = (x_disc + n // 2).astype(np.int32)

    # Sky-frame spin axis (0, sin i, cos i) -> v_z = -(ve/R)*sin(i)*x, a function of x alone; +x recedes.
    vel_col = (
        coords / star_pixel_rad * (ve / C) * np.sin(np.deg2rad(inc_star))
    ).astype(np.float32)

    return dict(
        n             = n,
        star_pixel_rad= star_pixel_rad,
        total_pixels  = int(flat_indices.size),
        flat_indices  = flat_indices,          # kept in NumPy for scatter
        x             = x_disc,
        y             = y_disc,
        mu            = mu_disc,
        col_idx       = col_idx,
        vel_col       = vel_col,
    )


# ---------------------------------------------------------------------------
# 1b. Phase oversampling  (NumPy -- runs once in build_system)
# ---------------------------------------------------------------------------

def _make_oversampled_phases(
    phases_rot: np.ndarray,
    oversample: int,
) -> np.ndarray:
    """
    Spread each phase into ``oversample`` sub-phases spanning one exposure
    window, centred on the original phase.

    The exposure window for each phase is defined as the interval between
    the midpoints to its neighbours (i.e. one phase step for uniform grids).
    Sub-phases are uniformly spaced within this window.

    Parameters
    ----------
    phases_rot : (nphase,) array
        Original rotational phases in degrees.
    oversample : int
        Number of sub-exposures per phase point.  Must be >= 1.

    Returns
    -------
    oversampled : (nphase * oversample,) array
        Sub-phases in degrees, wrapped to [0, 360).
        Ordered as [p0_sub0, p0_sub1, ..., p0_subN, p1_sub0, ...].
    """
    if oversample <= 1:
        return phases_rot

    n = len(phases_rot)

    # Phase step -- assumes approximately uniform spacing
    if n > 1:
        dp = phases_rot[1] - phases_rot[0]
    else:
        dp = 360.0 / n

    # Sub-phase offsets centred on zero
    # For oversample=3: [-dp/3, 0, +dp/3]
    offsets = np.linspace(-dp / 2, dp / 2, oversample, endpoint=False)
    offsets += dp / (2 * oversample)  # centre within each sub-bin

    # Broadcast: (nphase, 1) + (1, oversample) -> (nphase, oversample)
    oversampled = phases_rot[:, None] + offsets[None, :]

    return oversampled.ravel()


# ---------------------------------------------------------------------------
# 2. Active region shape  (JAX -- operates on 1D in-disc arrays)
# ---------------------------------------------------------------------------

def _compute_ar_shape(
    x_disc: jnp.ndarray,       # (total_pixels,)
    y_disc: jnp.ndarray,       # (total_pixels,)
    star_pixel_rad: float,
    spx: float,
    spy: float,
    spz: float,
    arsize_rad: float,
    ar_smoothness: float,
) -> jnp.ndarray:
    """
    Super-Gaussian shape over in-disc pixels, centred on the active region.

    The AR is a spherical cap of angular radius ``arsize_rad`` (the distribution's
    "sigma"). The shape falls off from the AR's centre as a super-Gaussian
    of order ``ar_smoothness``, in the great-circle angle ``theta`` from the AR centre:

      - ``ar_smoothness -> inf``  converges to a top-hat function.
      - ``ar_smoothness == 1``    is a Gaussian in theta, with
                                   sigma = arsize_rad / sqrt(2).

    This shape peaks at exactly 1 at the AR's centre and its amplitude within
    the light-curve formula is set separately by the AR's spectral contrast (see
    ``_flux_at_wavelength``). Thus, this function is purely geometric and
    carries no wavelength dependence.

    Uses the exact spherical "distance" variable ``x = 1 - cos(theta) = 2 sin^2(theta/2)``,
    so this *holds even for large active regions*, unlike a flat-sky ``theta``.

    Parameters
    ----------
    x_disc, y_disc : (total_pixels,)
        Pixel coordinates of in-disc pixels.
    star_pixel_rad : float
        Stellar radius in pixels.
    spx, spy, spz : float
        Active region centre Cartesian coordinates (after rotation + inclination).
    arsize_rad : float
        Active region angular radius in radians ("sigma" of the super-Gaussian).
    ar_smoothness : float
        Super-Gaussian order controlling the sharpness of the AR boundary.

    Returns
    -------
    jnp.ndarray, shape (total_pixels,), dtype float32, values in (0, 1]
    """
    # Pixel z-coordinates on the stellar sphere.
    r2     = x_disc ** 2 + y_disc ** 2
    z_disc = jnp.sqrt(jnp.maximum(star_pixel_rad ** 2 - r2, 0.0))

    # Chord between the pixel and the AR centre, both normalised onto the
    # unit sphere.
    dx = x_disc / star_pixel_rad - spx / star_pixel_rad
    dy = y_disc / star_pixel_rad - spy / star_pixel_rad
    dz = z_disc / star_pixel_rad - spz / star_pixel_rad

    # Exact spherical "distance" variable: x = 1 - cos(theta) = |chord|^2 / 2.
    x  = 0.5 * (dx ** 2 + dy ** 2 + dz ** 2)
    x0 = jnp.maximum(2.0 * jnp.sin(arsize_rad / 2.0) ** 2, _AR_SHAPE_TINY)

    exponent = ar_smoothness

    # Computed in log-space and clipped before exponentiating: for large
    # ar_smoothness / small arsize_rad, (x/x0)**exponent overflows to inf
    # in float32 for pixels far from the AR, and the resulting exp(-inf)
    # chain produces NaN gradients (a 0*inf indeterminate). The tiny floor
    # on x/x0 avoids log(0) at the AR-centre pixel (x == 0 exactly, which
    # occurs whenever a pixel coincides with the AR centre to float
    # precision) -- without it, d/d(ar_smoothness) of 0*log(0) is NaN even
    # though the true limiting gradient there is 0.
    u = jnp.maximum(x / x0, _AR_SHAPE_TINY)
    log_arg = jnp.minimum(exponent * jnp.log(u), _LOG_ARG_MAX)

    return jnp.exp(-jnp.exp(log_arg))


# ---------------------------------------------------------------------------
# 3. Limb-darkening law (shared by the quiet photosphere and every AR, each
#    with its own coefficients)
# ---------------------------------------------------------------------------

def _evaluate_ldc(
    mu_disc:        jnp.ndarray, # (total_pixels,) grid of in-disc mu values
    ld_coeffs_wl:  jnp.ndarray, # (n_coeffs,) one row for this wavelength
    I_prof_wl:      jnp.ndarray, # (n_mu_pts,) used only for "intensity_profile"
    mu_profile_pts: jnp.ndarray, # (n_mu_pts,) set of mu points of the user-provided intensity profile. used only for "intensity_profile"
    ld_mode:       LdMode,     # limb-darkening law to use
) -> jnp.ndarray:
    """
    Evaluate the limb-darkening law at each pixel for one wavelength bin.

    The same function is used for the quiet photosphere and for every
    active region -- each caller supplies its own ``ld_coeffs_wl`` (and,
    for ``ld_mode="intensity_profile"``, its own ``I_prof_wl``), but the
    functional law itself (``ld_mode``) is shared by the whole star.

    Returns
    -------
    jnp.ndarray, shape (total_pixels,)
    """
    if ld_mode == "intensity_profile":
        # Interpolate a user-supplied I(mu) profile.
        result = jnp.interp(mu_disc, mu_profile_pts, I_prof_wl,
                             left=0.0, right=0.0)
    elif ld_mode == "linear":
        # I(μ) = 1 - u*(1 - μ)
        result = 1.0 - ld_coeffs_wl[0] * (1.0 - mu_disc)
    elif ld_mode == "quadratic":
        # I(μ) = 1 - u1*(1-μ) - u2*(1-μ)^2
        result = (1.0
                  - ld_coeffs_wl[0] * (1.0 - mu_disc)
                  - ld_coeffs_wl[1] * (1.0 - mu_disc) ** 2)
    elif ld_mode == "power2":
        # I(μ) = 1 - a*(1 - μ^b)
        result = 1.0 - ld_coeffs_wl[0] * (1.0 - mu_disc ** ld_coeffs_wl[1])
    elif ld_mode == "kipping3":
        # I(μ) = 1 - c1*(1-μ^0.5) - c2*(1-μ) - c3*(1-μ^(3/2))
        result = (1.0
                  - ld_coeffs_wl[0] * (1.0 - mu_disc ** 0.5)
                  - ld_coeffs_wl[1] * (1.0 - mu_disc)
                  - ld_coeffs_wl[2] * (1.0 - mu_disc ** 1.5))
    else:  # "nonlinear4"  -- Claret (2000) four-parameter law
        # I(μ) = 1 - Σ_{k=1}^{4} c_k*(1 - μ^(k/2))
        result = (1.0
                  - ld_coeffs_wl[0] * (1.0 - mu_disc ** 0.5)
                  - ld_coeffs_wl[1] * (1.0 - mu_disc)
                  - ld_coeffs_wl[2] * (1.0 - mu_disc ** 1.5)
                  - ld_coeffs_wl[3] * (1.0 - mu_disc ** 2.0))

    # Intensity can't be negative. Unphysical LDCs (e.g. u1+u2 > 1 for the
    # quadratic law) otherwise dip slightly negative near the limb, which
    # the _FLUX_RATIO_EPS floor in _flux_at_wavelength turns into a division
    # by (almost) zero -- a spurious contrast blowup rather than a graceful
    # degradation to zero flux.
    return jnp.maximum(result, 0.0)


# ---------------------------------------------------------------------------
# 4. Single-wavelength flux  (vmapped over the spectral axis)
# ---------------------------------------------------------------------------

def _flux_at_wavelength(
    # --- vmapped: one scalar/slice per wavelength ---
    wavelength_target:    float,
    ld_coeffs_quiet_wl:  jnp.ndarray, # (n_coeffs,)
    ld_coeffs_active_wl: jnp.ndarray, # (nar, n_coeffs)
    I_prof_quiet_wl:      jnp.ndarray, # (n_mu_pts,)
    I_prof_active_wl:     jnp.ndarray, # (nar, n_mu_pts)
    k_wl:                 jnp.ndarray, # (nplanet,) Rp / R* of every planet at this wavelength
    # --- broadcast: shared across wavelengths ---
    wavelength_grid: jnp.ndarray, # (nwave,) full spectral axis
    flux_quiet:      jnp.ndarray, # (nwave,) full quiet-photosphere spectrum
    flux_active:     jnp.ndarray, # (nar, nwave) full per-AR spectra
    mu_disc:         jnp.ndarray, # (total_pixels,) grid of in-disc mu values
    total_pixels:    int,
    ar_shapes:       jnp.ndarray, # (nar, total_pixels)
    x_disc:          jnp.ndarray, # (total_pixels,) -- for the per-wavelength planet mask
    y_disc:          jnp.ndarray, # (total_pixels,)
    star_pixel_rad:  float,
    planet_xyz:      jnp.ndarray, # (nplanet, 3) planet sky positions, shared across wavelengths
    transit_softness: float,
    mu_profile_pts:  jnp.ndarray, # (n_mu_pts,)
    col_idx:         jnp.ndarray, # (total_pixels,) int
    vel_col:         jnp.ndarray, # (n,)
    ld_mode:        LdMode,
) -> tuple[float, jnp.ndarray]:
    """
    Compute disc-integrated flux for a single wavelength channel.

    Builds the dimensionless "contrast surface"::

        F_p/F_quiet = 1 - sum_a (1 - C_a) * ar_shapes[a]

    where ``C_a = F_a / F_quiet`` is active region ``a``'s spectral
    contrast, evaluated at each pixel's own Doppler-shifted wavelength and
    with its own limb-darkening law. Overlapping active regions sum, so
    e.g. an umbra sitting inside a penumbra contributes simultaneously (the
    combined dip exceeds either component's own contrast).

    The planet mask is computed here, per wavelength, from ``k_wl`` -- not
    precomputed once per phase like ``ar_shapes`` -- because ``k`` (unlike
    everything else that's shared across wavelengths) may itself vary by
    wavelength for a chromatic transit depth. Recomputing the mask once per
    (phase, wavelength) instead of once per phase is the price of that
    generality; the mask itself is a cheap elementwise op over the pixel
    grid relative to the rest of this function.

    Every planet's mask is combined multiplicatively via
    ``_compute_all_planets_mask`` -- planets are opaque occulters, so
    overlapping planets' "surviving flux" fractions multiply rather than
    sum. The exception is when ``nplanets=1``: in that case the code 
    defaults back to using ``_compute_planet_mask``. This ensures there is
    no loss in computational cost in the ``nplanets=1`` case.

    All arrays are 1D (in-disc pixels only) - no starmask needed.
    The combined planet mask is 1 for pixels occulted by any planet; those
    pixels contribute zero flux regardless of active-region status.

    Returns
    -------
    total_flux : float            - active-region'ed integrated flux
    arted_flux : (total_pixels,)  - per-pixel flux values (for map output)
    """
    planet_mask = _compute_all_planets_mask(
        x_disc, y_disc, star_pixel_rad, planet_xyz, k_wl, transit_softness,
    )

    # ---- Per-column Doppler-shifted spectral lookup ----------------------
    # Velocity depends on x alone (see build_stellar_grid), so resample once per column, not per pixel.
    query_wavelength_col = wavelength_target * (1.0 - vel_col)  # (n,)

    F_quiet_col   = jnp.interp(query_wavelength_col, wavelength_grid, flux_quiet)  # (n,)
    F_quiet_local = F_quiet_col[col_idx]                                          # (total_pixels,)

    F_active_col = vmap(
        lambda spec: jnp.interp(query_wavelength_col, wavelength_grid, spec)
    )(flux_active)                                   # (nar, n)
    F_active_local = F_active_col[:, col_idx]        # (nar, total_pixels)

    # ---- Limb darkening (own law coefficients per AR and for quiet) -----
    ldc_quiet = _evaluate_ldc(
        mu_disc, ld_coeffs_quiet_wl, I_prof_quiet_wl, mu_profile_pts, ld_mode,
    )  # (total_pixels,)

    ldc_active = vmap(
        lambda coeffs, iprof: _evaluate_ldc(
            mu_disc, coeffs, iprof, mu_profile_pts, ld_mode,
        )
    )(ld_coeffs_active_wl, I_prof_active_wl)  # (nar, total_pixels)

    # ---- Physical flux (Doppler-shifted spectrum x limb darkening) ------
    F_quiet_px  = F_quiet_local * ldc_quiet    # (total_pixels,)
    F_active_px = F_active_local * ldc_active  # (nar, total_pixels)

    # ---- Contrast surface: overlapping ARs sum -----
    # Floored denominator: see _FLUX_RATIO_EPS -- pixels where F_quiet_px
    # is exactly 0 (e.g. an intensity_profile with I(mu=0)=0) still
    # contribute exactly 0 to arted_flux below, but without the floor the
    # 0/0 here would inject a NaN that corrupts the whole light curve.
    C_a = F_active_px / jnp.maximum(F_quiet_px[None, :], _FLUX_RATIO_EPS)  # (nar, total_pixels)
    contrast_surface = 1.0 - jnp.sum((1.0 - C_a) * ar_shapes, axis=0)      # (total_pixels,)

    arted_flux = F_quiet_px * contrast_surface

    # ---- Planet occultation ---------------------------------------
    # Multiplication (not jnp.where) so gradients flow through the planet mask.
    arted_flux = arted_flux * (1.0 - planet_mask)

    total_flux = jnp.sum(arted_flux) / jnp.float32(total_pixels)
    return total_flux, arted_flux


# ---------------------------------------------------------------------------
# 5. Single-phase computation
# ---------------------------------------------------------------------------

def _compute_flux_discs(
    ar_cart_all:         jnp.ndarray,  # (nar, 3)
    planet_xyz:          jnp.ndarray,  # (nplanet, 3)
    *,
    wavelength:          jnp.ndarray,  # (nwave,)
    flux_quiet:          jnp.ndarray,  # (nwave,)
    flux_active:         jnp.ndarray,  # (nar, nwave)
    ld_coeffs_quiet:    jnp.ndarray,  # (nwave, n_coeffs)
    ld_coeffs_active:   jnp.ndarray,  # (nar, nwave, n_coeffs)
    I_profile_quiet:     jnp.ndarray,  # (nwave, n_mu_pts)
    I_profile_active:    jnp.ndarray,  # (nar, nwave, n_mu_pts)
    mu_profile_pts:      jnp.ndarray,  # (n_mu_pts,)
    x_disc:              jnp.ndarray,  # (total_pixels,)
    y_disc:              jnp.ndarray,  # (total_pixels,)
    mu_disc:             jnp.ndarray,  # (total_pixels,)
    col_idx:             jnp.ndarray,  # (total_pixels,)
    vel_col:             jnp.ndarray,  # (n_grid,)
    star_pixel_rad:      float,
    total_pixels:        int,
    arsize_rads:         jnp.ndarray,  # (nar,)
    ar_smoothness:       jnp.ndarray,  # (nar,)
    k:                   jnp.ndarray,  # (nplanet, nwave) Rp / R*, one value per planet per wavelength
    ld_mode:            LdMode,
    transit_softness:    float = 0.0, # see _compute_planet_mask
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shared core of one rotational phase's spectral computation --
    what the light-curve path (``_compute_single_phase_lc``, which also needs
    a 2D ``star_map`` reconstruction) and the RV path
    (``_compute_single_phase_rv``, which instead reduces ``flux_discs`` to
    a flux-weighted velocity) have in common: build the active-region
    shapes, then vmap ``_flux_at_wavelength`` over the wavelength axis.

    ``planet_xyz`` holds every planet's sky-plane position (X, Y, Z) in
    stellar radii. ``k`` differs per planet and per wavelength (to allow for
    chromatic transit depths), so the occultation mask itself is computed per
    wavelength inside ``_flux_at_wavelength`` (combining all planets
    multiplicatively via ``_compute_all_planets_mask``), not once here. Pass
    ``jnp.array([[0., 0., -1e10]])`` and an all-zero ``k`` to disable the
    transit mask (no performance cost -- the mask is all-False).

    Returns
    -------
    flux_per_wavelength   : (nwave,)  disc-integrated flux at each
                             wavelength bin, in the same units as
                             ``flux_quiet``/``flux_active`` (not normalised
                             to the quiet-star baseline -- divide by that
                             yourself, e.g. via a quiet-star-only call to
                             ``make_lc``, if you want relative flux)
    flux_discs             : (nwave, total_pixels)  per-pixel flux at each
                             wavelength bin, after limb darkening, the
                             active-region contrast surface, planet
                             occultation, and the per-column Doppler-shifted
                             spectral lookup.
    """
    # ---- active region shapes: (nar, total_pixels) -----------------------
    ar_shapes = vmap(
        lambda cart, sr, sm: _compute_ar_shape(
            x_disc, y_disc, star_pixel_rad,
            cart[0], cart[1], cart[2], sr, sm,
        )
    )(ar_cart_all, arsize_rads, ar_smoothness)

    # ---- vmap over wavelengths ----
    # ld_coeffs_active/I_profile_active have the wavelength axis second
    # (nar, nwave, ...), so they vmap on axis=1; k now has the same layout
    # (nplanet, nwave), so it also vmaps on axis=1 -- everything else vmaps
    # on its leading (wavelength) axis, so the planet mask (computed inside
    # _flux_at_wavelength) can differ per wavelength too.
    _flux_vmap = vmap(
        functools.partial(
            _flux_at_wavelength,
            wavelength_grid  = wavelength,
            flux_quiet       = flux_quiet,
            flux_active      = flux_active,
            mu_disc          = mu_disc,
            total_pixels     = total_pixels,
            ar_shapes        = ar_shapes,
            x_disc           = x_disc,
            y_disc           = y_disc,
            star_pixel_rad   = star_pixel_rad,
            planet_xyz       = planet_xyz,
            transit_softness = transit_softness,
            mu_profile_pts   = mu_profile_pts,
            col_idx          = col_idx,
            vel_col          = vel_col,
            ld_mode         = ld_mode,
        ),
        in_axes=(0, 0, 1, 0, 1, 1),
    )

    flux_per_wavelength, flux_discs = _flux_vmap(
        wavelength,
        ld_coeffs_quiet,
        ld_coeffs_active,
        I_profile_quiet,
        I_profile_active,
        k,
    )

    return flux_per_wavelength, flux_discs


def _reconstruct_star_map(
    flux_discs:          jnp.ndarray,  # (nwave, total_pixels)
    wavelength:          jnp.ndarray,  # (nwave,)
    plot_map_wavelength: float,
    n:                   int,         # full grid side (for map scatter)
    flat_indices:        jnp.ndarray, # (total_pixels,) scatter indices
) -> jnp.ndarray:
    """
    Scatter one wavelength bin's per-pixel flux (the row of ``flux_discs``
    nearest ``plot_map_wavelength``) back onto the full (n, n) pixel grid.
    Shared by ``_compute_single_phase_lc`` and ``_compute_single_phase_rv``.

    Returns
    -------
    jnp.ndarray, shape (n, n)
    """
    map_idx = jnp.argmin(jnp.abs(wavelength - plot_map_wavelength))
    flux_1d = flux_discs[map_idx]   # (total_pixels,)
    return jnp.zeros(n * n).at[flat_indices].set(flux_1d).reshape(n, n)


def _rv_from_flux_discs(
    flux_discs: jnp.ndarray,  # (nwave, total_pixels)
    vel_col:    jnp.ndarray,  # (n_grid,)
    col_idx:    jnp.ndarray,  # (total_pixels,)
) -> jnp.ndarray:
    """
    Reduce one phase's ``(nwave, total_pixels)`` ``flux_discs`` (see
    ``_compute_flux_discs``) to a flux-weighted radial-velocity vector, one
    value per wavelength bin. Shared by ``_compute_single_phase_rv`` and
    ``_compute_single_phase_lc_rv`` -- this is purely the reduction step,
    not the (expensive, already-shared-via-``_compute_flux_discs``) flux
    computation itself.

    Deliberately NOT integrated/averaged over wavelength -- see
    ``_compute_single_phase_rv``'s docstring for why, and for the sign
    derivation/empirical arbiter (``tests/test_core.py::TestEquatorialSpotRedLimbBlueshift``).

    Returns
    -------
    jnp.ndarray, shape (nwave,) -- RV anomaly [km/s] at each wavelength bin.
    """
    v_pixel = vel_col[col_idx]   # (total_pixels,), units of v/c
    denom = jnp.maximum(jnp.sum(flux_discs, axis=1), _RV_WEIGHT_EPS)   # (nwave,)
    return _C_KMS * jnp.sum(flux_discs * v_pixel[None, :], axis=1) / denom   # (nwave,)


def _compute_single_phase_lc(
    ar_cart_all:         jnp.ndarray,  # (nar, 3)
    planet_xyz:          jnp.ndarray,  # (nplanet, 3)
    *,
    wavelength:          jnp.ndarray,  # (nwave,)
    flux_quiet:          jnp.ndarray,  # (nwave,)
    flux_active:         jnp.ndarray,  # (nar, nwave)
    ld_coeffs_quiet:    jnp.ndarray,  # (nwave, n_coeffs)
    ld_coeffs_active:   jnp.ndarray,  # (nar, nwave, n_coeffs)
    I_profile_quiet:     jnp.ndarray,  # (nwave, n_mu_pts)
    I_profile_active:    jnp.ndarray,  # (nar, nwave, n_mu_pts)
    mu_profile_pts:      jnp.ndarray,  # (n_mu_pts,)
    x_disc:              jnp.ndarray,  # (total_pixels,)
    y_disc:              jnp.ndarray,  # (total_pixels,)
    mu_disc:             jnp.ndarray,  # (total_pixels,)
    col_idx:             jnp.ndarray,  # (total_pixels,)
    vel_col:             jnp.ndarray,  # (n_grid,)
    star_pixel_rad:      float,
    total_pixels:        int,
    arsize_rads:         jnp.ndarray,  # (nar,)
    ar_smoothness:       jnp.ndarray,  # (nar,)
    k:                   jnp.ndarray,  # (nplanet, nwave) Rp / R*, one value per planet per wavelength
    ld_mode:            LdMode,
    plot_map_wavelength: float,
    n:                   int,         # full grid side (for map scatter)
    flat_indices:        jnp.ndarray, # (total_pixels,) scatter indices
    transit_softness:    float = 0.0, # see _compute_planet_mask
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Full spectral computation for one rotational phase, including optional
    pixel-level planet occultation. See ``_compute_flux_discs`` (this
    function's shared core with the RV path) for the ``planet_xyz``/``k``
    semantics.

    Returns
    -------
    flux_per_wavelength   : (nwave,)  disc-integrated flux at each
                             wavelength bin, in the same units as
                             ``flux_quiet``/``flux_active`` (not normalised
                             to the quiet-star baseline -- divide by that
                             yourself, e.g. via a quiet-star-only call to
                             ``make_lc``, if you want relative flux)
    star_map              : (n, n)  flux map at plot_map_wavelength
    """
    flux_per_wavelength, flux_discs = _compute_flux_discs(
        ar_cart_all, planet_xyz,
        wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_active,
        ld_coeffs_quiet=ld_coeffs_quiet, ld_coeffs_active=ld_coeffs_active,
        I_profile_quiet=I_profile_quiet, I_profile_active=I_profile_active,
        mu_profile_pts=mu_profile_pts, x_disc=x_disc, y_disc=y_disc,
        mu_disc=mu_disc, col_idx=col_idx, vel_col=vel_col,
        star_pixel_rad=star_pixel_rad, total_pixels=total_pixels,
        arsize_rads=arsize_rads, ar_smoothness=ar_smoothness, k=k,
        ld_mode=ld_mode, transit_softness=transit_softness,
    )

    star_map = _reconstruct_star_map(flux_discs, wavelength, plot_map_wavelength, n, flat_indices)
    return flux_per_wavelength, star_map


# ---------------------------------------------------------------------------
# 6. All-phases computation -- vmapped over the phase axis
# ---------------------------------------------------------------------------

def _compute_all_phases_lc(
    all_ar_carts:    jnp.ndarray,   # (nphase, nar, 3)
    planet_xyz_all:  jnp.ndarray,   # (nphase, nplanet, 3)
    *,
    wavelength:          jnp.ndarray,
    flux_quiet:          jnp.ndarray,
    flux_active:         jnp.ndarray,
    ld_coeffs_quiet:    jnp.ndarray, # (nwave, n_coeffs)
    ld_coeffs_active:   jnp.ndarray, # (nar, nwave, n_coeffs)
    I_profile_quiet:     jnp.ndarray,
    I_profile_active:    jnp.ndarray,
    mu_profile_pts:      jnp.ndarray,
    x_disc:              jnp.ndarray,
    y_disc:              jnp.ndarray,
    mu_disc:             jnp.ndarray,
    col_idx:             jnp.ndarray,
    vel_col:             jnp.ndarray,
    star_pixel_rad:      float,
    total_pixels:        int,
    arsize_rads:         jnp.ndarray,
    ar_smoothness:       jnp.ndarray,
    k:                   jnp.ndarray,  # (nplanet, nwave) Rp / R*, one value per planet per wavelength
    ld_mode:            LdMode,
    plot_map_wavelength: float,
    n:                   int,
    flat_indices:        jnp.ndarray,
    transit_softness:    float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    vmap ``_compute_single_phase_lc`` over the phase axis.

    ``planet_xyz_all`` contains every planet's (X, Y, Z) position at each
    (oversampled) phase, shared across every wavelength. Pass
    ``jnp.full((nphase, 1, 3), [0, 0, -1e10])`` and an all-zero ``k`` to
    disable transit (no performance overhead).

    Returns
    -------
    lc_raw    : (nphase, nwave)
    star_maps : (nphase, n, n)
    """
    _phase_vmap = vmap(
        functools.partial(
            _compute_single_phase_lc,
            wavelength          = wavelength,
            flux_quiet          = flux_quiet,
            flux_active         = flux_active,
            ld_coeffs_quiet    = ld_coeffs_quiet,
            ld_coeffs_active   = ld_coeffs_active,
            I_profile_quiet     = I_profile_quiet,
            I_profile_active    = I_profile_active,
            mu_profile_pts      = mu_profile_pts,
            x_disc              = x_disc,
            y_disc              = y_disc,
            mu_disc             = mu_disc,
            col_idx             = col_idx,
            vel_col             = vel_col,
            star_pixel_rad      = star_pixel_rad,
            total_pixels        = total_pixels,
            arsize_rads         = arsize_rads,
            ar_smoothness       = ar_smoothness,
            k                   = k,
            ld_mode            = ld_mode,
            plot_map_wavelength = plot_map_wavelength,
            n                   = n,
            flat_indices        = flat_indices,
            transit_softness    = transit_softness,
        ),
        in_axes=(0,0), # vmap over both ar_carts and planet_xyz
    )
    return _phase_vmap(all_ar_carts, planet_xyz_all)


# ---------------------------------------------------------------------------
# 6b. All-phases computation -- time-varying active-region counterpart.
# ---------------------------------------------------------------------------

def _compute_all_phases_lc_evolving(
    ar_lat_all:         jnp.ndarray,   # (nphase, nar) degrees
    ar_long_all:        jnp.ndarray,   # (nphase, nar) degrees
    arsize_rads_all:    jnp.ndarray,   # (nphase, nar) radians
    ar_smoothness_all:  jnp.ndarray,   # (nphase, nar)
    flux_active_all:    jnp.ndarray,   # (nphase, nar, nwave)
    planet_xyz_all:     jnp.ndarray,   # (nphase, nplanet, 3)
    phases_rot:         jnp.ndarray,   # (nphase,) degrees -- stellar rotation phase
    *,
    inc_star:           float,
    wavelength:         jnp.ndarray,
    flux_quiet:         jnp.ndarray,
    ld_coeffs_quiet:    jnp.ndarray,   # (nwave, n_coeffs) -- fixed across phases
    ld_coeffs_active:   jnp.ndarray,   # (nar, nwave, n_coeffs) -- fixed across phases
    I_profile_quiet:    jnp.ndarray,
    I_profile_active:   jnp.ndarray,   # (nar, nwave, n_mu_pts) -- fixed across phases
    mu_profile_pts:     jnp.ndarray,
    x_disc:             jnp.ndarray,
    y_disc:             jnp.ndarray,
    mu_disc:            jnp.ndarray,
    col_idx:            jnp.ndarray,
    vel_col:            jnp.ndarray,
    star_pixel_rad:     float,
    total_pixels:       int,
    k:                  jnp.ndarray,  # (nplanet, nwave)
    ld_mode:            LdMode,
    plot_map_wavelength:float,
    n:                  int,
    flat_indices:       jnp.ndarray,
    transit_softness:   float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Time-varying counterpart to ``_compute_all_phases_lc``. Used only when make_lc() is given at least
    one AR parameter with a time axis. Identical to _compute_all_phases_lc except
    ar_lat/ar_long/arsize_rads/ ar_smoothness/flux_active are supplied per-phase and the AR's Cartesian
    position is rebuilt and rotated fresh every phase. This function's existence adds no cost to 
    _compute_all_phases_lc or any caller that doesn't use it.

    Returns
    -------
    lc_raw    : (nphase, nwave)
    star_maps : (nphase, n, n)
    """
    def _single_phase(ar_lat_ph, ar_long_ph, arsize_ph, smooth_ph, flux_ph,
                       planet_xyz_ph, phase_deg):
        lat_rad  = jnp.deg2rad(ar_lat_ph)
        long_rad = jnp.deg2rad(ar_long_ph)
        cart = jnp.stack([
            star_pixel_rad * jnp.sin(long_rad) * jnp.cos(lat_rad),
            star_pixel_rad * jnp.sin(lat_rad),
            star_pixel_rad * jnp.cos(long_rad) * jnp.cos(lat_rad),
        ], axis=-1)   # (nar, 3)
        cart_rot = vmap(
            lambda c: rotate_active_region(c, phase_deg, inc_star)
        )(cart)
        return _compute_single_phase_lc(
            cart_rot, planet_xyz_ph,
            wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_ph,
            ld_coeffs_quiet=ld_coeffs_quiet, ld_coeffs_active=ld_coeffs_active,
            I_profile_quiet=I_profile_quiet, I_profile_active=I_profile_active,
            mu_profile_pts=mu_profile_pts, x_disc=x_disc, y_disc=y_disc,
            mu_disc=mu_disc, col_idx=col_idx, vel_col=vel_col,
            star_pixel_rad=star_pixel_rad, total_pixels=total_pixels,
            arsize_rads=arsize_ph, ar_smoothness=smooth_ph, k=k, ld_mode=ld_mode,
            plot_map_wavelength=plot_map_wavelength, n=n, flat_indices=flat_indices,
            transit_softness=transit_softness,
        )

    _phase_vmap = vmap(_single_phase, in_axes=(0, 0, 0, 0, 0, 0, 0))
    return _phase_vmap(ar_lat_all, ar_long_all, arsize_rads_all, ar_smoothness_all,
                        flux_active_all, planet_xyz_all, phases_rot)


# ---------------------------------------------------------------------------
# 6c. Radial velocity, per phase / all phases
# ---------------------------------------------------------------------------

def _compute_single_phase_rv(
    ar_cart_all:         jnp.ndarray,  # (nar, 3)
    planet_xyz:          jnp.ndarray,  # (nplanet, 3)
    *,
    wavelength:          jnp.ndarray,  # (nwave,)
    flux_quiet:          jnp.ndarray,  # (nwave,)
    flux_active:         jnp.ndarray,  # (nar, nwave)
    ld_coeffs_quiet:    jnp.ndarray,  # (nwave, n_coeffs)
    ld_coeffs_active:   jnp.ndarray,  # (nar, nwave, n_coeffs)
    I_profile_quiet:     jnp.ndarray,  # (nwave, n_mu_pts)
    I_profile_active:    jnp.ndarray,  # (nar, nwave, n_mu_pts)
    mu_profile_pts:      jnp.ndarray,  # (n_mu_pts,)
    x_disc:              jnp.ndarray,  # (total_pixels,)
    y_disc:              jnp.ndarray,  # (total_pixels,)
    mu_disc:             jnp.ndarray,  # (total_pixels,)
    col_idx:             jnp.ndarray,  # (total_pixels,)
    vel_col:             jnp.ndarray,  # (n_grid,)
    star_pixel_rad:      float,
    total_pixels:        int,
    arsize_rads:         jnp.ndarray,  # (nar,)
    ar_smoothness:       jnp.ndarray,  # (nar,)
    k:                   jnp.ndarray,  # (nplanet, nwave)
    ld_mode:            LdMode,
    plot_map_wavelength: float,
    n:                   int,         # full grid side (for map scatter)
    flat_indices:        jnp.ndarray, # (total_pixels,) scatter indices
    transit_softness:    float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Radial velocit computation for one rotational phase, including optional
    pixel-level planet occultation. See ``_compute_flux_discs`` (this
    function's shared core with the LC path) for the ``planet_xyz``/``k``
    semantics.

    Returns
    -------
    rv       : jnp.ndarray, shape (nwave,) -- RV anomaly [km/s] for this
               phase, at each wavelength bin.
    rv_map   : (n, n)  radial-velocity map at plot_map_wavelength.
    """
    _, flux_discs = _compute_flux_discs(
        ar_cart_all, planet_xyz,
        wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_active,
        ld_coeffs_quiet=ld_coeffs_quiet, ld_coeffs_active=ld_coeffs_active,
        I_profile_quiet=I_profile_quiet, I_profile_active=I_profile_active,
        mu_profile_pts=mu_profile_pts, x_disc=x_disc, y_disc=y_disc,
        mu_disc=mu_disc, col_idx=col_idx, vel_col=vel_col,
        star_pixel_rad=star_pixel_rad, total_pixels=total_pixels,
        arsize_rads=arsize_rads, ar_smoothness=ar_smoothness, k=k,
        ld_mode=ld_mode, transit_softness=transit_softness,
    )   # flux_discs: (nwave, total_pixels)

    rv = _rv_from_flux_discs(flux_discs, vel_col, col_idx)

    map_idx = jnp.argmin(jnp.abs(wavelength - plot_map_wavelength))
    flux_1d = flux_discs[map_idx]   # (total_pixels,)
    v_pixel = _C_KMS * vel_col[col_idx]   # (total_pixels,), km/s
    weight  = flux_1d / jnp.maximum(jnp.max(flux_1d), _RV_WEIGHT_EPS)
    rv_map  = jnp.zeros(n * n).at[flat_indices].set(v_pixel * weight).reshape(n, n)

    return rv, rv_map

def _compute_all_phases_rv(
    all_ar_carts:    jnp.ndarray,   # (nphase, nar, 3)
    planet_xyz_all:  jnp.ndarray,   # (nphase, nplanet, 3)
    *,
    wavelength:          jnp.ndarray,
    flux_quiet:          jnp.ndarray,
    flux_active:         jnp.ndarray,
    ld_coeffs_quiet:    jnp.ndarray,
    ld_coeffs_active:   jnp.ndarray,
    I_profile_quiet:     jnp.ndarray,
    I_profile_active:    jnp.ndarray,
    mu_profile_pts:      jnp.ndarray,
    x_disc:              jnp.ndarray,
    y_disc:              jnp.ndarray,
    mu_disc:             jnp.ndarray,
    col_idx:             jnp.ndarray,
    vel_col:             jnp.ndarray,
    star_pixel_rad:      float,
    total_pixels:        int,
    arsize_rads:         jnp.ndarray,
    ar_smoothness:       jnp.ndarray,
    k:                   jnp.ndarray,
    ld_mode:            LdMode,
    plot_map_wavelength: float,
    n:                   int,
    flat_indices:        jnp.ndarray,
    transit_softness:    float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    vmap ``_compute_single_phase_rv`` over the phase axis (static-AR path;
    see ``_compute_all_phases_rv_evolving`` for the time-varying-AR
    counterpart). Mirrors ``_compute_all_phases_lc``'s structure exactly,
    including its "planet parked behind the star" convention for
    disabling the transit mask.

    Returns
    -------
    rv        : (nphase, nwave) -- RV anomaly [km/s] per phase, per
                wavelength bin (see ``_compute_single_phase_rv``).
    star_maps : (nphase, n, n) -- flux map per phase.
    """
    _phase_vmap = vmap(
        functools.partial(
            _compute_single_phase_rv,
            wavelength          = wavelength,
            flux_quiet          = flux_quiet,
            flux_active         = flux_active,
            ld_coeffs_quiet    = ld_coeffs_quiet,
            ld_coeffs_active   = ld_coeffs_active,
            I_profile_quiet     = I_profile_quiet,
            I_profile_active    = I_profile_active,
            mu_profile_pts      = mu_profile_pts,
            x_disc              = x_disc,
            y_disc              = y_disc,
            mu_disc             = mu_disc,
            col_idx             = col_idx,
            vel_col             = vel_col,
            star_pixel_rad      = star_pixel_rad,
            total_pixels        = total_pixels,
            arsize_rads         = arsize_rads,
            ar_smoothness       = ar_smoothness,
            k                   = k,
            ld_mode            = ld_mode,
            plot_map_wavelength = plot_map_wavelength,
            n                   = n,
            flat_indices        = flat_indices,
            transit_softness    = transit_softness,
        ),
        in_axes=(0, 0),
    )
    return _phase_vmap(all_ar_carts, planet_xyz_all)


def _compute_all_phases_rv_evolving(
    ar_lat_all:         jnp.ndarray,   # (nphase, nar) degrees
    ar_long_all:        jnp.ndarray,   # (nphase, nar) degrees
    arsize_rads_all:    jnp.ndarray,   # (nphase, nar) radians
    ar_smoothness_all:  jnp.ndarray,   # (nphase, nar)
    flux_active_all:    jnp.ndarray,   # (nphase, nar, nwave)
    planet_xyz_all:     jnp.ndarray,   # (nphase, nplanet, 3)
    phases_rot:         jnp.ndarray,   # (nphase,) degrees
    *,
    inc_star:           float,
    wavelength:         jnp.ndarray,
    flux_quiet:         jnp.ndarray,
    ld_coeffs_quiet:    jnp.ndarray,
    ld_coeffs_active:   jnp.ndarray,
    I_profile_quiet:    jnp.ndarray,
    I_profile_active:   jnp.ndarray,
    mu_profile_pts:     jnp.ndarray,
    x_disc:             jnp.ndarray,
    y_disc:             jnp.ndarray,
    mu_disc:            jnp.ndarray,
    col_idx:            jnp.ndarray,
    vel_col:            jnp.ndarray,
    star_pixel_rad:     float,
    total_pixels:       int,
    k:                  jnp.ndarray,
    ld_mode:            LdMode,
    plot_map_wavelength: float,
    n:                  int,
    flat_indices:       jnp.ndarray,
    transit_softness:   float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Time-varying-AR counterpart of ``_compute_all_phases_rv``, mirroring
    ``_compute_all_phases_lc_evolving``'s per-phase Cartesian-position rebuild.

    Returns
    -------
    rv        : (nphase, nwave) -- RV anomaly [km/s] per phase, per
                wavelength bin.
    star_maps : (nphase, n, n) -- flux map per phase.
    """
    def _single_phase(ar_lat_ph, ar_long_ph, arsize_ph, smooth_ph, flux_ph,
                       planet_xyz_ph, phase_deg):
        lat_rad  = jnp.deg2rad(ar_lat_ph)
        long_rad = jnp.deg2rad(ar_long_ph)
        cart = jnp.stack([
            star_pixel_rad * jnp.sin(long_rad) * jnp.cos(lat_rad),
            star_pixel_rad * jnp.sin(lat_rad),
            star_pixel_rad * jnp.cos(long_rad) * jnp.cos(lat_rad),
        ], axis=-1)   # (nar, 3)
        cart_rot = vmap(
            lambda c: rotate_active_region(c, phase_deg, inc_star)
        )(cart)
        return _compute_single_phase_rv(
            cart_rot, planet_xyz_ph,
            wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_ph,
            ld_coeffs_quiet=ld_coeffs_quiet, ld_coeffs_active=ld_coeffs_active,
            I_profile_quiet=I_profile_quiet, I_profile_active=I_profile_active,
            mu_profile_pts=mu_profile_pts, x_disc=x_disc, y_disc=y_disc,
            mu_disc=mu_disc, col_idx=col_idx, vel_col=vel_col,
            star_pixel_rad=star_pixel_rad, total_pixels=total_pixels,
            arsize_rads=arsize_ph, ar_smoothness=smooth_ph, k=k, ld_mode=ld_mode,
            plot_map_wavelength=plot_map_wavelength, n=n, flat_indices=flat_indices,
            transit_softness=transit_softness,
        )

    _phase_vmap = vmap(_single_phase, in_axes=(0, 0, 0, 0, 0, 0, 0))
    return _phase_vmap(ar_lat_all, ar_long_all, arsize_rads_all, ar_smoothness_all,
                        flux_active_all, planet_xyz_all, phases_rot)


# ---------------------------------------------------------------------------
# 6d. Combined light curve + radial velocity, per phase / all phases
# ---------------------------------------------------------------------------

def _compute_single_phase_lc_rv(
    ar_cart_all:         jnp.ndarray,  # (nar, 3)
    planet_xyz:          jnp.ndarray,  # (nplanet, 3)
    *,
    wavelength:          jnp.ndarray,  # (nwave,)
    flux_quiet:          jnp.ndarray,  # (nwave,)
    flux_active:         jnp.ndarray,  # (nar, nwave)
    ld_coeffs_quiet:    jnp.ndarray,  # (nwave, n_coeffs)
    ld_coeffs_active:   jnp.ndarray,  # (nar, nwave, n_coeffs)
    I_profile_quiet:     jnp.ndarray,  # (nwave, n_mu_pts)
    I_profile_active:    jnp.ndarray,  # (nar, nwave, n_mu_pts)
    mu_profile_pts:      jnp.ndarray,  # (n_mu_pts,)
    x_disc:              jnp.ndarray,  # (total_pixels,)
    y_disc:              jnp.ndarray,  # (total_pixels,)
    mu_disc:             jnp.ndarray,  # (total_pixels,)
    col_idx:             jnp.ndarray,  # (total_pixels,)
    vel_col:             jnp.ndarray,  # (n_grid,)
    star_pixel_rad:      float,
    total_pixels:        int,
    arsize_rads:         jnp.ndarray,  # (nar,)
    ar_smoothness:       jnp.ndarray,  # (nar,)
    k:                   jnp.ndarray,  # (nplanet, nwave)
    ld_mode:            LdMode,
    plot_map_wavelength: float,
    n:                   int,         # full grid side (for map scatter)
    flat_indices:        jnp.ndarray, # (total_pixels,) scatter indices
    transit_softness:    float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Combined light-curve + radial-velocity computation for one rotational
    phase, calling ``_compute_flux_discs`` -- the expensive step (AR
    shapes + the per-pixel-per-wavelength ``_flux_at_wavelength`` vmap) --
    only once and deriving both outputs from its single ``flux_discs``
    result. This avoids 2x redundant computation that calling
    ``_compute_single_phase_lc`` and ``_compute_single_phase_rv``
    in series would do.

    Returns
    -------
    flux_per_wavelength : (nwave,) -- see ``_compute_single_phase_lc``.
    rv                  : (nwave,) -- see ``_compute_single_phase_rv``.
    star_map            : (n, n)   -- see ``_reconstruct_star_map``; the
                           SAME map both a standalone ``make_lc`` call would
                            produce on its own.
    """
    flux_per_wavelength, flux_discs = _compute_flux_discs(
        ar_cart_all, planet_xyz,
        wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_active,
        ld_coeffs_quiet=ld_coeffs_quiet, ld_coeffs_active=ld_coeffs_active,
        I_profile_quiet=I_profile_quiet, I_profile_active=I_profile_active,
        mu_profile_pts=mu_profile_pts, x_disc=x_disc, y_disc=y_disc,
        mu_disc=mu_disc, col_idx=col_idx, vel_col=vel_col,
        star_pixel_rad=star_pixel_rad, total_pixels=total_pixels,
        arsize_rads=arsize_rads, ar_smoothness=ar_smoothness, k=k,
        ld_mode=ld_mode, transit_softness=transit_softness,
    )   # flux_discs: (nwave, total_pixels)

    rv = _rv_from_flux_discs(flux_discs, vel_col, col_idx)
    star_map = _reconstruct_star_map(flux_discs, wavelength, plot_map_wavelength, n, flat_indices)
    return flux_per_wavelength, rv, star_map


def _compute_all_phases_lc_rv(
    all_ar_carts:    jnp.ndarray,   # (nphase, nar, 3)
    planet_xyz_all:  jnp.ndarray,   # (nphase, nplanet, 3)
    *,
    wavelength:          jnp.ndarray,
    flux_quiet:          jnp.ndarray,
    flux_active:         jnp.ndarray,
    ld_coeffs_quiet:    jnp.ndarray,
    ld_coeffs_active:   jnp.ndarray,
    I_profile_quiet:     jnp.ndarray,
    I_profile_active:    jnp.ndarray,
    mu_profile_pts:      jnp.ndarray,
    x_disc:              jnp.ndarray,
    y_disc:              jnp.ndarray,
    mu_disc:             jnp.ndarray,
    col_idx:             jnp.ndarray,
    vel_col:             jnp.ndarray,
    star_pixel_rad:      float,
    total_pixels:        int,
    arsize_rads:         jnp.ndarray,
    ar_smoothness:       jnp.ndarray,
    k:                   jnp.ndarray,
    ld_mode:            LdMode,
    plot_map_wavelength: float,
    n:                   int,
    flat_indices:        jnp.ndarray,
    transit_softness:    float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    vmap ``_compute_single_phase_lc_rv`` over the phase axis (static-AR
    path; see ``_compute_all_phases_lc_rv_evolving`` for the time-varying-AR
    counterpart). Mirrors ``_compute_all_phases_lc``/``_compute_all_phases_rv``'s
    structure exactly, including the "planet parked behind the star"
    convention for disabling the transit mask.

    Returns
    -------
    lc        : (nphase, nwave) -- disc-integrated flux per phase.
    rv        : (nphase, nwave) -- RV anomaly [km/s] per phase, per
                wavelength bin.
    star_maps : (nphase, n, n) -- flux map per phase.
    """
    _phase_vmap = vmap(
        functools.partial(
            _compute_single_phase_lc_rv,
            wavelength          = wavelength,
            flux_quiet          = flux_quiet,
            flux_active         = flux_active,
            ld_coeffs_quiet    = ld_coeffs_quiet,
            ld_coeffs_active   = ld_coeffs_active,
            I_profile_quiet     = I_profile_quiet,
            I_profile_active    = I_profile_active,
            mu_profile_pts      = mu_profile_pts,
            x_disc              = x_disc,
            y_disc              = y_disc,
            mu_disc             = mu_disc,
            col_idx             = col_idx,
            vel_col             = vel_col,
            star_pixel_rad      = star_pixel_rad,
            total_pixels        = total_pixels,
            arsize_rads         = arsize_rads,
            ar_smoothness       = ar_smoothness,
            k                   = k,
            ld_mode            = ld_mode,
            plot_map_wavelength = plot_map_wavelength,
            n                   = n,
            flat_indices        = flat_indices,
            transit_softness    = transit_softness,
        ),
        in_axes=(0, 0),
    )
    return _phase_vmap(all_ar_carts, planet_xyz_all)


def _compute_all_phases_lc_rv_evolving(
    ar_lat_all:         jnp.ndarray,   # (nphase, nar) degrees
    ar_long_all:        jnp.ndarray,   # (nphase, nar) degrees
    arsize_rads_all:    jnp.ndarray,   # (nphase, nar) radians
    ar_smoothness_all:  jnp.ndarray,   # (nphase, nar)
    flux_active_all:    jnp.ndarray,   # (nphase, nar, nwave)
    planet_xyz_all:     jnp.ndarray,   # (nphase, nplanet, 3)
    phases_rot:         jnp.ndarray,   # (nphase,) degrees
    *,
    inc_star:           float,
    wavelength:         jnp.ndarray,
    flux_quiet:         jnp.ndarray,
    ld_coeffs_quiet:    jnp.ndarray,
    ld_coeffs_active:   jnp.ndarray,
    I_profile_quiet:    jnp.ndarray,
    I_profile_active:   jnp.ndarray,
    mu_profile_pts:     jnp.ndarray,
    x_disc:             jnp.ndarray,
    y_disc:             jnp.ndarray,
    mu_disc:            jnp.ndarray,
    col_idx:            jnp.ndarray,
    vel_col:            jnp.ndarray,
    star_pixel_rad:     float,
    total_pixels:       int,
    k:                  jnp.ndarray,
    ld_mode:            LdMode,
    plot_map_wavelength: float,
    n:                  int,
    flat_indices:       jnp.ndarray,
    transit_softness:   float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Time-varying-AR counterpart of ``_compute_all_phases_lc_rv``, mirroring
    ``_compute_all_phases_lc_evolving``/``_compute_all_phases_rv_evolving``'s
    per-phase Cartesian-position rebuild.

    Returns
    -------
    lc        : (nphase, nwave) -- disc-integrated flux per phase.
    rv        : (nphase, nwave) -- RV anomaly [km/s] per phase, per
                wavelength bin.
    star_maps : (nphase, n, n) -- flux map per phase.
    """
    def _single_phase(ar_lat_ph, ar_long_ph, arsize_ph, smooth_ph, flux_ph,
                       planet_xyz_ph, phase_deg):
        lat_rad  = jnp.deg2rad(ar_lat_ph)
        long_rad = jnp.deg2rad(ar_long_ph)
        cart = jnp.stack([
            star_pixel_rad * jnp.sin(long_rad) * jnp.cos(lat_rad),
            star_pixel_rad * jnp.sin(lat_rad),
            star_pixel_rad * jnp.cos(long_rad) * jnp.cos(lat_rad),
        ], axis=-1)   # (nar, 3)
        cart_rot = vmap(
            lambda c: rotate_active_region(c, phase_deg, inc_star)
        )(cart)
        return _compute_single_phase_lc_rv(
            cart_rot, planet_xyz_ph,
            wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_ph,
            ld_coeffs_quiet=ld_coeffs_quiet, ld_coeffs_active=ld_coeffs_active,
            I_profile_quiet=I_profile_quiet, I_profile_active=I_profile_active,
            mu_profile_pts=mu_profile_pts, x_disc=x_disc, y_disc=y_disc,
            mu_disc=mu_disc, col_idx=col_idx, vel_col=vel_col,
            star_pixel_rad=star_pixel_rad, total_pixels=total_pixels,
            arsize_rads=arsize_ph, ar_smoothness=smooth_ph, k=k, ld_mode=ld_mode,
            plot_map_wavelength=plot_map_wavelength, n=n, flat_indices=flat_indices,
            transit_softness=transit_softness,
        )

    _phase_vmap = vmap(_single_phase, in_axes=(0, 0, 0, 0, 0, 0, 0))
    return _phase_vmap(ar_lat_all, ar_long_all, arsize_rads_all, ar_smoothness_all,
                        flux_active_all, planet_xyz_all, phases_rot)


# ---------------------------------------------------------------------------
# 7. Public API -- light curve
#
#   Stage 1:  build_system()         - NumPy, call once before sampling.
#                                     Pre-builds the grid and all static
#                                     arrays that are fixed across MCMC steps.
#
#   Stage 2:  make_lc() - Pure JAX, call at every MCMC step.
#                                      Accepts JAX arrays / tracers so it is
#                                      fully compatible with jit, vmap, and
#                                      gradient-based samplers.
#
#   quick_lc()            - Convenience wrapper that calls both
#                                      stages in sequence.  Useful for
#                                      one-off calls outside MCMC.
# ---------------------------------------------------------------------------

def _prepare_ld_coeffs(raw, ld_mode: LdMode, nwave: int, label: str,
                       verbose: bool = False) -> np.ndarray:
    """
    Validate and broadcast a set of LDC coefficients to shape (nwave, n_coeffs).

    Used for the quiet photosphere's coefficients in ``build_system``.
    Each element of ``raw`` may be a scalar (broadcast across wavelength)
    or an array of length ``nwave``.
    """
    if ld_mode == "intensity_profile":
        return np.zeros((nwave, 1), dtype=np.float32)

    if ld_mode not in _N_COEFFS:
        raise ValueError(
            f"unknown ld_mode '{ld_mode}'. "
            f"Must be one of {list(_N_COEFFS.keys()) + ['intensity_profile']}."
        )
    n_coeffs = _N_COEFFS[ld_mode]

    if raw is None:
        raise ValueError(
            f"{label} must be provided for ld_mode='{ld_mode}'. "
            f"Expected {n_coeffs} coefficient(s)."
        )
    raw = list(raw) if not isinstance(raw, (list, tuple)) else list(raw)
    if len(raw) != n_coeffs:
        raise ValueError(
            f"{label}: ld_mode='{ld_mode}' expects {n_coeffs} "
            f"coefficient(s) but {len(raw)} were provided."
        )

    coeff_arrays = []
    all_scalar   = True
    for i, coeff in enumerate(raw):
        c = np.asarray(coeff, dtype=np.float32)
        if c.ndim == 0:
            coeff_arrays.append(np.full(nwave, float(c)))
        else:
            if len(c) != nwave:
                raise ValueError(
                    f"{label}[{i}] has length {len(c)} "
                    f"but wavelength grid has {nwave} bins. They must match."
                )
            coeff_arrays.append(c)
            all_scalar = False

    coeffs = np.stack(coeff_arrays, axis=1)  # (nwave, n_coeffs)

    if verbose:
        if all_scalar:
            coeff_str = ", ".join(f"{float(c[0]):.4f}" for c in coeff_arrays)
            print(
                f"{label}: scalar LDCs provided for '{ld_mode}' law "
                f"([{coeff_str}]) - broadcasting across all {nwave} wavelength bins."
            )
        else:
            print(
                f"{label}: per-wavelength LDCs provided for '{ld_mode}' law "
                f"({n_coeffs} coefficient(s), {nwave} wavelength bins)."
            )

    return coeffs


def _prepare_transit_k(
    k, nplanet: int, nwave: int,
) -> jnp.ndarray:
    """
    Validate and broadcast the planet-to-star radius ratio to (nplanet, nwave).

    Accepted input shapes
    ~~~~~~~~~~~~~~~~~~~~~
    - scalar                          -> same value for every planet and wavelength
    - (nplanet,)                      -> one achromatic value per planet,
                                         broadcast across wavelength
    - (nplanet, nwave)                -> fully chromatic, independent per
                                         planet and wavelength
    - (nwave,), only when nplanet==1 and nwave>1
                                      -> legacy single-planet chromatic depth
                                         (pre-multi-planet convention),
                                         unambiguous since it can't collide
                                         with the (nplanet,)=(1,) case when
                                         nwave != 1

    A bare 1D array is *always* interpreted as per-planet (never
    per-wavelength) outside of the narrow nplanet==1 case.
    This matches every other transit parameter's "1D = trailing nplanet
    axis" convention (see ``compute_multi_planet_sky_positions``) and
    removes any nplanet-vs-nwave length-collision ambiguity.

    Parameters
    ----------
    k       : scalar or array_like, one of the shapes above
    nplanet : number of planets
    nwave   : number of wavelength bins

    ``k`` is not forced to a particular dtype here (unlike most other
    build-time-only arrays) -- ``make_lc``'s dynamic-override path may pass a
    traced ``k`` through this function, and gradients w.r.t. ``k`` must keep
    flowing (mirrors the pre-multi-planet code, which likewise left a
    dynamically-overridden ``k`` uncast).

    Returns
    -------
    jnp.ndarray, shape (nplanet, nwave)
    """
    k_arr = jnp.atleast_1d(jnp.asarray(k))

    if k_arr.ndim == 1:
        if k_arr.shape[0] == 1:
            return jnp.broadcast_to(k_arr, (nplanet, nwave))
        if nplanet == 1 and nwave > 1 and k_arr.shape[0] == nwave:
            return k_arr[None, :]   # legacy single-planet chromatic depth
        if k_arr.shape[0] == nplanet:
            return jnp.broadcast_to(k_arr[:, None], (nplanet, nwave))
        raise ValueError(
            f"k shape mismatch: got a 1D array of length "
            f"{k_arr.shape[0]}, but a 1D k is interpreted as "
            f"one achromatic value per planet and must have length "
            f"nplanet ({nplanet})"
            + (f" or, for a single planet, nwave ({nwave})." if nplanet == 1 else ".")
        )

    if k_arr.shape == (nplanet, nwave):
        return k_arr

    raise ValueError(
        f"k shape mismatch: got shape {k_arr.shape}, but expected a "
        f"scalar, ({nplanet},) [per-planet, achromatic], or ({nplanet}, {nwave}) "
        "[per-planet, per-wavelength]."
    )


# ---------------------------------------------------------------------------
# 7b. Shared AR/transit/LDC parameter resolution -- used by make_lc, make_rv,
# and make_lc_and_rv so a call to either given the same arguments describes the
# same physical system.
# ---------------------------------------------------------------------------

def _resolve_ar_params(
    model: dict,
    flux_active: Optional[jnp.ndarray],
    ar_lat: Optional[jnp.ndarray],
    ar_long: Optional[jnp.ndarray],
    ar_size: Optional[jnp.ndarray],
    ar_smoothness: Optional[jnp.ndarray],
) -> dict:
    """
    Resolve the all-or-nothing active-region parameter group: validate,
    detect the static-vs-time-varying path, broadcast/validate shapes, and
    (time-varying path) interpolate onto the oversampled sub-exposure
    times. See ``make_lc``'s docstring for the full parameter semantics --
    this is a behavior-preserving extraction of what used to be make_lc's
    own body, so it is documented there, not duplicated here.

    The static-path AR Cartesian position build (``all_ar_carts``) is
    folded in here too, even though it has no data dependency on the LDC
    resolution that used to sit between it and this block in the original
    make_lc -- callers only need one dict back, and both make_lc and
    make_rv need this exact position array.

    Returns
    -------
    dict with keys:
      nar             : int, number of active regions.
      ar_time_varying : bool.
      flux_active, ar_lat, ar_long, ar_size, ar_smoothness :
          resolved JAX arrays -- static shapes (nar,[nwave]) or
          time-varying shapes (ntime, nar,[nwave]), already interpolated
          onto model["times_oversampled"] and re-clipped to their physical
          domains in the time-varying case. ``ar_size`` is still in
          DEGREES here (not yet converted to radians -- callers do that,
          exactly as make_lc's own dispatch always has).
      all_ar_carts    : (nphase_compute, nar, 3) jnp.ndarray if not
          ar_time_varying, else None.
    """
    # ---- AR parameters: all-or-nothing ------------------------------------
    _ar_args = dict(flux_active=flux_active, ar_lat=ar_lat, ar_long=ar_long,
                     ar_size=ar_size, ar_smoothness=ar_smoothness)
    _ar_given = {name: v for name, v in _ar_args.items() if v is not None}
    if _ar_given and len(_ar_given) != len(_ar_args):
        _missing = [name for name in _ar_args if name not in _ar_given]
        raise ValueError(
            "make_lc: partial active-region parameters given "
            f"({sorted(_ar_given)}); missing {_missing}. Provide all of "
            f"{list(_ar_args)} to add active region(s), or none for a quiet star."
        )
    if not _ar_given:
        # No active region requested: a single AR whose own spectrum and LDC
        # exactly equal the quiet photosphere's has contrast C_a == 1
        # everywhere, so its (1 - C_a) contribution is exactly 0 regardless
        # of its position/size -- i.e. an exact quiet-star light curve.
        flux_active   = model["flux_quiet"]
        ar_lat        = jnp.array([0.0])
        ar_long       = jnp.array([0.0])
        ar_size       = jnp.array([1.0])
        ar_smoothness = jnp.array([1.0])

    flux_active   = jnp.atleast_1d(jnp.asarray(flux_active))
    ar_lat        = jnp.atleast_1d(jnp.asarray(ar_lat))
    ar_long       = jnp.atleast_1d(jnp.asarray(ar_long))
    ar_size       = jnp.atleast_1d(jnp.asarray(ar_size))
    ar_smoothness = jnp.atleast_1d(jnp.asarray(ar_smoothness))

    # Determine number of active regions from the trailing axis, so this
    # works whether or not a parameter carries an extra leading time axis
    nar   = ar_lat.shape[-1]
    nwave = model["nwave"]

    spr             = model["star_pixel_rad"]
    inc_star        = model["inc_star"]
    oversample      = model["oversample"]
    nphase_original = model["nphase_original"]

    # ---- Time-varying active regions (optional) ---------------------------
    # Any of flux_active/ar_lat/ar_long/ar_size/ar_smoothness may carry an
    # extra leading axis of length nphase_original (the model's original,
    # pre-oversampling `times`) to make that property evolve over the
    # observation.
    # No extra axis on any of them (today's usage) takes the usual code path
    # so it costs nothing extra. ld_coeffs_active/ I_profile_active do not support
    # time evolution and stay fixed.
    ar_time_varying = (
        ar_lat.ndim == 2 or ar_long.ndim == 2 or ar_size.ndim == 2
        or ar_smoothness.ndim == 2 or flux_active.ndim == 3
    )

    # ``nar`` was read off ar_lat's trailing axis (above) before we knew
    # whether any parameter was time-varying. If a caller meant a single AR
    # to evolve over ntime epochs but passed one of ar_lat/ar_long/ar_size
    # as a flat (ntime,) array -- forgetting the extra leading axis that
    # would mark it time-varying, i.e. shape (ntime, 1) -- and ntime happens
    # to equal nar, that array's own per-array shape check below passes
    # vacuously (ntime == nar by coincidence), and it silently gets
    # broadcast as ntime separate *static* active regions instead of one
    # evolving one. This can't be told apart from a genuine ntime-active-region
    # model by shape alone, so we warn rather than raise an error. Excluded
    # when nar == 1: with a single AR, "ntime separate static regions" and
    # "one region evolving over ntime epochs" are the same computation, so
    # there's no ambiguity to warn about.
    if ar_time_varying and nar > 1 and nar == nphase_original:
        warnings.warn(
            f"SAJAX: nar (number of active regions, inferred as {nar} from "
            "the trailing axis of ar_lat/ar_long/ar_size) equals ntime "
            f"({nphase_original}, the number of `times` this model was "
            "built with), while at least one active-region parameter "
            "carries an explicit time axis. If you intended a single "
            "time-varying active region and passed ar_lat/ar_long/ar_size/"
            "ar_smoothness as a flat (ntime,) array instead of (ntime, 1), "
            "it will silently be interpreted as ntime separate static "
            "active regions rather than one evolving active region -- give "
            "it an explicit (ntime, nar) shape to evolve it. Ignore this "
            f"warning if {nar} distinct active regions is what you intended."
        )

    if not ar_time_varying:
        # ---- Static path ---
        # Handle broadcasting: if flux_active is (nwave,), broadcast to (nar, nwave)
        if flux_active.ndim == 1:
            if flux_active.size != nwave:
                raise ValueError(
                    f"flux_active shape mismatch: got size {flux_active.size} "
                    f"but wavelength grid has {nwave} bins."
                )
            flux_active = jnp.broadcast_to(flux_active[None, :], (nar, nwave))
        elif flux_active.ndim == 2:
            if flux_active.shape != (nar, nwave):
                raise ValueError(
                    f"flux_active shape mismatch: got {flux_active.shape} "
                    f"but expected ({nar}, {nwave})."
                )
        else:
            raise ValueError(
                f"flux_active must be 1D or 2D, got shape {flux_active.shape}."
            )

        # Broadcast a shared ar_smoothness (scalar or size-1) to all ARs if only one smoothness value is provided
        if ar_smoothness.size == 1:
            ar_smoothness = jnp.broadcast_to(ar_smoothness, (nar,))
        elif ar_smoothness.shape != (nar,):
            raise ValueError(
                f"ar_smoothness shape mismatch: got shape {ar_smoothness.shape} "
                f"but expected a scalar or shape ({nar},)."
            )
        # Check the *computed* boolean result for tracer-ness, not the input:
        # inside an active jax.jit trace (e.g. numpyro_ext/jaxopt's MAP
        # optimizer), a jnp operation always produces a traced output, even
        # when applied to an input that is itself concrete -- so isinstance
        # on ar_smoothness/ar_size/flux_active alone is not a reliable guard.
        _ar_smoothness_check = jnp.any(ar_smoothness < 1)
        if not isinstance(_ar_smoothness_check, jax.core.Tracer) and bool(_ar_smoothness_check):
            raise ValueError(
                f"ar_smoothness must be >= 1 (got {ar_smoothness}); 1 is a "
                "Gaussian AR boundary and larger values sharpen it towards a "
                "top-hat, but values below 1 do not correspond to a physically "
                "meaningful AR shape."
            )
        _ar_size_check = jnp.any(ar_size < 0)
        if not isinstance(_ar_size_check, jax.core.Tracer) and bool(_ar_size_check):
            raise ValueError(
                f"ar_size must be >= 0 (got {ar_size}); ar_size is an "
                "angular radius in degrees and cannot be negative."
            )
        _flux_active_check = jnp.any(flux_active < 0)
        if not isinstance(_flux_active_check, jax.core.Tracer) and bool(_flux_active_check):
            raise ValueError(
                f"flux_active must be >= 0 (got min {float(jnp.min(flux_active))}); "
                "flux_active is a flux/spectrum and cannot be negative."
            )
    else:
        # ---- Time-varying path--
        # static properties (still their old shape) get broadcast across time;
        # time-varying ones (one extra leading axis) are validated as-is.
        ntime = nphase_original

        ar_time_interp_val = model["ar_time_interp"]

        if ar_time_interp_val == "cubic" and ntime < 2:
            raise ValueError(
                "make_lc: ar_time_interp='cubic' needs at least 2 distinct "
                f"times to fit a spline, but the model was built with "
                f"{ntime}. Use ar_time_interp='linear' instead."
            )

        #Expands static properties over time axis and raise warnings
        def _expand_position(name, arr):
            if arr.ndim == 1:
                if arr.shape[0] != nar:
                    raise ValueError(
                        f"{name} shape mismatch: got shape {arr.shape} but "
                        f"expected ({nar},) or ({ntime}, {nar}) for a "
                        "time-varying value."
                    )
                return jnp.broadcast_to(arr[None, :], (ntime, nar))
            if arr.shape != (ntime, nar):
                raise ValueError(
                    f"{name} shape mismatch: got shape {arr.shape} but "
                    f"expected ({nar},) or ({ntime}, {nar})."
                )
            return arr

        ar_lat        = _expand_position("ar_lat", ar_lat)
        ar_long       = _expand_position("ar_long", ar_long)
        ar_size       = _expand_position("ar_size", ar_size)

        _ar_size_check = jnp.any(ar_size < 0)
        if not isinstance(_ar_size_check, jax.core.Tracer) and bool(_ar_size_check):
            raise ValueError(
                f"ar_size must be >= 0 (got min {float(jnp.min(ar_size))}); "
                "ar_size is an angular radius in degrees and cannot be negative."
            )

        if ar_smoothness.ndim == 1:
            if ar_smoothness.size == 1:
                ar_smoothness = jnp.broadcast_to(ar_smoothness, (ntime, nar))
            elif ar_smoothness.shape == (nar,):
                ar_smoothness = jnp.broadcast_to(ar_smoothness[None, :], (ntime, nar))
            else:
                raise ValueError(
                    f"ar_smoothness shape mismatch: got shape {ar_smoothness.shape} "
                    f"but expected a scalar, ({nar},), or ({ntime}, {nar})."
                )
        elif ar_smoothness.shape != (ntime, nar):
            raise ValueError(
                f"ar_smoothness shape mismatch: got shape {ar_smoothness.shape} "
                f"but expected a scalar, ({nar},), or ({ntime}, {nar})."
            )
        _ar_smoothness_check = jnp.any(ar_smoothness < 1)
        if not isinstance(_ar_smoothness_check, jax.core.Tracer) and bool(_ar_smoothness_check):
            raise ValueError(
                f"ar_smoothness must be >= 1 (got min {float(jnp.min(ar_smoothness))}); "
                "1 is a Gaussian AR boundary and larger values sharpen it "
                "towards a top-hat, but values below 1 do not correspond to "
                "a physically meaningful AR shape."
            )

        if flux_active.ndim == 1:
            if flux_active.size != nwave:
                raise ValueError(
                    f"flux_active shape mismatch: got size {flux_active.size} "
                    f"but wavelength grid has {nwave} bins."
                )
            flux_active = jnp.broadcast_to(
                flux_active[None, None, :], (ntime, nar, nwave)
            )
        elif flux_active.ndim == 2:
            if flux_active.shape != (nar, nwave):
                raise ValueError(
                    f"flux_active shape mismatch: got {flux_active.shape} but "
                    f"expected ({nar}, {nwave}) or ({ntime}, {nar}, {nwave})."
                )
            flux_active = jnp.broadcast_to(flux_active[None, :, :], (ntime, nar, nwave))
        elif flux_active.shape != (ntime, nar, nwave):
            raise ValueError(
                f"flux_active shape mismatch: got shape {flux_active.shape} but "
                f"expected (nwave,), ({nar}, {nwave}), or ({ntime}, {nar}, {nwave})."
            )
        _flux_active_check = jnp.any(flux_active < 0)
        if not isinstance(_flux_active_check, jax.core.Tracer) and bool(_flux_active_check):
            raise ValueError(
                f"flux_active must be >= 0 (got min {float(jnp.min(flux_active))}); "
                "flux_active is a flux/spectrum and cannot be negative."
            )

        # ---- Resolve from the original per-cadence grid onto the exact
        # oversampled sub-exposure times already used for the planet's
        # position (model["times_oversampled"]), via interpolation ----
        _interp_method = _AR_TIME_INTERP_METHOD[ar_time_interp_val]
        if model.get("verbose", False) and oversample > 1:
            print(
                f"make_lc: time-varying active-region parameter(s) detected "
                f"with oversample={oversample} -- resolving onto sub-exposure "
                f"times using ar_time_interp='{ar_time_interp_val}' interpolation."
            )
        # model["times"] is stored unshifted (original BJD-scale epochs) but
        # model["times_oversampled"] was shifted by model["t_ref"] in
        # build_system -- both sides of the interpolation must be on the same scale,
        # or querying times_oversampled against the unshifted grid silently extrapolates.
        _times_orig = model["times"] - model["t_ref"]
        _times_over = model["times_oversampled"]
        ar_lat        = interpax.interp1d(_times_over, _times_orig, ar_lat, method=_interp_method, extrap=True)
        ar_long       = interpax.interp1d(_times_over, _times_orig, ar_long, method=_interp_method, extrap=True)
        ar_size       = interpax.interp1d(_times_over, _times_orig, ar_size, method=_interp_method, extrap=True)
        ar_smoothness = interpax.interp1d(_times_over, _times_orig, ar_smoothness, method=_interp_method, extrap=True)
        flux_active   = interpax.interp1d(_times_over, _times_orig, flux_active, method=_interp_method, extrap=True)

        # ar_time_interp="cubic" is a natural spline and can overshoot the
        # input node values between/around them even though the nodes
        # themselves satisfy ar_size >= 0 / ar_smoothness >= 1 / flux_active
        # >= 0 (checked above) -- re-clip post-interpolation so no
        # sub-exposure time sees an out-of-range value (otherwise ar_size
        # crossing zero would make the AR boundary bounce unphysically, see
        # _compute_ar_shape).
        if model.get("verbose", False):
            # Best-effort diagnostic only -- even when ar_size itself looks
            # concrete, jnp.sum(...) on it can still come back traced inside
            # an active jax.jit trace (e.g. a MAP optimizer), so int() may
            # legitimately fail here; silently skip the message rather than
            # let a verbose-only diagnostic crash the actual computation.
            try:
                n_size_clipped   = int(jnp.sum(ar_size < 0))
                n_smooth_clipped = int(jnp.sum(ar_smoothness < 1))
                n_flux_clipped   = int(jnp.sum(flux_active < 0))
            except jax.errors.ConcretizationTypeError:
                n_size_clipped = n_smooth_clipped = n_flux_clipped = 0
            if n_size_clipped or n_smooth_clipped or n_flux_clipped:
                print(
                    f"make_lc: ar_time_interp='{ar_time_interp_val}' interpolation "
                    f"overshot the physical domain at {n_size_clipped} sub-exposure "
                    f"ar_size value(s), {n_smooth_clipped} sub-exposure ar_smoothness "
                    f"value(s), and {n_flux_clipped} sub-exposure flux_active "
                    "value(s) -- clipping to ar_size >= 0, ar_smoothness >= 1, "
                    "and flux_active >= 0."
                )
        ar_size       = jnp.maximum(ar_size, 0.0)
        ar_smoothness = jnp.maximum(ar_smoothness, 1.0)
        flux_active   = jnp.maximum(flux_active, 0.0)

    if not ar_time_varying:
        # ---- Static path: build the AR's Cartesian position once --
        ar_lat_rad  = jnp.deg2rad(ar_lat)
        ar_long_rad = jnp.deg2rad(ar_long)

        ar_cart = jnp.stack([
            spr * jnp.sin(ar_long_rad) * jnp.cos(ar_lat_rad),
            spr * jnp.sin(ar_lat_rad),
            spr * jnp.cos(ar_long_rad) * jnp.cos(ar_lat_rad),
        ], axis=-1)   # (nar, 3)

        # phases_rot in the model is already oversampled if oversample > 1
        def _rotate_ars_at_phase(phase_deg):
            return vmap(
                lambda cart: rotate_active_region(cart, phase_deg, inc_star)
            )(ar_cart)

        all_ar_carts = vmap(_rotate_ars_at_phase)(
            model["phases_rot"]
        )   # (nphase_compute, nar, 3)
    else:
        all_ar_carts = None

    return dict(
        nar=nar,
        ar_time_varying=ar_time_varying,
        flux_active=flux_active,
        ar_lat=ar_lat,
        ar_long=ar_long,
        ar_size=ar_size,
        ar_smoothness=ar_smoothness,
        all_ar_carts=all_ar_carts,
    )


def _resolve_ld_coeffs(
    model: dict,
    ld_coeffs_quiet: Optional[jnp.ndarray],
    ld_coeffs_active: Optional[jnp.ndarray],
    I_profile_active: Optional[jnp.ndarray],
    nar: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Resolve the quiet photosphere's (possibly dynamically-overridden) LDC
    coefficients, and default/broadcast each active region's own LDC
    coefficients and intensity profile. Behavior-preserving extraction of
    what used to be make_lc's own body -- see make_lc's docstring for the
    full parameter semantics.

    Returns
    -------
    (ld_coeffs_quiet_val, ld_coeffs_active, I_profile_active)
    """
    nwave = model["nwave"]
    ld_mode = model["ld_mode"]
    n_coeffs = 1 if ld_mode == "intensity_profile" else _N_COEFFS[ld_mode]

    # ---- Quiet photosphere's own LDC coefficients: static (model-built)
    # value by default, or a per-call (possibly traced) override -----------
    if ld_coeffs_quiet is None:
        ld_coeffs_quiet_val = model["ld_coeffs"]
    else:
        ld_coeffs_quiet_val = jnp.asarray(ld_coeffs_quiet)
        if ld_coeffs_quiet_val.shape != (nwave, n_coeffs):
            raise ValueError(
                f"ld_coeffs_quiet shape mismatch: got {ld_coeffs_quiet_val.shape} "
                f"but expected ({nwave}, {n_coeffs})."
            )

    # ---- Per-AR limb-darkening coefficients: default to the quiet
    # photosphere's own (dynamic override included), otherwise broadcast
    # (nwave, n_coeffs) -> (nar, ...)
    if ld_coeffs_active is None:
        ld_coeffs_active = jnp.broadcast_to(
            ld_coeffs_quiet_val[None, :, :], (nar, nwave, n_coeffs)
        )
    else:
        ld_coeffs_active = jnp.asarray(ld_coeffs_active)
        if ld_coeffs_active.ndim == 2:
            if ld_coeffs_active.shape != (nwave, n_coeffs):
                raise ValueError(
                    f"ld_coeffs_active shape mismatch: got {ld_coeffs_active.shape} "
                    f"but expected ({nwave}, {n_coeffs})."
                )
            ld_coeffs_active = jnp.broadcast_to(
                ld_coeffs_active[None, :, :], (nar, nwave, n_coeffs)
            )
        elif ld_coeffs_active.shape != (nar, nwave, n_coeffs):
            raise ValueError(
                f"ld_coeffs_active shape mismatch: got {ld_coeffs_active.shape} "
                f"but expected ({nar}, {nwave}, {n_coeffs})."
            )

    n_mu_pts = model["mu_profile_pts"].shape[0]
    if I_profile_active is None:
        I_profile_active = jnp.broadcast_to(
            model["I_profile"][None, :, :], (nar, nwave, n_mu_pts)
        )
    else:
        I_profile_active = jnp.asarray(I_profile_active)
        if I_profile_active.ndim == 2:
            if I_profile_active.shape != (nwave, n_mu_pts):
                raise ValueError(
                    f"I_profile_active shape mismatch: got {I_profile_active.shape} "
                    f"but expected ({nwave}, {n_mu_pts})."
                )
            I_profile_active = jnp.broadcast_to(
                I_profile_active[None, :, :], (nar, nwave, n_mu_pts)
            )
        elif I_profile_active.shape != (nar, nwave, n_mu_pts):
            raise ValueError(
                f"I_profile_active shape mismatch: got {I_profile_active.shape} "
                f"but expected ({nar}, {nwave}, {n_mu_pts})."
            )

    return ld_coeffs_quiet_val, ld_coeffs_active, I_profile_active


def _resolve_transit_params(
    model: dict,
    t0: Optional[float | jnp.ndarray],
    period: Optional[float | jnp.ndarray],
    a_over_rstar: Optional[float | jnp.ndarray],
    inclination: Optional[float | jnp.ndarray],
    ecc: Optional[float | jnp.ndarray],
    omega_peri: Optional[float | jnp.ndarray],
    sp_orb: Optional[float | jnp.ndarray],
    k: Optional[float | jnp.ndarray],
    nwave: int,
) -> dict:
    """
    Resolve the all-or-nothing transit parameter group: validate, dispatch
    to the dynamic-override / static (model-built) / dummy (no transit)
    path, and broadcast ``k`` to ``(nplanet, nwave)``. Behavior-preserving
    extraction of what used to be make_lc's own body -- see make_lc's
    docstring for the full parameter semantics.

    Extended (beyond make_lc's own needs) to also resolve and return
    ``t0_val``/``period_val``/``ecc_val``/``omega_peri_val``/
    ``inclination_val`` -- the exact orbital elements consistent with
    ``planet_xyz_all``, on the same ``t_ref``-shifted timescale -- across
    ALL THREE branches (make_lc's original code only ever named these
    inline, in the dynamic branch, since only compute_multi_planet_sky_positions
    needed them there). ``make_rv``'s Keplerian term needs them from
    whichever branch actually ran, so they must not be re-derived or
    re-validated a second time by callers.

    Returns
    -------
    dict with keys:
      planet_xyz_all : (nphase_compute, nplanet, 3)
      k_val          : (nplanet, nwave)
      nplanet        : int
      has_transit    : bool -- True for the static or dynamic path, False
          for the dummy (no-transit) path.
      t0_val, period_val, ecc_val, omega_peri_val, inclination_val :
          resolved orbital elements consistent with planet_xyz_all, or
          None if has_transit is False.
    """
    # ---- Transit parameters: all-or-nothing (required 5), ecc/omega_peri
    # only meaningful alongside them -------------------------------------
    _transit_required = dict(t0=t0, period=period, a_over_rstar=a_over_rstar,
                               inclination=inclination, k=k)
    _transit_optional  = dict(ecc=ecc, omega_peri=omega_peri, sp_orb=sp_orb)
    _transit_given = {name: v for name, v in {**_transit_required, **_transit_optional}.items()
                       if v is not None}
    if _transit_given:
        _missing_required = [name for name, v in _transit_required.items() if v is None]
        if _missing_required:
            raise ValueError(
                "make_lc: partial transit parameters given "
                f"({sorted(_transit_given)}); missing {_missing_required}. Provide all "
                f"of {list(_transit_required)} to evaluate a transit "
                "(ecc/omega_peri/sp_orb are optional and default to the model's "
                "build-time values), or none to leave the transit as-is."
            )
        if not model.get("has_transit", False):
            raise ValueError(
                "make_lc: transit parameters were given, but this model "
                "has no transit attached. Build it with build_system(...) first."
            )

    # ---- Planet positions (if a transit model is present) ---------------
    if _transit_given:
        # Dynamic path: recompute positions from (possibly traced) orbital
        # parameters every call, exactly like ar_cart is recomputed from
        # ar_lat/ar_long every call. k_val is deliberately left as-is here
        # (uncast) so gradients w.r.t. k propagate through
        # _compute_all_planets_mask, which is written to accept a traced k.
        # nplanet is inferred fresh from this call's own t0 -- if
        # ecc/omega_peri/sp_orb are *not* overridden here and the caller
        # changed nplanet via t0, the stale-length fallback array below will
        # make compute_multi_planet_sky_positions raise a clear shape error;
        # override ecc/omega_peri/sp_orb too when changing nplanet dynamically.
        _defaults  = model["transit_params"]
        ecc_val        = ecc        if ecc        is not None else _defaults.get("ecc", 0.0)
        omega_peri_val = omega_peri if omega_peri is not None else _defaults.get("omega_peri", 0.0)
        sp_orb_deg     = sp_orb     if sp_orb     is not None else _defaults.get("sp_orb", 0.0)
        sp_orb_val     = jnp.deg2rad(sp_orb_deg)   # sp_orb is given in degrees; planet.py takes radians
        # model["times_oversampled"] was already shifted by model["t_ref"] in
        # build_system; t0 (possibly traced/sampled here) must be shifted by
        # the same constant so time - t_peri still cancels correctly.
        t0_val = t0 - model["t_ref"]
        planet_xyz_all = compute_multi_planet_sky_positions(
            model["times_oversampled"], t0_val, period, a_over_rstar, inclination,
            ecc_val, omega_peri_val, sp_orb_val,
        )   # (nphase_compute, nplanet, 3)
        nplanet = jnp.atleast_1d(jnp.asarray(t0)).shape[-1]
        k_val = k
        period_val      = period
        inclination_val = inclination
        has_transit     = True
    elif model.get("has_transit", False):
        # Static, backwards-compatible path: positions baked at build time.
        planet_xyz_all = model["planet_xyz"]    # (nphase_compute, nplanet, 3)
        k_val          = model["k"]             # (nplanet, nwave), already dense
        nplanet        = model["nplanet"]
        # Resolve the same orbital elements build_system used to bake
        # planet_xyz_all, on the same t_ref-shifted timescale, so make_rv's
        # Keplerian term stays consistent with this static transit.
        _defaults       = model["transit_params"]
        t0_val          = jnp.asarray(_defaults["t0"]) - model["t_ref"]
        period_val      = jnp.asarray(_defaults["period"])
        ecc_val         = jnp.asarray(_defaults.get("ecc", 0.0))
        omega_peri_val  = jnp.asarray(_defaults.get("omega_peri", 0.0))
        inclination_val = jnp.asarray(_defaults["inclination"])
        has_transit     = True
    else:
        # Dummy: planet permanently behind the star, zero-radius disc.
        nphase_compute = model["phases_rot"].shape[0]
        nplanet        = 1
        planet_xyz_all = jnp.zeros((nphase_compute, 1, 3)).at[:, :, 2].set(-1e10)
        k_val          = 0.0
        t0_val = period_val = ecc_val = omega_peri_val = inclination_val = None
        has_transit    = False

    # ---- Broadcast k to (nplanet, nwave): see _prepare_transit_k for the
    # full shape convention. A scalar means the same (achromatic) radius
    # ratio at every planet/wavelength; a genuinely chromatic and/or
    # per-planet depth is supported (the occultation mask is computed per
    # wavelength -- see _flux_at_wavelength -- specifically to support this).
    # Idempotent (a no-op) on the already-dense static-path value above.
    k_val = _prepare_transit_k(k_val, nplanet, nwave)

    return dict(
        planet_xyz_all=planet_xyz_all,
        k_val=k_val,
        nplanet=nplanet,
        has_transit=has_transit,
        t0_val=t0_val,
        period_val=period_val,
        ecc_val=ecc_val,
        omega_peri_val=omega_peri_val,
        inclination_val=inclination_val,
    )


def build_system(
    wavelength: np.ndarray,
    flux_quiet: np.ndarray,
    times: np.ndarray,
    P_rot: float,
    stellar_grid_size: int = 100,
    ve: float = 0.0,
    ld_coeffs: Optional[list] = None,
    inc_star: float = 90.0,
    mu_profile: Optional[np.ndarray] = None,
    I_profile: Optional[np.ndarray] = None,
    ld_mode: LdMode = "quadratic",
    plot_map_wavelength: Optional[float] = None,
    oversample: int = 1,
    ar_time_interp: ArTimeInterp = "linear",
    t0: Optional[float] = None,
    period: Optional[float] = None,
    a_over_rstar: Optional[float] = None,
    inclination: Optional[float] = None,
    k: Optional[float | np.ndarray] = None,
    ecc: float = 0.0,
    omega_peri: float = 0.0,
    sp_orb: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Pre-build all static model arrays.  Call this **once** before MCMC.

    Everything that does not change between MCMC steps is computed here in
    NumPy and stored in the returned model dict.  The only quantities that
    vary per step -- ``flux_active``, ``ar_lat``, ``ar_long``, ``ar_size``,
    ``ar_smoothness``, and each active region's own limb-darkening
    coefficients -- are intentionally excluded and passed to
    ``make_lc`` instead (they may of course still be held
    fixed at every step if you don't want to sample/optimize them).

    ``times``/``P_rot`` -- not a bare rotational-phase array -- are the
    only user-facing time input: the internal rotational phase grid
    (``phases_rot = (times / P_rot * 360) % 360``) is always derived from
    them, since a bare phase array wraps every 360deg and can't recover the
    absolute time reference a transit needs.

    Transit (optional, all-or-nothing) -- give every one of ``t0``,
    ``period``, ``a_over_rstar``, ``inclination``, ``k`` to attach a
    planetary transit to the model, or omit all five for a quiet-star/
    active-region-only model. Multiple planets are supported the same way
    active regions are: give ``t0`` (and, as needed, the other transit
    parameters) a trailing ``(nplanet,)`` axis instead of a scalar --
    ``nplanet`` is inferred from ``t0``. When occulting a starspot or
    facula, the planet mask is applied at the individual pixel level, so the
    resulting light-curve anomaly is computed correctly -- see
    ``make_lc`` for details on how this interacts with active
    regions. Compared to multiplying independent stellar and transit light
    curves, this correctly handles:

    - Planet occulting a spot (spot-crossing anomaly).
    - Planet occulting a facula (facula-crossing anomaly).
    - The varying limb-darkening depth of the transit as a function of
      the stellar surface brightness profile.

    Parameters
    ----------
    wavelength : array_like, shape (nwave,)
        Wavelength value or array at which the quiet photosphere and active-region spectra are defined.
    flux_quiet : array_like, shape (nwave,)
        Quiet-photosphere flux / spectrum.
    times : array_like, shape (ntime,)
        Absolute observation times [days].
    P_rot : float
        Stellar rotation period [days].
    stellar_grid_size : int
        Radius of the stellar grid.
    ve : float
        Stellar equatorial rotational velocity [km/s].
    ld_coeffs : list of float or list of array(nwave,), optional
        Quiet-photosphere limb-darkening coefficients for the chosen
        ``ld_mode``:
        - ``"linear"``:     [u]
        - ``"quadratic"``:  [u1, u2]
        - ``"power2"``:     [c, alpha]
        - ``"kipping3"``:   [c1, c2, c3]
        - ``"nonlinear4"``: [c1, c2, c3, c4]
        Each element may be a scalar (broadcast to all wavelengths) or an
        array of length ``nwave``. Active regions carry their own,
        independent coefficients -- see ``make_lc``. Not used
        (and not required) when ``ld_mode="intensity_profile"``; its
        required length for every other mode is checked against
        ``ld_mode`` here.
    inc_star : float, optional
        Stellar inclination in degrees (default: 90.0).
        90deg = equator-on, 0deg = pole-on.
    mu_profile : array-like, optional
        Monotonically increasing mu grid points for
        ``ld_mode="intensity_profile"`` (default: [0, 1]).
    I_profile : array-like, shape (nwave, n_mu_pts), optional
        Quiet-photosphere specific intensity at each (wavelength, mu) grid
        point. Required when ``ld_mode="intensity_profile"``.
    ld_mode : str (default "quadratic")
        Limb-darkening law, shared by the quiet photosphere and every
        active region (each with its own coefficient values).
    plot_map_wavelength : float, optional
        Wavelength at which to plot the stellar map (see ``build_system``).
    oversample : int, optional
        Number of sub-exposures per phase point.  Each requested phase is
        spread into ``oversample`` uniformly spaced sub-phases spanning one
        phase step, and the resulting fluxes are averaged.  This mimics
        finite-exposure integration and smooths limb-crossing artefacts.
        Default: 1 (no oversampling).
    ar_time_interp : "linear" or "cubic", optional (default "linear")
        Interpolation method for when time-varying active-region parameters
        need to be interpolated onto ``oversample``'d sub-exposures.
        Only matters when ``oversample > 1`` and at least one AR parameter
        is time-varying -- see ``make_lc`` for the full description.
    t0, period, a_over_rstar, inclination : float or array-like, shape (nplanet,), optional
        Transit-geometry parameters: mid-transit epoch, orbital period [days],
        semi-major axis/R*, and orbital inclination [rad]. All-or-nothing with
        ``k``. Each is a scalar or an array with a trailing ``(nplanet,)``
        axis -- exactly like ``ar_lat``/``ar_long``/etc. for active regions
        (see ``make_lc``) -- with ``nplanet`` inferred from ``t0``'s trailing
        axis. A scalar (or size-1 array) among the others broadcasts to every
        planet; giving more than one planet requires giving each of these
        four its own length-``nplanet`` array (or leaving it scalar to share
        one value across all planets).
    k : float or array-like, shape (nplanet,) or (nplanet, nwave), optional
        Planet-to-star radius ratio Rp/R*. A scalar (the same value for
        every planet and wavelength), an array of length ``nplanet`` (one
        achromatic value per planet), or an array of shape
        ``(nplanet, nwave)`` (a chromatic transit depth, independent per
        planet). For a single planet (``nplanet == 1``), a bare array of
        length ``nwave`` is also accepted as that planet's chromatic depth
        (legacy convention). Multiple overlapping planets combine their
        occultation multiplicatively (they are opaque, unlike active
        regions' additive contrast -- see ``make_lc``), so overlapping
        transits never drive flux negative.
    ecc, omega_peri : float or array-like, shape (nplanet,), optional
        Orbital eccentricity and argument of periastron [rad]. Only
        meaningful together with a transit; default to 0.0 (circular,
        non-precessing orbit). Scalar or ``(nplanet,)``, same broadcasting
        rule as ``t0``/``period``/etc.
    sp_orb : float or jnp.ndarray, shape (nplanet,), optional
        sky-projected spin-orbit angle, λ  [deg]
        Rotates the transit chord about the stellar
        centre, in the sky plane. Angle is relative to the
        stellar equator. Only meaningful together with a transit.
        Converted to radians internally before use --
        ``sajax.planet.planet_sky_position`` itself takes radians.
        Scalar or ``(nplanet,)``, same broadcasting rule as ``t0``/``period``/etc.
    verbose : bool, optional
        If True, print informational messages (LDC broadcasting, phase
        oversampling) while building the model. Default False.

    Returns
    -------
    dict  - pass directly to ``make_lc``
    """
    # Validate oversample
    if not isinstance(oversample, int) or oversample < 1:
        raise ValueError(
            f"oversample must be an integer >= 1, got {oversample}."
        )

    # ---- Rotational phase grid, derived internally from times/P_rot --------
    times_arr_full = np.asarray(times, dtype=np.float64)
    phases_rot = (times_arr_full / P_rot * 360.0) % 360.0

    wavelength = np.atleast_1d(np.asarray(wavelength, dtype=np.float32))
    flux_quiet = np.atleast_1d(np.asarray(flux_quiet,  dtype=np.float32))
    phases_rot = np.atleast_1d(np.asarray(phases_rot, dtype=np.float32))

    nwave  = len(wavelength)
    nphase = len(phases_rot)  # original number of phases (before oversampling)

    # ---- Phase and absolute time oversampling -------------------------------------------------
    if oversample > 1:
        phases_oversampled = _make_oversampled_phases(phases_rot, oversample)
        nphase_compute = len(phases_oversampled)
        if verbose:
            print(
                f"build_system: oversampling enabled - {oversample} sub-exposures "
                f"per phase ({nphase} phases -> {nphase_compute} sub-phases)."
            )
    else:
        phases_oversampled = phases_rot
        nphase_compute = nphase

    # ---- Reference epoch -----------------------------------------------------
    # Subtracted from times and, for a transit, t0, before either is cast to
    # a (possibly float32) JAX array, so downstream numerical work operates
    # on small numbers regardless of jax_enable_x64 -- e.g. the mean-anomaly
    # cancellation `time - t_peri` in planet.py, but also anything else that
    # might consume model["times"]/model["times_oversampled"] at BJD scale.
    t_ref = float(np.floor(np.min(times_arr_full)))
    times_arr_shifted = times_arr_full - t_ref
    if oversample > 1:
        n_t = len(times_arr_full)
        dt  = (times_arr_full[1] - times_arr_full[0]) if n_t > 1 else P_rot
        # Same offset scheme used in _make_oversampled_phases -- results align.
        offsets = np.linspace(-dt / 2.0, dt / 2.0, oversample, endpoint=False)
        offsets += dt / (2.0 * oversample)                          # centre sub-bins
        times_oversampled = (
            times_arr_shifted[:, None] + offsets[None, :]
        ).ravel()
    else:
        times_oversampled = times_arr_shifted

    inc_star       = float(inc_star)
    mu_profile_pts = np.asarray(mu_profile if mu_profile is not None else [0.0, 1.0],
                                dtype=np.float32)
    if not np.all(np.diff(mu_profile_pts) > 0):
        raise ValueError(
            "build_system: 'mu_profile' must be strictly increasing. "
            f"Got: {mu_profile_pts}"
        )
    I_profile = np.asarray(
        I_profile if I_profile is not None
        else np.ones((nwave, len(mu_profile_pts)), dtype=np.float32),
        dtype=np.float32,
    )

    ld_coeffs = _prepare_ld_coeffs(ld_coeffs, ld_mode, nwave, label="build_system: quiet ld_coeffs",
                                   verbose=verbose)

    grid = build_stellar_grid(stellar_grid_size, ve, inc_star)

    if plot_map_wavelength is None:
        plot_map_wavelength = float(wavelength[nwave // 2])

    model = dict(
        # spectral
        wavelength          = jnp.asarray(wavelength),
        flux_quiet          = jnp.asarray(flux_quiet),
        ld_coeffs          = jnp.asarray(ld_coeffs),
        I_profile           = jnp.asarray(I_profile),
        mu_profile_pts      = jnp.asarray(mu_profile_pts),
        # grid
        x_disc              = jnp.asarray(grid["x"]),
        y_disc              = jnp.asarray(grid["y"]),
        mu_disc             = jnp.asarray(grid["mu"]),
        col_idx             = jnp.asarray(grid["col_idx"]),
        vel_col             = jnp.asarray(grid["vel_col"]),
        star_pixel_rad      = grid["star_pixel_rad"],
        total_pixels        = grid["total_pixels"],
        n                   = grid["n"],
        flat_indices        = jnp.asarray(grid["flat_indices"]),
        phases_rot          = jnp.asarray(phases_oversampled),
        oversample          = oversample,
        ar_time_interp      = ar_time_interp,
        nphase_original     = nphase,
        t_ref               = t_ref,
        times               = times_arr_full,               # original, unshifted
        times_oversampled   = jnp.asarray(times_oversampled), # shifted -- safe to cast
        inc_star            = inc_star,
        ld_mode            = ld_mode,
        plot_map_wavelength = float(plot_map_wavelength),
        nwave               = nwave,
        nphase              = nphase_compute,
        verbose             = verbose,
    )

    # ---- Optional transit ----------------------------------------------------
    transit_args_given = dict(t0=t0, period=period, a_over_rstar=a_over_rstar,
                               inclination=inclination, k=k)
    given = [v is not None for v in transit_args_given.values()]
    if any(given) and not all(given):
        raise ValueError(
            "build_system: t0/period/a_over_rstar/inclination/k are "
            "all-or-nothing -- give every one of them to attach a transit, "
            "or omit all five for a quiet-star/active-region-only model."
        )
    if all(given):
        from .planet import build_transit_model   # local import avoids circular dep.

        # ---- nplanet is inferred from t0's trailing axis, exactly like nar
        # is inferred from ar_lat's trailing axis in make_lc -----------------
        t0_arr  = np.atleast_1d(np.asarray(t0, dtype=np.float64))
        nplanet = t0_arr.shape[-1]

        # ---- Validate/broadcast k against nplanet and the wavelength grid --
        k_dense = _prepare_transit_k(k, nplanet, nwave)

        # ---- Array-preserving normalisation of the other orbital elements --
        period_arr       = np.atleast_1d(np.asarray(period, dtype=np.float32))
        a_over_rstar_arr = np.atleast_1d(np.asarray(a_over_rstar, dtype=np.float32))
        inclination_arr  = np.atleast_1d(np.asarray(inclination, dtype=np.float32))
        ecc_arr          = np.atleast_1d(np.asarray(ecc, dtype=np.float32))
        omega_peri_arr   = np.atleast_1d(np.asarray(omega_peri, dtype=np.float32))
        sp_orb_arr       = np.atleast_1d(np.asarray(sp_orb, dtype=np.float32))

        # ---- Build transit model (planet positions at oversampled times) ---
        transit = build_transit_model(
            times        = times_oversampled,
            t0           = t0_arr - t_ref,
            period       = period_arr,
            a_over_rstar = a_over_rstar_arr,
            inclination  = inclination_arr,
            ecc          = ecc_arr,
            omega_peri   = omega_peri_arr,
            k            = k_dense,
            sp_orb       = np.deg2rad(sp_orb_arr),   # sp_orb is given in degrees; planet.py takes radians
        )

        # ---- Attach transit data to the model dict --------------------------
        model["planet_xyz"]  = transit["planet_xyz"]   # (nphase_compute, nplanet, 3) -- static, from the
                                                        #   concrete parameters given here
        model["k"]           = transit["k"]            # (nplanet, nwave)
        model["nplanet"]     = transit["nplanet"]
        model["has_transit"] = True
        model["P_rot"]       = P_rot
        # Defaults for make_lc's dynamic path: t0/period/a_over_rstar/
        # inclination/k must all be given together (as individual keyword args,
        # possibly traced) to override these; ecc/omega_peri/sp_orb may be
        # overridden independently and fall back to the values stored here.
        model["transit_params"] = dict(
            t0=t0, period=period, a_over_rstar=a_over_rstar,
            inclination=inclination, ecc=ecc, omega_peri=omega_peri, k=k,
            sp_orb=sp_orb,
        )

    return model


def make_lc(
    model: dict,
    flux_active: Optional[jnp.ndarray] = None,
    ar_lat: Optional[jnp.ndarray] = None,
    ar_long: Optional[jnp.ndarray] = None,
    ar_size: Optional[jnp.ndarray] = None,
    ar_smoothness: Optional[jnp.ndarray] = None,
    ld_coeffs_active: Optional[jnp.ndarray] = None,
    I_profile_active: Optional[jnp.ndarray] = None,
    ld_coeffs_quiet: Optional[jnp.ndarray] = None,
    t0: Optional[float | jnp.ndarray] = None,
    period: Optional[float | jnp.ndarray] = None,
    a_over_rstar: Optional[float | jnp.ndarray] = None,
    inclination: Optional[float | jnp.ndarray] = None,
    ecc: Optional[float | jnp.ndarray] = None,
    omega_peri: Optional[float | jnp.ndarray] = None,
    sp_orb: Optional[float | jnp.ndarray] = None,
    k: Optional[float | jnp.ndarray] = None,
    transit_softness: float = 0.0,
) -> tuple:
    """
    Evaluate the light curve for a given set of active region and planetary parameters.

    This function is **pure JAX** -- all inputs may be JAX arrays or tracers,
    making it fully compatible with ``jit``, ``vmap``, and gradient-based
    samplers such as ``emcee_jax`` or ``blackjax``.

    When the model was built with ``oversample > 1``, the computation runs
    on the oversampled phase grid and the results are averaged back to the
    original phase grid before returning.

    Parameters
    ----------
    model : dict
        Pre-built model dict returned by ``build_system``.
    flux_active : jnp.ndarray, shape (nar, nwave), (nwave,), or (ntime, nar, nwave), optional
        Per-active-region flux / spectrum. Must be >= 0.
        - If (nwave,):     broadcasts to all active regions.
        - If (nar, nwave): each active region gets its own spectrum.
        - If (ntime, nar, nwave): each active region gets its own time-varying spectrum.
    ar_lat : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region latitudes in degrees. Must be in [-90, 90].
    ar_long : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region longitudes in degrees. Must be in [0, 360).
    ar_size : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region angular radii in degrees. Must be >= 0.
    ar_smoothness : jnp.ndarray, shape (nar,), scalar, or (ntime, nar), optional
        Super-Gaussian order controlling the sharpness of each AR's
        boundary (see ``_compute_ar_shape``). ``1`` is a true Gaussian;
        larger values sharpen the edge, converging to a hard-edged cap as
        ``ar_smoothness -> inf``. A scalar (or size-1 array) is broadcast
        to all active regions.

        ``flux_active``/``ar_lat``/``ar_long``/``ar_size``/``ar_smoothness``
        are all-or-nothing: give every one of them to add active region(s),
        or omit all five for a quiet star. Giving some but not all raises
        ``ValueError``.

        **Time-varying active regions.** Any of the above five properties may
        independently carry an extra leading axis of length ``ntime`` instead
        of its usual shape, to let that property evolve over the observation,
        e.g. a spot that grows/decays (``ar_size``), drifts in latitude/longitude
        (``ar_lat``/``ar_long``), or changes contrast (``flux_active``).
        Values are given per original time and interpolated internally onto
        ``oversample`` sub-exposures when oversampling is active, using the
        model's ``ar_time_interp`` law (set at ``build_system`` time, like
        ``ld_mode`` -- see there for the "linear"/"cubic" choice). Mixing is
        allowed: only the parameters you want to evolve need the extra axis,
        the rest keep their usual constant-in-time shape. Using none of them
        (the default) is exactly as fast as before this capability existed.
        ``ld_coeffs_active``/``I_profile_active`` do not support time evolution
        and always stay fixed across the observation. Each epoch's values are
        used exactly as given -- nothing couples them across epochs, so it's
        up to the user to keep a fitted active region from changing unphysically
        fast between epochs. If ``nar`` (read off the parameters' own trailing axis)
        happens to equal ``ntime``, make_lc warns: this is the shape signature of
        the common mistake of leaving a parameter meant to evolve as a flat ``(ntime,)``
        array instead of ``(ntime, 1)``, which silently produces ``ntime`` separate
        static active regions instead of one evolving one.
    ld_coeffs_active : jnp.ndarray, shape (nar, nwave, n_coeffs) or (nwave, n_coeffs), optional
        Per-active-region limb-darkening coefficients, same law as the
        quiet photosphere (``model["ld_mode"]``) but independent values.
        A (nwave, n_coeffs) array broadcasts to all active regions.
        Defaults to the quiet photosphere's own coefficients if omitted.
        Not used when ``ld_mode="intensity_profile"``.
    I_profile_active : jnp.ndarray, shape (nar, nwave, n_mu_pts) or (nwave, n_mu_pts), optional
        Per-active-region specific-intensity profile, used only when
        ``ld_mode="intensity_profile"``. Defaults to the quiet
        photosphere's own profile if omitted.
    ld_coeffs_quiet : jnp.ndarray, shape (nwave, n_coeffs), optional
        Dynamic override for the quiet photosphere's own limb-darkening
        coefficients (JAX values/tracers are fine) -- the build-time
        ``ld_coeffs`` given to ``build_system`` is
        otherwise fixed for the life of the model, exactly like the
        transit-geometry parameters were before they got this same
        treatment. Defaults to the static value ``model`` was built with.
        When given, active regions that don't specify their own
        ``ld_coeffs_active`` default to this (possibly traced) value too,
        instead of the static one.
    t0, period, a_over_rstar, inclination : float or jnp.ndarray, shape (nplanet,), optional
        Transit-geometry parameters: mid-transit epoch, orbital period [days],
        semi-major axis/R*, and orbital inclination [rad]. Scalar or
        ``(nplanet,)``, same trailing-axis convention as ``build_system``
        (``nplanet`` inferred from ``t0``). When overriding a model built
        with a different ``nplanet``, also override ``ecc``/``omega_peri``/
        ``sp_orb`` if you don't want them to fall back to the build-time defaults.
    k : float or array-like, shape (nplanet,) or (nplanet, nwave), optional
        Planet-to-star radius ratio Rp/R*. A scalar (the same value for
        every planet and wavelength), an array of length ``nplanet`` (one
        achromatic value per planet), or an array of shape
        ``(nplanet, nwave)`` (a chromatic transit depth, independent per
        planet). For a single planet (``nplanet == 1``), a bare array of
        length ``nwave`` is also accepted as that planet's chromatic depth
        (legacy convention). Multiple overlapping planets combine their
        occultation multiplicatively (they are opaque, unlike active
        regions' additive contrast -- see ``make_lc``), so overlapping
        transits never drive flux negative.
        The planet's parameters are all-or-nothing, exactly like the AR parameters:
        give every one of them to evaluate a transit at those (possibly
        traced) values instead of the static ones ``model`` was built
        with, or omit all five to fall back to the model's static transit
        (or to no transit, if it doesn't have one) -- ``ValueError`` if
        only some are given.
    ecc, omega_peri : float or jnp.ndarray, shape (nplanet,), optional
        Orbital eccentricity and argument of periastron [rad]. Only
        meaningful together with a transit; default to 0.0 (circular,
        non-precessing orbit). Only used together with the five required
        transit parameters above (giving either of these without the rest
        also raises ``ValueError``). Default to the values ``model``'s transit
        was built with. Scalar or ``(nplanet,)``, same broadcasting
        rule as ``t0``/``period``/etc.
    sp_orb : float or jnp.ndarray, shape (nplanet,), optional
        Sky-projected spin-orbit angle λ [deg]. Same all-or-nothing-with-
        ``ecc``/``omega_peri`` treatment: only used together with a
        transit, defaults to the value ``model``'s transit was built with
        (also in degrees). Converted to radians internally before use --
        ``sajax.planet.planet_sky_position`` itself takes radians. Scalar or ``(nplanet,)``,
        same broadcasting rule as ``t0``/``period``/etc.
    transit_softness : float, optional
        Sigmoid transition width [R*] for the planet occultation mask
        (default 0.0: exact hard edge, matching the physical simulation).
        The hard edge makes occulted flux a staircase function of every
        transit-geometry parameter on the fixed pixel grid, so
        ``jax.grad`` w.r.t. ``k``/``a_over_rstar``/``inclination``/``t0``/
        ``period``/``ecc``/``omega_peri`` is exactly 0 almost everywhere
        regardless of the values passed in above. Set this > 0 (e.g. a few
        tenths of a pixel in R* units) to get a smooth, non-zero gradient
        for gradient-based retrieval of those parameters. See
        ``_compute_planet_mask`` for details and trade-offs.

    Returns
    -------
    (lc, star_maps) tuple
    ~~~~~~~~~~~~~~~~~~~~~
    ``lc``        - (ntimes, nwave) disc-integrated flux at each
                    wavelength bin, in the same units as
                    ``flux_quiet``/``flux_active`` (not normalised to the
                    quiet-star baseline -- divide by that yourself if you
                    want relative flux). If ``nwave == 1``, the wavelength
                    axis is dropped and this is shape (ntimes,).
    ``star_maps`` - (ntimes, n, n) stellar flux map per phase
                    (maps are from the *first* sub-exposure of each phase
                    when oversampling is active)
    """
    # ---- Resolve active-region, LDC, and transit parameter groups --------
    ar = _resolve_ar_params(model, flux_active, ar_lat, ar_long, ar_size, ar_smoothness)
    nar   = ar["nar"]
    nwave = model["nwave"]

    spr             = model["star_pixel_rad"]
    inc_star        = model["inc_star"]
    oversample      = model["oversample"]
    nphase_original = model["nphase_original"]

    ar_time_varying = ar["ar_time_varying"]
    flux_active     = ar["flux_active"]
    ar_lat          = ar["ar_lat"]
    ar_long         = ar["ar_long"]
    ar_size         = ar["ar_size"]
    ar_smoothness   = ar["ar_smoothness"]
    all_ar_carts    = ar["all_ar_carts"]

    ld_coeffs_quiet_val, ld_coeffs_active, I_profile_active = _resolve_ld_coeffs(
        model, ld_coeffs_quiet, ld_coeffs_active, I_profile_active, nar,
    )

    transit = _resolve_transit_params(
        model, t0, period, a_over_rstar, inclination, ecc, omega_peri, sp_orb, k, nwave,
    )
    planet_xyz_all = transit["planet_xyz_all"]
    k_val          = transit["k_val"]

    # ---- All-phases computation ------------------------------------------
    if not ar_time_varying:
        lc_raw, star_maps = _compute_all_phases_lc(
            all_ar_carts,
            planet_xyz_all,
            wavelength          = model["wavelength"],
            flux_quiet          = model["flux_quiet"],
            flux_active         = flux_active,
            ld_coeffs_quiet     = ld_coeffs_quiet_val,
            ld_coeffs_active    = ld_coeffs_active,
            I_profile_quiet     = model["I_profile"],
            I_profile_active    = I_profile_active,
            mu_profile_pts      = model["mu_profile_pts"],
            x_disc              = model["x_disc"],
            y_disc              = model["y_disc"],
            mu_disc             = model["mu_disc"],
            col_idx             = model["col_idx"],
            vel_col             = model["vel_col"],
            star_pixel_rad      = spr,
            total_pixels        = model["total_pixels"],
            arsize_rads         = jnp.deg2rad(ar_size),
            ar_smoothness       = ar_smoothness,
            k                   = k_val,
            ld_mode             = model["ld_mode"],
            plot_map_wavelength = model["plot_map_wavelength"],
            n                   = model["n"],
            flat_indices        = model["flat_indices"],
            transit_softness    = transit_softness,
        )
    else:
        lc_raw, star_maps = _compute_all_phases_lc_evolving(
            ar_lat, ar_long,
            jnp.deg2rad(ar_size),
            ar_smoothness,
            flux_active,
            planet_xyz_all,
            model["phases_rot"],
            inc_star            = inc_star,
            star_pixel_rad      = spr,
            wavelength          = model["wavelength"],
            flux_quiet          = model["flux_quiet"],
            ld_coeffs_quiet     = ld_coeffs_quiet_val,
            ld_coeffs_active    = ld_coeffs_active,
            I_profile_quiet     = model["I_profile"],
            I_profile_active    = I_profile_active,
            mu_profile_pts      = model["mu_profile_pts"],
            x_disc              = model["x_disc"],
            y_disc              = model["y_disc"],
            mu_disc             = model["mu_disc"],
            col_idx             = model["col_idx"],
            vel_col             = model["vel_col"],
            total_pixels        = model["total_pixels"],
            k                   = k_val,
            ld_mode             = model["ld_mode"],
            plot_map_wavelength = model["plot_map_wavelength"],
            n                   = model["n"],
            flat_indices        = model["flat_indices"],
            transit_softness    = transit_softness,
        )

    # ---- Oversample averaging --------------------------------------------
    if oversample > 1:
        # lc_raw: (nphase_compute, nwave) -> (nphase_original, oversample, nwave) -> mean
        lc_raw = lc_raw.reshape(nphase_original, oversample, nwave).mean(axis=1)

        # star_maps: take only the first sub-exposure per original phase
        # (averaging 2D maps is expensive and rarely useful)
        star_maps = star_maps[::oversample]

    # ---- Single-wavelength convenience: drop the now-degenerate nwave
    # axis so single-channel callers get (nphase,) instead of (nphase, 1) ---
    if nwave == 1:
        lc_raw = lc_raw[..., 0]

    return lc_raw, star_maps


def quick_lc(
    wavelength: np.ndarray,
    flux_quiet: np.ndarray,
    flux_active: np.ndarray,
    ar_lat: np.ndarray,
    ar_long: np.ndarray,
    ar_size: np.ndarray,
    ar_smoothness: np.ndarray,
    times: np.ndarray,
    P_rot: float,
    stellar_grid_size: int = 100,
    ve: float = 0.0,
    ld_coeffs: Optional[list] = None,
    inc_star: float = 90.0,
    mu_profile: Optional[np.ndarray] = None,
    I_profile: Optional[np.ndarray] = None,
    ld_mode: LdMode = "quadratic",
    ld_coeffs_active: Optional[np.ndarray] = None,
    I_profile_active: Optional[np.ndarray] = None,
    plot_map_wavelength: Optional[float] = None,
    oversample: int = 1,
    t0: Optional[float | np.ndarray] = None,
    period: Optional[float | np.ndarray] = None,
    a_over_rstar: Optional[float | np.ndarray] = None,
    inclination: Optional[float | np.ndarray] = None,
    k: Optional[float | np.ndarray] = None,
    ecc: float | np.ndarray = 0.0,
    omega_peri: float | np.ndarray = 0.0,
    sp_orb: float | np.ndarray = 0.0,
    ar_time_interp: ArTimeInterp = "linear",
    verbose: bool = False,
) -> tuple:
    """
    Convenience wrapper: build model and evaluate in one call.

    Equivalent to::

        model  = build_system(wavelength, flux_quiet, times, P_rot,
                             stellar_grid_size, ve, ld_coeffs, inc_star,
                             mu_profile, I_profile, ld_mode,
                             plot_map_wavelength, oversample,
                             t0, period, a_over_rstar, inclination, k,
                             ecc, omega_peri, sp_orb)
        lc, star_maps = make_lc(model, flux_active, ar_lat, ar_long,
                                      ar_size, ar_smoothness,
                                      ld_coeffs_active, I_profile_active)

    Use ``build_system`` + ``make_lc`` directly when running
    MCMC so the grid is built only once.

    Parameters
    ----------
    wavelength : array_like, shape (nwave,)
        Wavelength value or array at which the quiet photosphere and active-region spectra are defined.
    flux_quiet : array_like, shape (nwave,)
        Quiet-photosphere flux / spectrum.
    flux_active : jnp.ndarray, shape (nar, nwave), (nwave,), or (ntime, nar, nwave), optional
        Per-active-region flux / spectrum. Must be >= 0.
        - If (nwave,):     broadcasts to all active regions.
        - If (nar, nwave): each active region gets its own spectrum.
        - If (ntime, nar, nwave): each active region gets its own time-varying spectrum.
    ar_lat : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region latitudes in degrees. Must be in [-90, 90].
    ar_long : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region longitudes in degrees. Must be in [0, 360).
    ar_size : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region angular radii in degrees. Must be >= 0.
    ar_smoothness : jnp.ndarray, shape (nar,), scalar, or (ntime, nar), optional
        Super-Gaussian order controlling the sharpness of each AR's
        boundary (see ``_compute_ar_shape``). ``1`` is a true Gaussian;
        larger values sharpen the edge, converging to a hard-edged cap as
        ``ar_smoothness -> inf``. A scalar (or size-1 array) is broadcast
        to all active regions.

        ``flux_active``/``ar_lat``/``ar_long``/``ar_size``/``ar_smoothness``
        are all-or-nothing: give every one of them to add active region(s),
        or omit all five for a quiet star. Giving some but not all raises
        ``ValueError``.

        **Time-varying active regions.** Any of the above five properties may
        independently carry an extra leading axis of length ``ntime`` instead
        of its usual shape, to let that property evolve over the observation,
        e.g. a spot that grows/decays (``ar_size``), drifts in latitude/longitude
        (``ar_lat``/``ar_long``), or changes contrast (``flux_active``).
        Values are given per original time and interpolated internally onto
        ``oversample`` sub-exposures when oversampling is active, using the
        model's ``ar_time_interp`` law (set at ``build_system`` time, like
        ``ld_mode`` -- see there for the "linear"/"cubic" choice). Mixing is
        allowed: only the parameters you want to evolve need the extra axis,
        the rest keep their usual constant-in-time shape. Using none of them
        (the default) is exactly as fast as before this capability existed.
        ``ld_coeffs_active``/``I_profile_active`` do not support time evolution
        and always stay fixed across the observation. Each epoch's values are
        used exactly as given -- nothing couples them across epochs, so it's
        up to the user to keep a fitted active region from changing unphysically
        fast between epochs. If ``nar`` (read off the parameters' own trailing axis)
        happens to equal ``ntime``, make_lc warns: this is the shape signature of
        the common mistake of leaving a parameter meant to evolve as a flat ``(ntime,)``
        array instead of ``(ntime, 1)``, which silently produces ``ntime`` separate
        static active regions instead of one evolving one.
    times : array_like, shape (ntime,)
        Absolute observation times [days].
    P_rot : float
        Stellar rotation period [days].
    stellar_grid_size : int
        Radius of the stellar grid.
    ve : float
        Stellar equatorial rotational velocity [km/s].
    ld_coeffs : list of float or list of array(nwave,), optional
        Quiet-photosphere limb-darkening coefficients for the chosen
        ``ld_mode``:
        - ``"linear"``:     [u]
        - ``"quadratic"``:  [u1, u2]
        - ``"power2"``:     [c, alpha]
        - ``"kipping3"``:   [c1, c2, c3]
        - ``"nonlinear4"``: [c1, c2, c3, c4]
        Each element may be a scalar (broadcast to all wavelengths) or an
        array of length ``nwave``. Active regions carry their own,
        independent coefficients -- see ``make_lc``. Not used
        (and not required) when ``ld_mode="intensity_profile"``; its
        required length for every other mode is checked against
        ``ld_mode`` here.
    inc_star : float, optional
        Stellar inclination in degrees (default: 90.0).
        90deg = equator-on, 0deg = pole-on.
    mu_profile : array-like, optional
        Monotonically increasing mu grid points for
        ``ld_mode="intensity_profile"`` (default: [0, 1]).
    I_profile : array-like, shape (nwave, n_mu_pts), optional
        Quiet-photosphere specific intensity at each (wavelength, mu) grid
        point. Required when ``ld_mode="intensity_profile"``.
    ld_mode : str (default "quadratic")
        Limb-darkening law, shared by the quiet photosphere and every
        active region (each with its own coefficient values).
    ld_coeffs_active : jnp.ndarray, shape (nar, nwave, n_coeffs) or (nwave, n_coeffs), optional
        Per-active-region limb-darkening coefficients, same law as the
        quiet photosphere (``model["ld_mode"]``) but independent values.
        A (nwave, n_coeffs) array broadcasts to all active regions.
        Defaults to the quiet photosphere's own coefficients if omitted.
        Not used when ``ld_mode="intensity_profile"``.
    I_profile_active : jnp.ndarray, shape (nar, nwave, n_mu_pts) or (nwave, n_mu_pts), optional
        Per-active-region specific-intensity profile, used only when
        ``ld_mode="intensity_profile"``. Defaults to the quiet
        photosphere's own profile if omitted.
    plot_map_wavelength : float, optional
        Wavelength at which to plot the stellar map (see ``build_system``).
    oversample : int, optional
        Number of sub-exposures per phase point.  Each requested phase is
        spread into ``oversample`` uniformly spaced sub-phases spanning one
        phase step, and the resulting fluxes are averaged.  This mimics
        finite-exposure integration and smooths limb-crossing artefacts.
        Default: 1 (no oversampling).
    t0, period, a_over_rstar, inclination : float or array-like, shape (nplanet,), optional
        Transit-geometry parameters: mid-transit epoch, orbital period [days],
        semi-major axis/R*, and orbital inclination [rad]. All-or-nothing with
        ``k``. Each is a scalar or an array with a trailing ``(nplanet,)``
        axis -- exactly like ``ar_lat``/``ar_long``/etc. for active regions
        (see ``make_lc``) -- with ``nplanet`` inferred from ``t0``'s trailing
        axis. A scalar (or size-1 array) among the others broadcasts to every
        planet; giving more than one planet requires giving each of these
        four its own length-``nplanet`` array (or leaving it scalar to share
        one value across all planets).
    k : float or array-like, shape (nplanet,) or (nplanet, nwave), optional
        Planet-to-star radius ratio Rp/R*. A scalar (the same value for
        every planet and wavelength), an array of length ``nplanet`` (one
        achromatic value per planet), or an array of shape
        ``(nplanet, nwave)`` (a chromatic transit depth, independent per
        planet). For a single planet (``nplanet == 1``), a bare array of
        length ``nwave`` is also accepted as that planet's chromatic depth
        (legacy convention). Multiple overlapping planets combine their
        occultation multiplicatively (they are opaque, unlike active
        regions' additive contrast -- see ``make_lc``), so overlapping
        transits never drive flux negative.
        The planet's parameters are all-or-nothing, exactly like the AR parameters:
        give every one of them to evaluate a transit at those (possibly
        traced) values instead of the static ones ``model`` was built
        with, or omit all five to fall back to the model's static transit
        (or to no transit, if it doesn't have one) -- ``ValueError`` if
        only some are given.
    ecc, omega_peri : float or array-like, shape (nplanet,), optional
        Orbital eccentricity and argument of periastron [rad]. Only
        meaningful together with a transit; default to 0.0 (circular,
        non-precessing orbit). Scalar or ``(nplanet,)``, same broadcasting
        rule as ``t0``/``period``/etc.
    sp_orb : float or jnp.ndarray, shape (nplanet,), optional
        sky-projected spin-orbit angle, λ  [deg]
        Rotates the transit chord about the stellar
        centre, in the sky plane. Angle is relative to the
        stellar equator. Only meaningful together with a transit.
        Converted to radians internally before use --
        ``sajax.planet.planet_sky_position`` itself takes radians.
        Scalar or ``(nplanet,)``, same broadcasting rule as ``t0``/``period``/etc.
    ar_time_interp : "linear" or "cubic", optional (default "linear")
        Interpolation method for when time-varying active-region parameters
        need to be interpolated onto ``oversample``'d sub-exposures.
        Only matters when ``oversample > 1`` and at least one AR parameter
        is time-varying -- see ``make_lc`` for the full description.
    verbose : bool, optional
        If True, print informational messages (LDC broadcasting, phase
        oversampling) while building the model. Default False.

    Returns
    -------
    (lc, star_maps) tuple
    ~~~~~~~~~~~~~~~~~~~~~
    ``lc``        - (ntimes, nwave) disc-integrated flux at each
                    wavelength bin, in the same units as
                    ``flux_quiet``/``flux_active`` (not normalised to the
                    quiet-star baseline -- divide by that yourself if you
                    want relative flux). If ``nwave == 1``, the wavelength
                    axis is dropped and this is shape (ntimes,).
    ``star_maps`` - (ntimes, n, n) stellar flux map per phase
                    (maps are from the *first* sub-exposure of each phase
                    when oversampling is active)
    """
    model  = build_system(
        wavelength, flux_quiet, times=times, P_rot=P_rot,
        stellar_grid_size=stellar_grid_size, ve=ve,
        ld_coeffs=ld_coeffs, inc_star=inc_star, mu_profile=mu_profile,
        I_profile=I_profile, ld_mode=ld_mode,
        plot_map_wavelength=plot_map_wavelength, oversample=oversample,
        ar_time_interp=ar_time_interp,
        t0=t0, period=period, a_over_rstar=a_over_rstar,
        inclination=inclination, k=k, ecc=ecc, omega_peri=omega_peri,
        sp_orb=sp_orb,
        verbose=verbose,
    )

    flux_active_arr = np.atleast_1d(np.asarray(flux_active, dtype=np.float32))
    lc, star_maps = make_lc(
        model,
        jnp.asarray(flux_active_arr),
        jnp.asarray(np.atleast_1d(np.asarray(ar_lat,  dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_long, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_size, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_smoothness, dtype=np.float32))),
        None if ld_coeffs_active is None else jnp.asarray(np.asarray(ld_coeffs_active, dtype=np.float32)),
        None if I_profile_active  is None else jnp.asarray(np.asarray(I_profile_active,  dtype=np.float32)),
    )
    return np.array(lc), np.array(star_maps)


# ---------------------------------------------------------------------------
# 8. Public API -- radial velocity
#
#   Stage 1:  build_system()         - NumPy, call once before sampling.
#                                     Pre-builds the grid and all static
#                                     arrays that are fixed across MCMC steps.
#
#   Stage 2:  make_rv() - Pure JAX, call at every MCMC step.
#                                      Accepts JAX arrays / tracers so it is
#                                      fully compatible with jit, vmap, and
#                                      gradient-based samplers.
#
#   quick_rv()            - Convenience wrapper that calls both
#                                      stages in sequence.  Useful for
#                                      one-off calls outside MCMC.
# ---------------------------------------------------------------------------

def make_rv(
    model: dict,
    flux_active: Optional[jnp.ndarray] = None,
    ar_lat: Optional[jnp.ndarray] = None,
    ar_long: Optional[jnp.ndarray] = None,
    ar_size: Optional[jnp.ndarray] = None,
    ar_smoothness: Optional[jnp.ndarray] = None,
    ld_coeffs_active: Optional[jnp.ndarray] = None,
    I_profile_active: Optional[jnp.ndarray] = None,
    ld_coeffs_quiet: Optional[jnp.ndarray] = None,
    t0: Optional[float | jnp.ndarray] = None,
    period: Optional[float | jnp.ndarray] = None,
    a_over_rstar: Optional[float | jnp.ndarray] = None,
    inclination: Optional[float | jnp.ndarray] = None,
    ecc: Optional[float | jnp.ndarray] = None,
    omega_peri: Optional[float | jnp.ndarray] = None,
    sp_orb: Optional[float | jnp.ndarray] = None,
    k: Optional[float | jnp.ndarray] = None,
    planet_mass: Optional[float | jnp.ndarray] = None,
    stellar_mass: Optional[float] = None,
    gamma: float = 0.0,
    transit_softness: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate the radial-velocity curve for a given set of active region and
    planetary parameters.

    This function is **pure JAX** -- all inputs may be JAX arrays or tracers,
    making it fully compatible with ``jit``, ``vmap``, and gradient-based
    samplers such as ``emcee_jax`` or ``blackjax``.

    When the model was built with ``oversample > 1``, the computation runs
    on the oversampled phase grid and the results are averaged back to the
    original phase grid before returning.
    
    Accepts the **identical** active-region and transit parameter set as
    ``make_lc`` (``flux_active``/``ar_lat``/``ar_long``/``ar_size``/
    ``ar_smoothness``, ``ld_coeffs_active``/``I_profile_active``/
    ``ld_coeffs_quiet``, ``t0``/``period``/``a_over_rstar``/``inclination``/
    ``ecc``/``omega_peri``/``sp_orb``/``k``) -- same all-or-nothing rules,
    same time-varying-AR support, same static/dynamic/dummy transit
    dispatch (via the shared ``_resolve_ar_params``/``_resolve_ld_coeffs``/
    ``_resolve_transit_params`` helpers ``make_lc`` also uses). The only 
    additional ``make_rv`` requires are the ``planet_mass``, ``stellar_mass``,
    and ``gamma`` parameters.

    
    Parameters
    ----------
    model : dict
        Pre-built model dict returned by ``build_system``.
    flux_active : jnp.ndarray, shape (nar, nwave), (nwave,), or (ntime, nar, nwave), optional
        Per-active-region flux / spectrum. Must be >= 0.
        - If (nwave,):     broadcasts to all active regions.
        - If (nar, nwave): each active region gets its own spectrum.
        - If (ntime, nar, nwave): each active region gets its own time-varying spectrum.
    ar_lat : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region latitudes in degrees. Must be in [-90, 90].
    ar_long : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region longitudes in degrees. Must be in [0, 360).
    ar_size : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region angular radii in degrees. Must be >= 0.
    ar_smoothness : jnp.ndarray, shape (nar,), scalar, or (ntime, nar), optional
        Super-Gaussian order controlling the sharpness of each AR's
        boundary (see ``_compute_ar_shape``). ``1`` is a true Gaussian;
        larger values sharpen the edge, converging to a hard-edged cap as
        ``ar_smoothness -> inf``. A scalar (or size-1 array) is broadcast
        to all active regions.

        ``flux_active``/``ar_lat``/``ar_long``/``ar_size``/``ar_smoothness``
        are all-or-nothing: give every one of them to add active region(s),
        or omit all five for a quiet star. Giving some but not all raises
        ``ValueError``.

        **Time-varying active regions.** Any of the above five properties may
        independently carry an extra leading axis of length ``ntime`` instead
        of its usual shape, to let that property evolve over the observation,
        e.g. a spot that grows/decays (``ar_size``), drifts in latitude/longitude
        (``ar_lat``/``ar_long``), or changes contrast (``flux_active``).
        Values are given per original time and interpolated internally onto
        ``oversample`` sub-exposures when oversampling is active, using the
        model's ``ar_time_interp`` law (set at ``build_system`` time, like
        ``ld_mode`` -- see there for the "linear"/"cubic" choice). Mixing is
        allowed: only the parameters you want to evolve need the extra axis,
        the rest keep their usual constant-in-time shape. Using none of them
        (the default) is exactly as fast as before this capability existed.
        ``ld_coeffs_active``/``I_profile_active`` do not support time evolution
        and always stay fixed across the observation. Each epoch's values are
        used exactly as given -- nothing couples them across epochs, so it's
        up to the user to keep a fitted active region from changing unphysically
        fast between epochs. If ``nar`` (read off the parameters' own trailing axis)
        happens to equal ``ntime``, make_lc warns: this is the shape signature of
        the common mistake of leaving a parameter meant to evolve as a flat ``(ntime,)``
        array instead of ``(ntime, 1)``, which silently produces ``ntime`` separate
        static active regions instead of one evolving one.
    ld_coeffs_active : jnp.ndarray, shape (nar, nwave, n_coeffs) or (nwave, n_coeffs), optional
        Per-active-region limb-darkening coefficients, same law as the
        quiet photosphere (``model["ld_mode"]``) but independent values.
        A (nwave, n_coeffs) array broadcasts to all active regions.
        Defaults to the quiet photosphere's own coefficients if omitted.
        Not used when ``ld_mode="intensity_profile"``.
    I_profile_active : jnp.ndarray, shape (nar, nwave, n_mu_pts) or (nwave, n_mu_pts), optional
        Per-active-region specific-intensity profile, used only when
        ``ld_mode="intensity_profile"``. Defaults to the quiet
        photosphere's own profile if omitted.
    ld_coeffs_quiet : jnp.ndarray, shape (nwave, n_coeffs), optional
        Dynamic override for the quiet photosphere's own limb-darkening
        coefficients (JAX values/tracers are fine) -- the build-time
        ``ld_coeffs`` given to ``build_system`` is
        otherwise fixed for the life of the model, exactly like the
        transit-geometry parameters were before they got this same
        treatment. Defaults to the static value ``model`` was built with.
        When given, active regions that don't specify their own
        ``ld_coeffs_active`` default to this (possibly traced) value too,
        instead of the static one.
    t0, period, a_over_rstar, inclination : float or jnp.ndarray, shape (nplanet,), optional
        Transit-geometry parameters: mid-transit epoch, orbital period [days],
        semi-major axis/R*, and orbital inclination [rad]. Scalar or
        ``(nplanet,)``, same trailing-axis convention as ``build_system``
        (``nplanet`` inferred from ``t0``). When overriding a model built
        with a different ``nplanet``, also override ``ecc``/``omega_peri``/
        ``sp_orb`` if you don't want them to fall back to the build-time defaults.
    k : float or array-like, shape (nplanet,) or (nplanet, nwave), optional
        Planet-to-star radius ratio Rp/R*. A scalar (the same value for
        every planet and wavelength), an array of length ``nplanet`` (one
        achromatic value per planet), or an array of shape
        ``(nplanet, nwave)`` (a chromatic transit depth, independent per
        planet). For a single planet (``nplanet == 1``), a bare array of
        length ``nwave`` is also accepted as that planet's chromatic depth
        (legacy convention). Multiple overlapping planets combine their
        occultation multiplicatively (they are opaque, unlike active
        regions' additive contrast -- see ``make_lc``), so overlapping
        transits never drive flux negative.
        The planet's parameters are all-or-nothing, exactly like the AR parameters:
        give every one of them to evaluate a transit at those (possibly
        traced) values instead of the static ones ``model`` was built
        with, or omit all five to fall back to the model's static transit
        (or to no transit, if it doesn't have one) -- ``ValueError`` if
        only some are given.
    ecc, omega_peri : float or jnp.ndarray, shape (nplanet,), optional
        Orbital eccentricity and argument of periastron [rad]. Only
        meaningful together with a transit; default to 0.0 (circular,
        non-precessing orbit). Only used together with the five required
        transit parameters above (giving either of these without the rest
        also raises ``ValueError``). Default to the values ``model``'s transit
        was built with. Scalar or ``(nplanet,)``, same broadcasting
        rule as ``t0``/``period``/etc.
    sp_orb : float or jnp.ndarray, shape (nplanet,), optional
        Sky-projected spin-orbit angle λ [deg]. Same all-or-nothing-with-
        ``ecc``/``omega_peri`` treatment: only used together with a
        transit, defaults to the value ``model``'s transit was built with
        (also in degrees). Converted to radians internally before use --
        ``sajax.planet.planet_sky_position`` itself takes radians. Scalar or ``(nplanet,)``,
        same broadcasting rule as ``t0``/``period``/etc.
    planet_mass : float or jnp.ndarray, shape (nplanet,), optional
        Planet mass(es) [M_Jup], >= 0. Required together with
        ``stellar_mass`` whenever a transit/orbit is attached (via
        ``build_system`` or this call's own ``t0``/``period``/
        ``a_over_rstar``/``inclination``/``k``). A scalar broadcasts
        to every planet; pass an explicit ``(nplanet,)``
        array for independent masses.
    stellar_mass : float, optional
        Stellar mass [M_sun], > 0. Required together with ``planet_mass``.
    gamma : float, optional (default 0.0)
        Systemic velocity offset [km/s], added to the total RV.
    transit_softness : float, optional
        Sigmoid transition width [R*] for the planet occultation mask
        (default 0.0: exact hard edge, matching the physical simulation).
        The hard edge makes occulted flux a staircase function of every
        transit-geometry parameter on the fixed pixel grid, so
        ``jax.grad`` w.r.t. ``k``/``a_over_rstar``/``inclination``/``t0``/
        ``period``/``ecc``/``omega_peri`` is exactly 0 almost everywhere
        regardless of the values passed in above. Set this > 0 (e.g. a few
        tenths of a pixel in R* units) to get a smooth, non-zero gradient
        for gradient-based retrieval of those parameters. See
        ``_compute_planet_mask`` for details and trade-offs.

    Returns
    -------
    (rv, star_maps) tuple
    ~~~~~~~~~~~~~~~~~~~~~
    ``rv``        - (ntimes, nwave) radial velocity [km/s] at each
                    wavelength bin: ``gamma`` + the Keplerian reflex term
                    + the activity/Rossiter-McLaughlin term. If ``nwave == 1``,
                    the wavelength axis is dropped and this is shape (ntimes,).
    ``star_maps`` - (ntimes, n, n) per-pixel radial-velocity map [km/s] per
                    phase (maps are from the *first* sub-exposure of each
                    phase when oversampling is active). Unlike
                    ``make_lc``'s ``star_maps``, this is a velocity map,
                    not a flux map: each pixel shows its own line-of-sight
                    velocity weighted by its brightness relative to the
                    map's brightest pixel, so a dimmed active-region pixel
                    fades toward zero and a fully-occulted (transiting)
                    pixel is exactly zero.
    """
    ar = _resolve_ar_params(model, flux_active, ar_lat, ar_long, ar_size, ar_smoothness)
    nar   = ar["nar"]
    nwave = model["nwave"]

    oversample      = model["oversample"]
    nphase_original = model["nphase_original"]
    nphase_compute  = model["phases_rot"].shape[0]

    ar_time_varying = ar["ar_time_varying"]
    flux_active_r   = ar["flux_active"]
    ar_lat_r        = ar["ar_lat"]
    ar_long_r       = ar["ar_long"]
    ar_size_r       = ar["ar_size"]
    ar_smoothness_r = ar["ar_smoothness"]
    all_ar_carts    = ar["all_ar_carts"]

    ld_coeffs_quiet_val, ld_coeffs_active_r, I_profile_active_r = _resolve_ld_coeffs(
        model, ld_coeffs_quiet, ld_coeffs_active, I_profile_active, nar,
    )

    transit = _resolve_transit_params(
        model, t0, period, a_over_rstar, inclination, ecc, omega_peri, sp_orb, k, nwave,
    )
    planet_xyz_all = transit["planet_xyz_all"]
    k_val          = transit["k_val"]

    # ---- Keplerian term ---------------------------------------------------
    _mass_args = dict(planet_mass=planet_mass, stellar_mass=stellar_mass)
    _mass_given = {name: v for name, v in _mass_args.items() if v is not None}
    if _mass_given and len(_mass_given) != len(_mass_args):
        _missing = [name for name in _mass_args if name not in _mass_given]
        raise ValueError(
            "make_rv: partial mass parameters given "
            f"({sorted(_mass_given)}); missing {_missing}. Provide both "
            "planet_mass and stellar_mass to compute the Keplerian RV "
            "term, or neither for a pure activity/RM RV curve."
        )
    if _mass_given and not transit["has_transit"]:
        raise ValueError(
            "make_rv: planet_mass/stellar_mass were given, but no "
            "transit/orbit is attached -- provide t0/period/a_over_rstar/"
            "inclination/k (via build_system or this call) to attach one, "
            "or omit planet_mass/stellar_mass for a pure activity/RM RV curve."
        )
    if transit["has_transit"] and not _mass_given:
        raise ValueError(
            "make_rv: this model has a transit/orbit attached, but "
            "planet_mass and stellar_mass were not given -- both are "
            "required to compute the Keplerian RV amplitude for an "
            "attached transit (there is no physically sensible default "
            "for a system known to host a planet). Omit the transit's "
            "parameters entirely for a pure activity/RM RV curve."
        )

    if _mass_given:
        # Check the *computed* boolean result for tracer-ness, not
        # planet_mass itself -- see the identical comment in _resolve_ar_params.
        _planet_mass_check = jnp.any(jnp.asarray(planet_mass) < 0)
        if not isinstance(_planet_mass_check, jax.core.Tracer) and bool(_planet_mass_check):
            raise ValueError(
                f"planet_mass must be >= 0 (got {planet_mass}); planet_mass "
                "is a mass and cannot be negative."
            )
        if not isinstance(stellar_mass, jax.core.Tracer) and float(stellar_mass) <= 0:
            raise ValueError(
                f"stellar_mass must be > 0 (got {stellar_mass})."
            )
        # Broadcast/validate planet_mass against nplanet up front (same
        # convention and error semantics as every other per-planet
        # parameter -- see _broadcast_orbital_param), so a length mismatch
        # raises a clear ValueError here rather than an opaque shape error
        # from keplerian_rv_semi_amplitude's raw elementwise arithmetic.
        planet_mass_arr = _broadcast_orbital_param(
            "planet_mass", planet_mass, transit["nplanet"], "make_rv"
        )
        K_arr = keplerian_rv_semi_amplitude(
            planet_mass_arr, stellar_mass, transit["period_val"],
            transit["ecc_val"], transit["inclination_val"],
        )
        rv_kep_raw = compute_multi_planet_keplerian_rv(
            model["times_oversampled"], transit["t0_val"], transit["period_val"],
            transit["ecc_val"], transit["omega_peri_val"], K_arr,
        )   # (nphase_compute,)
    else:
        rv_kep_raw = jnp.zeros(nphase_compute)

    # ---- Activity + Rossiter-McLaughlin term -------------------------------
    if not ar_time_varying:
        rv_arm_raw, star_maps_raw = _compute_all_phases_rv(
            all_ar_carts,
            planet_xyz_all,
            wavelength          = model["wavelength"],
            flux_quiet          = model["flux_quiet"],
            flux_active         = flux_active_r,
            ld_coeffs_quiet     = ld_coeffs_quiet_val,
            ld_coeffs_active    = ld_coeffs_active_r,
            I_profile_quiet     = model["I_profile"],
            I_profile_active    = I_profile_active_r,
            mu_profile_pts      = model["mu_profile_pts"],
            x_disc              = model["x_disc"],
            y_disc              = model["y_disc"],
            mu_disc             = model["mu_disc"],
            col_idx             = model["col_idx"],
            vel_col             = model["vel_col"],
            star_pixel_rad      = model["star_pixel_rad"],
            total_pixels        = model["total_pixels"],
            arsize_rads         = jnp.deg2rad(ar_size_r),
            ar_smoothness       = ar_smoothness_r,
            k                   = k_val,
            ld_mode             = model["ld_mode"],
            plot_map_wavelength = model["plot_map_wavelength"],
            n                   = model["n"],
            flat_indices        = model["flat_indices"],
            transit_softness    = transit_softness,
        )
    else:
        rv_arm_raw, star_maps_raw = _compute_all_phases_rv_evolving(
            ar_lat_r, ar_long_r,
            jnp.deg2rad(ar_size_r),
            ar_smoothness_r,
            flux_active_r,
            planet_xyz_all,
            model["phases_rot"],
            inc_star            = model["inc_star"],
            star_pixel_rad      = model["star_pixel_rad"],
            wavelength          = model["wavelength"],
            flux_quiet          = model["flux_quiet"],
            ld_coeffs_quiet     = ld_coeffs_quiet_val,
            ld_coeffs_active    = ld_coeffs_active_r,
            I_profile_quiet     = model["I_profile"],
            I_profile_active    = I_profile_active_r,
            mu_profile_pts      = model["mu_profile_pts"],
            x_disc              = model["x_disc"],
            y_disc              = model["y_disc"],
            mu_disc             = model["mu_disc"],
            col_idx             = model["col_idx"],
            vel_col             = model["vel_col"],
            total_pixels        = model["total_pixels"],
            k                   = k_val,
            ld_mode             = model["ld_mode"],
            plot_map_wavelength = model["plot_map_wavelength"],
            n                   = model["n"],
            flat_indices        = model["flat_indices"],
            transit_softness    = transit_softness,
        )

    # ---- Oversample averaging ---------------------------------------------
    # Matters more here than for the light curve: the RM velocity anomaly
    # can change sharply near ingress/egress (the occulted chord crossing
    # high-|vel_col| limb regions), more so than the smooth light-curve
    # dip. Both RV terms are averaged uniformly -- the Keplerian term is
    # smooth on exposure timescales, so this is a cheap no-op for it.
    # rv_arm_raw carries a trailing (nwave,) axis (see
    # _compute_single_phase_rv); rv_kep_raw does not (the Keplerian term
    # is wavelength-independent), so it reshapes/averages over phase alone
    # exactly like make_lc's own lc_raw does for its (nphase, nwave) shape.
    # star_maps takes the first sub-exposure per original phase, exactly
    # like make_lc's own star_maps (averaging 2D maps is expensive and
    # rarely useful).
    if oversample > 1:
        rv_kep_avg = rv_kep_raw.reshape(nphase_original, oversample).mean(axis=1)
        rv_arm_avg = rv_arm_raw.reshape(nphase_original, oversample, nwave).mean(axis=1)
        star_maps  = star_maps_raw[::oversample]
    else:
        rv_kep_avg, rv_arm_avg = rv_kep_raw, rv_arm_raw
        star_maps = star_maps_raw

    # rv_kep_avg (nphase,) broadcasts against rv_arm_avg (nphase, nwave) --
    # the same Keplerian value applies at every wavelength bin.
    rv = gamma + rv_kep_avg[:, None] + rv_arm_avg   # (nphase, nwave)

    # ---- Single-wavelength convenience: drop the now-degenerate nwave
    # axis, exactly like make_lc's own lc output. ----
    if nwave == 1:
        rv = rv[..., 0]

    return rv, star_maps

def quick_rv(
    wavelength: np.ndarray,
    flux_quiet: np.ndarray,
    flux_active: np.ndarray,
    ar_lat: np.ndarray,
    ar_long: np.ndarray,
    ar_size: np.ndarray,
    ar_smoothness: np.ndarray,
    times: np.ndarray,
    P_rot: float,
    stellar_grid_size: int = 100,
    ve: float = 0.0,
    ld_coeffs: Optional[list] = None,
    inc_star: float = 90.0,
    mu_profile: Optional[np.ndarray] = None,
    I_profile: Optional[np.ndarray] = None,
    ld_mode: LdMode = "quadratic",
    ld_coeffs_active: Optional[np.ndarray] = None,
    I_profile_active: Optional[np.ndarray] = None,
    plot_map_wavelength: Optional[float] = None,
    oversample: int = 1,
    t0: Optional[float | np.ndarray] = None,
    period: Optional[float | np.ndarray] = None,
    a_over_rstar: Optional[float | np.ndarray] = None,
    inclination: Optional[float | np.ndarray] = None,
    k: Optional[float | np.ndarray] = None,
    ecc: float | np.ndarray = 0.0,
    omega_peri: float | np.ndarray = 0.0,
    sp_orb: float | np.ndarray = 0.0,
    ar_time_interp: ArTimeInterp = "linear",
    planet_mass: Optional[float | np.ndarray] = None,
    stellar_mass: Optional[float] = None,
    gamma: float = 0.0,
    transit_softness: float = 0.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: build model and evaluate in one call.

    Equivalent to::

        model  = build_system(wavelength, flux_quiet, times, P_rot,
                                     stellar_grid_size, ve, ld_coeffs, inc_star,
                                     mu_profile, I_profile, ld_mode,
                                     plot_map_wavelength, oversample,
                                     t0, period, a_over_rstar, inclination, k,
                                     ecc, omega_peri, sp_orb)
        rv, star_maps = make_rv(model, flux_active, ar_lat, ar_long, ar_size,
                                 ar_smoothness, ..., planet_mass=planet_mass,
                                 stellar_mass=stellar_mass, gamma=gamma)

    Use ``build_system`` + ``make_rv`` directly when running MCMC so the
    grid is built only once.

    Parameters
    ----------
    wavelength : array_like, shape (nwave,)
        Wavelength value or array at which the quiet photosphere and active-region spectra are defined.
    flux_quiet : array_like, shape (nwave,)
        Quiet-photosphere flux / spectrum.
    flux_active : jnp.ndarray, shape (nar, nwave), (nwave,), or (ntime, nar, nwave), optional
        Per-active-region flux / spectrum. Must be >= 0.
        - If (nwave,):     broadcasts to all active regions.
        - If (nar, nwave): each active region gets its own spectrum.
        - If (ntime, nar, nwave): each active region gets its own time-varying spectrum.
    ar_lat : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region latitudes in degrees. Must be in [-90, 90].
    ar_long : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region longitudes in degrees. Must be in [0, 360).
    ar_size : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region angular radii in degrees. Must be >= 0.
    ar_smoothness : jnp.ndarray, shape (nar,), scalar, or (ntime, nar), optional
        Super-Gaussian order controlling the sharpness of each AR's
        boundary (see ``_compute_ar_shape``). ``1`` is a true Gaussian;
        larger values sharpen the edge, converging to a hard-edged cap as
        ``ar_smoothness -> inf``. A scalar (or size-1 array) is broadcast
        to all active regions.

        ``flux_active``/``ar_lat``/``ar_long``/``ar_size``/``ar_smoothness``
        are all-or-nothing: give every one of them to add active region(s),
        or omit all five for a quiet star. Giving some but not all raises
        ``ValueError``.

        **Time-varying active regions.** Any of the above five properties may
        independently carry an extra leading axis of length ``ntime`` instead
        of its usual shape, to let that property evolve over the observation,
        e.g. a spot that grows/decays (``ar_size``), drifts in latitude/longitude
        (``ar_lat``/``ar_long``), or changes contrast (``flux_active``).
        Values are given per original time and interpolated internally onto
        ``oversample`` sub-exposures when oversampling is active, using the
        model's ``ar_time_interp`` law (set at ``build_system`` time, like
        ``ld_mode`` -- see there for the "linear"/"cubic" choice). Mixing is
        allowed: only the parameters you want to evolve need the extra axis,
        the rest keep their usual constant-in-time shape. Using none of them
        (the default) is exactly as fast as before this capability existed.
        ``ld_coeffs_active``/``I_profile_active`` do not support time evolution
        and always stay fixed across the observation. Each epoch's values are
        used exactly as given -- nothing couples them across epochs, so it's
        up to the user to keep a fitted active region from changing unphysically
        fast between epochs. If ``nar`` (read off the parameters' own trailing axis)
        happens to equal ``ntime``, make_lc warns: this is the shape signature of
        the common mistake of leaving a parameter meant to evolve as a flat ``(ntime,)``
        array instead of ``(ntime, 1)``, which silently produces ``ntime`` separate
        static active regions instead of one evolving one.
    times : array_like, shape (ntime,)
        Absolute observation times [days].
    P_rot : float
        Stellar rotation period [days].
    stellar_grid_size : int
        Radius of the stellar grid.
    ve : float
        Stellar equatorial rotational velocity [km/s].
    ld_coeffs : list of float or list of array(nwave,), optional
        Quiet-photosphere limb-darkening coefficients for the chosen
        ``ld_mode``:
        - ``"linear"``:     [u]
        - ``"quadratic"``:  [u1, u2]
        - ``"power2"``:     [c, alpha]
        - ``"kipping3"``:   [c1, c2, c3]
        - ``"nonlinear4"``: [c1, c2, c3, c4]
        Each element may be a scalar (broadcast to all wavelengths) or an
        array of length ``nwave``. Active regions carry their own,
        independent coefficients -- see ``make_lc``. Not used
        (and not required) when ``ld_mode="intensity_profile"``; its
        required length for every other mode is checked against
        ``ld_mode`` here.
    inc_star : float, optional
        Stellar inclination in degrees (default: 90.0).
        90deg = equator-on, 0deg = pole-on.
    mu_profile : array-like, optional
        Monotonically increasing mu grid points for
        ``ld_mode="intensity_profile"`` (default: [0, 1]).
    I_profile : array-like, shape (nwave, n_mu_pts), optional
        Quiet-photosphere specific intensity at each (wavelength, mu) grid
        point. Required when ``ld_mode="intensity_profile"``.
    ld_mode : str (default "quadratic")
        Limb-darkening law, shared by the quiet photosphere and every
        active region (each with its own coefficient values).
    ld_coeffs_active : jnp.ndarray, shape (nar, nwave, n_coeffs) or (nwave, n_coeffs), optional
        Per-active-region limb-darkening coefficients, same law as the
        quiet photosphere (``model["ld_mode"]``) but independent values.
        A (nwave, n_coeffs) array broadcasts to all active regions.
        Defaults to the quiet photosphere's own coefficients if omitted.
        Not used when ``ld_mode="intensity_profile"``.
    I_profile_active : jnp.ndarray, shape (nar, nwave, n_mu_pts) or (nwave, n_mu_pts), optional
        Per-active-region specific-intensity profile, used only when
        ``ld_mode="intensity_profile"``. Defaults to the quiet
        photosphere's own profile if omitted.
    plot_map_wavelength : float, optional
        Wavelength at which to plot the stellar map (see ``build_system``).
    oversample : int, optional
        Number of sub-exposures per phase point.  Each requested phase is
        spread into ``oversample`` uniformly spaced sub-phases spanning one
        phase step, and the resulting fluxes are averaged.  This mimics
        finite-exposure integration and smooths limb-crossing artefacts.
        Default: 1 (no oversampling).
    t0, period, a_over_rstar, inclination : float or array-like, shape (nplanet,), optional
        Transit-geometry parameters: mid-transit epoch, orbital period [days],
        semi-major axis/R*, and orbital inclination [rad]. All-or-nothing with
        ``k``. Each is a scalar or an array with a trailing ``(nplanet,)``
        axis -- exactly like ``ar_lat``/``ar_long``/etc. for active regions
        (see ``make_lc``) -- with ``nplanet`` inferred from ``t0``'s trailing
        axis. A scalar (or size-1 array) among the others broadcasts to every
        planet; giving more than one planet requires giving each of these
        four its own length-``nplanet`` array (or leaving it scalar to share
        one value across all planets).
    k : float or array-like, shape (nplanet,) or (nplanet, nwave), optional
        Planet-to-star radius ratio Rp/R*. A scalar (the same value for
        every planet and wavelength), an array of length ``nplanet`` (one
        achromatic value per planet), or an array of shape
        ``(nplanet, nwave)`` (a chromatic transit depth, independent per
        planet). For a single planet (``nplanet == 1``), a bare array of
        length ``nwave`` is also accepted as that planet's chromatic depth
        (legacy convention). Multiple overlapping planets combine their
        occultation multiplicatively (they are opaque, unlike active
        regions' additive contrast -- see ``make_lc``), so overlapping
        transits never drive flux negative.
        The planet's parameters are all-or-nothing, exactly like the AR parameters:
        give every one of them to evaluate a transit at those (possibly
        traced) values instead of the static ones ``model`` was built
        with, or omit all five to fall back to the model's static transit
        (or to no transit, if it doesn't have one) -- ``ValueError`` if
        only some are given.
    ecc, omega_peri : float or array-like, shape (nplanet,), optional
        Orbital eccentricity and argument of periastron [rad]. Only
        meaningful together with a transit; default to 0.0 (circular,
        non-precessing orbit). Scalar or ``(nplanet,)``, same broadcasting
        rule as ``t0``/``period``/etc.
    sp_orb : float or jnp.ndarray, shape (nplanet,), optional
        sky-projected spin-orbit angle, λ  [deg]
        Rotates the transit chord about the stellar
        centre, in the sky plane. Angle is relative to the
        stellar equator. Only meaningful together with a transit.
        Converted to radians internally before use --
        ``sajax.planet.planet_sky_position`` itself takes radians.
        Scalar or ``(nplanet,)``, same broadcasting rule as ``t0``/``period``/etc.
    ar_time_interp : "linear" or "cubic", optional (default "linear")
        Interpolation method for when time-varying active-region parameters
        need to be interpolated onto ``oversample``'d sub-exposures.
        Only matters when ``oversample > 1`` and at least one AR parameter
        is time-varying -- see ``make_lc`` for the full description.
    planet_mass : float or jnp.ndarray, shape (nplanet,), optional
        Planet mass(es) [M_Jup], >= 0. Required together with
        ``stellar_mass`` whenever a transit/orbit is attached (via
        ``build_system`` or this call's own ``t0``/``period``/
        ``a_over_rstar``/``inclination``/``k``). A scalar broadcasts
        to every planet; pass an explicit ``(nplanet,)``
        array for independent masses.
    stellar_mass : float, optional
        Stellar mass [M_sun], > 0. Required together with ``planet_mass``.
    gamma : float, optional (default 0.0)
        Systemic velocity offset [km/s], added to the total RV.
    verbose : bool, optional
        If True, print informational messages (LDC broadcasting, phase
        oversampling) while building the model. Default False.

    Returns
    -------
    (rv, star_maps) tuple
    ~~~~~~~~~~~~~~~~~~~~~
    ``rv``        - (ntimes, nwave) radial velocity [km/s] at each
                    wavelength bin: ``gamma`` + the Keplerian reflex term
                    + the activity/Rossiter-McLaughlin term. If ``nwave == 1``,
                    the wavelength axis is dropped and this is shape (ntimes,).
    ``star_maps`` - (ntimes, n, n) per-pixel radial-velocity map [km/s] per
                    phase (maps are from the *first* sub-exposure of each
                    phase when oversampling is active). Unlike
                    ``quick_lc``'s ``star_maps``, this is a velocity map,
                    not a flux map -- see ``make_rv``'s docstring.
    """
    model = build_system(
        wavelength, flux_quiet, times=times, P_rot=P_rot,
        stellar_grid_size=stellar_grid_size, ve=ve,
        ld_coeffs=ld_coeffs, inc_star=inc_star, mu_profile=mu_profile,
        I_profile=I_profile, ld_mode=ld_mode,
        plot_map_wavelength=plot_map_wavelength,
        oversample=oversample, ar_time_interp=ar_time_interp,
        t0=t0, period=period, a_over_rstar=a_over_rstar,
        inclination=inclination, k=k, ecc=ecc, omega_peri=omega_peri,
        sp_orb=sp_orb,
        verbose=verbose,
    )

    flux_active_arr = np.atleast_1d(np.asarray(flux_active, dtype=np.float32))
    rv, star_maps = make_rv(
        model,
        jnp.asarray(flux_active_arr),
        jnp.asarray(np.atleast_1d(np.asarray(ar_lat,  dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_long, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_size, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_smoothness, dtype=np.float32))),
        None if ld_coeffs_active is None else jnp.asarray(np.asarray(ld_coeffs_active, dtype=np.float32)),
        None if I_profile_active  is None else jnp.asarray(np.asarray(I_profile_active,  dtype=np.float32)),
        planet_mass=planet_mass, stellar_mass=stellar_mass, gamma=gamma,
        transit_softness=transit_softness,
    )
    return np.array(rv), np.array(star_maps)


# ---------------------------------------------------------------------------
# 9. Public API -- light curve + radial velocity
#
#   Stage 1:  build_system()         - NumPy, call once before sampling.
#                                     Pre-builds the grid and all static
#                                     arrays that are fixed across MCMC steps.
#
#   Stage 2:  make_lc_and_rv() - Pure JAX, call at every MCMC step.
#                                      Accepts JAX arrays / tracers so it is
#                                      fully compatible with jit, vmap, and
#                                      gradient-based samplers.
#
#   quick_lc_and_rv()            - Convenience wrapper that calls both
#                                      stages in sequence.  Useful for
#                                      one-off calls outside MCMC.
# ---------------------------------------------------------------------------

def make_lc_and_rv(
    model: dict,
    flux_active: Optional[jnp.ndarray] = None,
    ar_lat: Optional[jnp.ndarray] = None,
    ar_long: Optional[jnp.ndarray] = None,
    ar_size: Optional[jnp.ndarray] = None,
    ar_smoothness: Optional[jnp.ndarray] = None,
    ld_coeffs_active: Optional[jnp.ndarray] = None,
    I_profile_active: Optional[jnp.ndarray] = None,
    ld_coeffs_quiet: Optional[jnp.ndarray] = None,
    t0: Optional[float | jnp.ndarray] = None,
    period: Optional[float | jnp.ndarray] = None,
    a_over_rstar: Optional[float | jnp.ndarray] = None,
    inclination: Optional[float | jnp.ndarray] = None,
    ecc: Optional[float | jnp.ndarray] = None,
    omega_peri: Optional[float | jnp.ndarray] = None,
    sp_orb: Optional[float | jnp.ndarray] = None,
    k: Optional[float | jnp.ndarray] = None,
    planet_mass: Optional[float | jnp.ndarray] = None,
    stellar_mass: Optional[float] = None,
    gamma: float = 0.0,
    transit_softness: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Evaluate the light curve AND the radial-velocity curve together,
    sharing the expensive per-pixel-per-wavelength flux computation
    between them (see ``_compute_single_phase_lc_rv``) instead of
    computing it twice.

    For callers who need both from the same underlying system -- a joint
    transit-photometry + radial-velocity fit -- this is roughly 2x
    cheaper per call than ``make_lc(model, ...)`` followed by
    ``make_rv(model, ...)`` with the same arguments, each of which would
    otherwise independently recompute ``_compute_flux_discs``. If you only
    need one of the two, ``make_lc``/``make_rv`` alone remain the right
    (and no more expensive) choice -- this function exists purely to avoid
    that redundant computation when both outputs are wanted, not to
    replace either.

    This function is **pure JAX** -- all inputs may be JAX arrays or
    tracers, making it fully compatible with ``jit``, ``vmap``, and
    gradient-based samplers, exactly like ``make_lc``/``make_rv``.

    Accepts the **identical** parameter set as ``make_lc``/``make_rv``
    (same all-or-nothing rules, same time-varying-AR support, same
    static/dynamic/dummy transit dispatch, same ``planet_mass``/
    ``stellar_mass``/``gamma`` semantics for the Keplerian RV term) -- see
    their docstrings for the full parameter semantics; nothing here is
    new or combined-specific except the return value.

    Parameters
    ----------
    model : dict
        Pre-built model dict returned by ``build_system``.
    flux_active : jnp.ndarray, shape (nar, nwave), (nwave,), or (ntime, nar, nwave), optional
        Per-active-region flux / spectrum. Must be >= 0.
        - If (nwave,):     broadcasts to all active regions.
        - If (nar, nwave): each active region gets its own spectrum.
        - If (ntime, nar, nwave): each active region gets its own time-varying spectrum.
    ar_lat : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region latitudes in degrees. Must be in [-90, 90].
    ar_long : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region longitudes in degrees. Must be in [0, 360).
    ar_size : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region angular radii in degrees. Must be >= 0.
    ar_smoothness : jnp.ndarray, shape (nar,), scalar, or (ntime, nar), optional
        Super-Gaussian order controlling the sharpness of each AR's
        boundary (see ``_compute_ar_shape``). ``1`` is a true Gaussian;
        larger values sharpen the edge, converging to a hard-edged cap as
        ``ar_smoothness -> inf``. A scalar (or size-1 array) is broadcast
        to all active regions.

        ``flux_active``/``ar_lat``/``ar_long``/``ar_size``/``ar_smoothness``
        are all-or-nothing: give every one of them to add active region(s),
        or omit all five for a quiet star. Giving some but not all raises
        ``ValueError``.

        **Time-varying active regions.** Any of the above five properties may
        independently carry an extra leading axis of length ``ntime`` instead
        of its usual shape, to let that property evolve over the observation,
        e.g. a spot that grows/decays (``ar_size``), drifts in latitude/longitude
        (``ar_lat``/``ar_long``), or changes contrast (``flux_active``).
        Values are given per original time and interpolated internally onto
        ``oversample`` sub-exposures when oversampling is active, using the
        model's ``ar_time_interp`` law (set at ``build_system`` time, like
        ``ld_mode`` -- see there for the "linear"/"cubic" choice). Mixing is
        allowed: only the parameters you want to evolve need the extra axis,
        the rest keep their usual constant-in-time shape. Using none of them
        (the default) is exactly as fast as before this capability existed.
        ``ld_coeffs_active``/``I_profile_active`` do not support time evolution
        and always stay fixed across the observation. Each epoch's values are
        used exactly as given -- nothing couples them across epochs, so it's
        up to the user to keep a fitted active region from changing unphysically
        fast between epochs. If ``nar`` (read off the parameters' own trailing axis)
        happens to equal ``ntime``, make_lc warns: this is the shape signature of
        the common mistake of leaving a parameter meant to evolve as a flat ``(ntime,)``
        array instead of ``(ntime, 1)``, which silently produces ``ntime`` separate
        static active regions instead of one evolving one.
    ld_coeffs_active : jnp.ndarray, shape (nar, nwave, n_coeffs) or (nwave, n_coeffs), optional
        Per-active-region limb-darkening coefficients, same law as the
        quiet photosphere (``model["ld_mode"]``) but independent values.
        A (nwave, n_coeffs) array broadcasts to all active regions.
        Defaults to the quiet photosphere's own coefficients if omitted.
        Not used when ``ld_mode="intensity_profile"``.
    I_profile_active : jnp.ndarray, shape (nar, nwave, n_mu_pts) or (nwave, n_mu_pts), optional
        Per-active-region specific-intensity profile, used only when
        ``ld_mode="intensity_profile"``. Defaults to the quiet
        photosphere's own profile if omitted.
    ld_coeffs_quiet : jnp.ndarray, shape (nwave, n_coeffs), optional
        Dynamic override for the quiet photosphere's own limb-darkening
        coefficients (JAX values/tracers are fine) -- the build-time
        ``ld_coeffs`` given to ``build_system`` is
        otherwise fixed for the life of the model, exactly like the
        transit-geometry parameters were before they got this same
        treatment. Defaults to the static value ``model`` was built with.
        When given, active regions that don't specify their own
        ``ld_coeffs_active`` default to this (possibly traced) value too,
        instead of the static one.
    t0, period, a_over_rstar, inclination : float or jnp.ndarray, shape (nplanet,), optional
        Transit-geometry parameters: mid-transit epoch, orbital period [days],
        semi-major axis/R*, and orbital inclination [rad]. Scalar or
        ``(nplanet,)``, same trailing-axis convention as ``build_system``
        (``nplanet`` inferred from ``t0``). When overriding a model built
        with a different ``nplanet``, also override ``ecc``/``omega_peri``/
        ``sp_orb`` if you don't want them to fall back to the build-time defaults.
    k : float or array-like, shape (nplanet,) or (nplanet, nwave), optional
        Planet-to-star radius ratio Rp/R*. A scalar (the same value for
        every planet and wavelength), an array of length ``nplanet`` (one
        achromatic value per planet), or an array of shape
        ``(nplanet, nwave)`` (a chromatic transit depth, independent per
        planet). For a single planet (``nplanet == 1``), a bare array of
        length ``nwave`` is also accepted as that planet's chromatic depth
        (legacy convention). Multiple overlapping planets combine their
        occultation multiplicatively (they are opaque, unlike active
        regions' additive contrast -- see ``make_lc``), so overlapping
        transits never drive flux negative.
        The planet's parameters are all-or-nothing, exactly like the AR parameters:
        give every one of them to evaluate a transit at those (possibly
        traced) values instead of the static ones ``model`` was built
        with, or omit all five to fall back to the model's static transit
        (or to no transit, if it doesn't have one) -- ``ValueError`` if
        only some are given.
    ecc, omega_peri : float or jnp.ndarray, shape (nplanet,), optional
        Orbital eccentricity and argument of periastron [rad]. Only
        meaningful together with a transit; default to 0.0 (circular,
        non-precessing orbit). Only used together with the five required
        transit parameters above (giving either of these without the rest
        also raises ``ValueError``). Default to the values ``model``'s transit
        was built with. Scalar or ``(nplanet,)``, same broadcasting
        rule as ``t0``/``period``/etc.
    sp_orb : float or jnp.ndarray, shape (nplanet,), optional
        Sky-projected spin-orbit angle λ [deg]. Same all-or-nothing-with-
        ``ecc``/``omega_peri`` treatment: only used together with a
        transit, defaults to the value ``model``'s transit was built with
        (also in degrees). Converted to radians internally before use --
        ``sajax.planet.planet_sky_position`` itself takes radians. Scalar or ``(nplanet,)``,
        same broadcasting rule as ``t0``/``period``/etc.
    planet_mass : float or jnp.ndarray, shape (nplanet,), optional
        Planet mass(es) [M_Jup], >= 0. Required together with
        ``stellar_mass`` whenever a transit/orbit is attached (via
        ``build_system`` or this call's own ``t0``/``period``/
        ``a_over_rstar``/``inclination``/``k``). A scalar broadcasts
        to every planet; pass an explicit ``(nplanet,)``
        array for independent masses.
    stellar_mass : float, optional
        Stellar mass [M_sun], > 0. Required together with ``planet_mass``.
    gamma : float, optional (default 0.0)
        Systemic velocity offset [km/s], added to the total RV.
    transit_softness : float, optional
        Sigmoid transition width [R*] for the planet occultation mask
        (default 0.0: exact hard edge, matching the physical simulation).
        The hard edge makes occulted flux a staircase function of every
        transit-geometry parameter on the fixed pixel grid, so
        ``jax.grad`` w.r.t. ``k``/``a_over_rstar``/``inclination``/``t0``/
        ``period``/``ecc``/``omega_peri`` is exactly 0 almost everywhere
        regardless of the values passed in above. Set this > 0 (e.g. a few
        tenths of a pixel in R* units) to get a smooth, non-zero gradient
        for gradient-based retrieval of those parameters. See
        ``_compute_planet_mask`` for details and trade-offs.


    Returns
    -------
    (lc, rv, star_maps) tuple
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    ``lc``        - (ntimes, nwave) disc-integrated flux at each
                    wavelength bin, in the same units as
                    ``flux_quiet``/``flux_active`` (not normalised to the
                    quiet-star baseline -- divide by that yourself if you
                    want relative flux). If ``nwave == 1``, the wavelength
                    axis is dropped and this is shape (ntimes,).
    ``rv``        - (ntimes, nwave) radial velocity [km/s] at each
                    wavelength bin: ``gamma`` + the Keplerian reflex term
                    + the activity/Rossiter-McLaughlin term. If ``nwave == 1``,
                    the wavelength axis is dropped and this is shape (ntimes,).
    ``star_maps`` - (ntimes, n, n) stellar flux map per phase
                    (maps are from the *first* sub-exposure of each phase
                    when oversampling is active)
    """
    ar = _resolve_ar_params(model, flux_active, ar_lat, ar_long, ar_size, ar_smoothness)
    nar   = ar["nar"]
    nwave = model["nwave"]

    oversample      = model["oversample"]
    nphase_original = model["nphase_original"]
    nphase_compute  = model["phases_rot"].shape[0]

    ar_time_varying = ar["ar_time_varying"]
    flux_active_r   = ar["flux_active"]
    ar_lat_r        = ar["ar_lat"]
    ar_long_r       = ar["ar_long"]
    ar_size_r       = ar["ar_size"]
    ar_smoothness_r = ar["ar_smoothness"]
    all_ar_carts    = ar["all_ar_carts"]

    ld_coeffs_quiet_val, ld_coeffs_active_r, I_profile_active_r = _resolve_ld_coeffs(
        model, ld_coeffs_quiet, ld_coeffs_active, I_profile_active, nar,
    )

    transit = _resolve_transit_params(
        model, t0, period, a_over_rstar, inclination, ecc, omega_peri, sp_orb, k, nwave,
    )
    planet_xyz_all = transit["planet_xyz_all"]
    k_val          = transit["k_val"]

    # ---- Keplerian term (identical validation/derivation to make_rv) ------
    _mass_args = dict(planet_mass=planet_mass, stellar_mass=stellar_mass)
    _mass_given = {name: v for name, v in _mass_args.items() if v is not None}
    if _mass_given and len(_mass_given) != len(_mass_args):
        _missing = [name for name in _mass_args if name not in _mass_given]
        raise ValueError(
            "make_lc_and_rv: partial mass parameters given "
            f"({sorted(_mass_given)}); missing {_missing}. Provide both "
            "planet_mass and stellar_mass to compute the Keplerian RV "
            "term, or neither for a pure activity/RM RV curve."
        )
    if _mass_given and not transit["has_transit"]:
        raise ValueError(
            "make_lc_and_rv: planet_mass/stellar_mass were given, but no "
            "transit/orbit is attached -- provide t0/period/a_over_rstar/"
            "inclination/k (via build_system or this call) to attach one, "
            "or omit planet_mass/stellar_mass for a pure activity/RM RV curve."
        )
    if transit["has_transit"] and not _mass_given:
        raise ValueError(
            "make_lc_and_rv: this model has a transit/orbit attached, but "
            "planet_mass and stellar_mass were not given -- both are "
            "required to compute the Keplerian RV amplitude for an "
            "attached transit (there is no physically sensible default "
            "for a system known to host a planet). Omit the transit's "
            "parameters entirely for a pure activity/RM RV curve."
        )

    if _mass_given:
        # Check the *computed* boolean result for tracer-ness, not
        # planet_mass itself -- see the identical comment in _resolve_ar_params.
        _planet_mass_check = jnp.any(jnp.asarray(planet_mass) < 0)
        if not isinstance(_planet_mass_check, jax.core.Tracer) and bool(_planet_mass_check):
            raise ValueError(
                f"planet_mass must be >= 0 (got {planet_mass}); planet_mass "
                "is a mass and cannot be negative."
            )
        if not isinstance(stellar_mass, jax.core.Tracer) and float(stellar_mass) <= 0:
            raise ValueError(
                f"stellar_mass must be > 0 (got {stellar_mass})."
            )
        planet_mass_arr = _broadcast_orbital_param(
            "planet_mass", planet_mass, transit["nplanet"], "make_lc_and_rv"
        )
        K_arr = keplerian_rv_semi_amplitude(
            planet_mass_arr, stellar_mass, transit["period_val"],
            transit["ecc_val"], transit["inclination_val"],
        )
        rv_kep_raw = compute_multi_planet_keplerian_rv(
            model["times_oversampled"], transit["t0_val"], transit["period_val"],
            transit["ecc_val"], transit["omega_peri_val"], K_arr,
        )   # (nphase_compute,)
    else:
        rv_kep_raw = jnp.zeros(nphase_compute)

    # ---- Light curve + activity/RM RV + star map, all from ONE shared
    # per-phase flux computation --------------------------------------------
    if not ar_time_varying:
        lc_raw, rv_arm_raw, star_maps_raw = _compute_all_phases_lc_rv(
            all_ar_carts,
            planet_xyz_all,
            wavelength          = model["wavelength"],
            flux_quiet          = model["flux_quiet"],
            flux_active         = flux_active_r,
            ld_coeffs_quiet     = ld_coeffs_quiet_val,
            ld_coeffs_active    = ld_coeffs_active_r,
            I_profile_quiet     = model["I_profile"],
            I_profile_active    = I_profile_active_r,
            mu_profile_pts      = model["mu_profile_pts"],
            x_disc              = model["x_disc"],
            y_disc              = model["y_disc"],
            mu_disc             = model["mu_disc"],
            col_idx             = model["col_idx"],
            vel_col             = model["vel_col"],
            star_pixel_rad      = model["star_pixel_rad"],
            total_pixels        = model["total_pixels"],
            arsize_rads         = jnp.deg2rad(ar_size_r),
            ar_smoothness       = ar_smoothness_r,
            k                   = k_val,
            ld_mode             = model["ld_mode"],
            plot_map_wavelength = model["plot_map_wavelength"],
            n                   = model["n"],
            flat_indices        = model["flat_indices"],
            transit_softness    = transit_softness,
        )
    else:
        lc_raw, rv_arm_raw, star_maps_raw = _compute_all_phases_lc_rv_evolving(
            ar_lat_r, ar_long_r,
            jnp.deg2rad(ar_size_r),
            ar_smoothness_r,
            flux_active_r,
            planet_xyz_all,
            model["phases_rot"],
            inc_star            = model["inc_star"],
            star_pixel_rad      = model["star_pixel_rad"],
            wavelength          = model["wavelength"],
            flux_quiet          = model["flux_quiet"],
            ld_coeffs_quiet     = ld_coeffs_quiet_val,
            ld_coeffs_active    = ld_coeffs_active_r,
            I_profile_quiet     = model["I_profile"],
            I_profile_active    = I_profile_active_r,
            mu_profile_pts      = model["mu_profile_pts"],
            x_disc              = model["x_disc"],
            y_disc              = model["y_disc"],
            mu_disc             = model["mu_disc"],
            col_idx             = model["col_idx"],
            vel_col             = model["vel_col"],
            total_pixels        = model["total_pixels"],
            k                   = k_val,
            ld_mode             = model["ld_mode"],
            plot_map_wavelength = model["plot_map_wavelength"],
            n                   = model["n"],
            flat_indices        = model["flat_indices"],
            transit_softness    = transit_softness,
        )

    # ---- Oversample averaging ----------
    if oversample > 1:
        lc_avg     = lc_raw.reshape(nphase_original, oversample, nwave).mean(axis=1)
        rv_kep_avg = rv_kep_raw.reshape(nphase_original, oversample).mean(axis=1)
        rv_arm_avg = rv_arm_raw.reshape(nphase_original, oversample, nwave).mean(axis=1)
        star_maps  = star_maps_raw[::oversample]
    else:
        lc_avg, rv_kep_avg, rv_arm_avg = lc_raw, rv_kep_raw, rv_arm_raw
        star_maps = star_maps_raw

    rv = gamma + rv_kep_avg[:, None] + rv_arm_avg   # (nphase, nwave)

    # ---- Single-wavelength convenience: drop the now-degenerate nwave axis on both outputs --
    if nwave == 1:
        lc_avg = lc_avg[..., 0]
        rv = rv[..., 0]

    return lc_avg, rv, star_maps


def quick_lc_and_rv(
    wavelength: np.ndarray,
    flux_quiet: np.ndarray,
    flux_active: np.ndarray,
    ar_lat: np.ndarray,
    ar_long: np.ndarray,
    ar_size: np.ndarray,
    ar_smoothness: np.ndarray,
    times: np.ndarray,
    P_rot: float,
    stellar_grid_size: int = 100,
    ve: float = 0.0,
    ld_coeffs: Optional[list] = None,
    inc_star: float = 90.0,
    mu_profile: Optional[np.ndarray] = None,
    I_profile: Optional[np.ndarray] = None,
    ld_mode: LdMode = "quadratic",
    ld_coeffs_active: Optional[np.ndarray] = None,
    I_profile_active: Optional[np.ndarray] = None,
    plot_map_wavelength: Optional[float] = None,
    oversample: int = 1,
    t0: Optional[float | np.ndarray] = None,
    period: Optional[float | np.ndarray] = None,
    a_over_rstar: Optional[float | np.ndarray] = None,
    inclination: Optional[float | np.ndarray] = None,
    k: Optional[float | np.ndarray] = None,
    ecc: float | np.ndarray = 0.0,
    omega_peri: float | np.ndarray = 0.0,
    sp_orb: float | np.ndarray = 0.0,
    ar_time_interp: ArTimeInterp = "linear",
    planet_mass: Optional[float | np.ndarray] = None,
    stellar_mass: Optional[float] = None,
    gamma: float = 0.0,
    transit_softness: float = 0.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: build model and evaluate in one call.

    Equivalent to::

        model = build_system(wavelength, flux_quiet, times, P_rot, ...)
        lc, rv, star_maps = make_lc_and_rv(
            model, flux_active, ar_lat, ar_long, ar_size, ar_smoothness,
            ..., planet_mass=planet_mass, stellar_mass=stellar_mass,
            gamma=gamma,
        )

    Use ``build_system`` + ``make_lc_and_rv`` directly when running MCMC
    so the grid is built only once.

    Parameters
    ----------
    wavelength : array_like, shape (nwave,)
        Wavelength value or array at which the quiet photosphere and active-region spectra are defined.
    flux_quiet : array_like, shape (nwave,)
        Quiet-photosphere flux / spectrum.
    flux_active : jnp.ndarray, shape (nar, nwave), (nwave,), or (ntime, nar, nwave), optional
        Per-active-region flux / spectrum. Must be >= 0.
        - If (nwave,):     broadcasts to all active regions.
        - If (nar, nwave): each active region gets its own spectrum.
        - If (ntime, nar, nwave): each active region gets its own time-varying spectrum.
    ar_lat : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region latitudes in degrees. Must be in [-90, 90].
    ar_long : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region longitudes in degrees. Must be in [0, 360).
    ar_size : jnp.ndarray, shape (nar,) or (ntime, nar), optional
        active region angular radii in degrees. Must be >= 0.
    ar_smoothness : jnp.ndarray, shape (nar,), scalar, or (ntime, nar), optional
        Super-Gaussian order controlling the sharpness of each AR's
        boundary (see ``_compute_ar_shape``). ``1`` is a true Gaussian;
        larger values sharpen the edge, converging to a hard-edged cap as
        ``ar_smoothness -> inf``. A scalar (or size-1 array) is broadcast
        to all active regions.

        ``flux_active``/``ar_lat``/``ar_long``/``ar_size``/``ar_smoothness``
        are all-or-nothing: give every one of them to add active region(s),
        or omit all five for a quiet star. Giving some but not all raises
        ``ValueError``.

        **Time-varying active regions.** Any of the above five properties may
        independently carry an extra leading axis of length ``ntime`` instead
        of its usual shape, to let that property evolve over the observation,
        e.g. a spot that grows/decays (``ar_size``), drifts in latitude/longitude
        (``ar_lat``/``ar_long``), or changes contrast (``flux_active``).
        Values are given per original time and interpolated internally onto
        ``oversample`` sub-exposures when oversampling is active, using the
        model's ``ar_time_interp`` law (set at ``build_system`` time, like
        ``ld_mode`` -- see there for the "linear"/"cubic" choice). Mixing is
        allowed: only the parameters you want to evolve need the extra axis,
        the rest keep their usual constant-in-time shape. Using none of them
        (the default) is exactly as fast as before this capability existed.
        ``ld_coeffs_active``/``I_profile_active`` do not support time evolution
        and always stay fixed across the observation. Each epoch's values are
        used exactly as given -- nothing couples them across epochs, so it's
        up to the user to keep a fitted active region from changing unphysically
        fast between epochs. If ``nar`` (read off the parameters' own trailing axis)
        happens to equal ``ntime``, make_lc warns: this is the shape signature of
        the common mistake of leaving a parameter meant to evolve as a flat ``(ntime,)``
        array instead of ``(ntime, 1)``, which silently produces ``ntime`` separate
        static active regions instead of one evolving one.
    times : array_like, shape (ntime,)
        Absolute observation times [days].
    P_rot : float
        Stellar rotation period [days].
    stellar_grid_size : int
        Radius of the stellar grid.
    ve : float
        Stellar equatorial rotational velocity [km/s].
    ld_coeffs : list of float or list of array(nwave,), optional
        Quiet-photosphere limb-darkening coefficients for the chosen
        ``ld_mode``:
        - ``"linear"``:     [u]
        - ``"quadratic"``:  [u1, u2]
        - ``"power2"``:     [c, alpha]
        - ``"kipping3"``:   [c1, c2, c3]
        - ``"nonlinear4"``: [c1, c2, c3, c4]
        Each element may be a scalar (broadcast to all wavelengths) or an
        array of length ``nwave``. Active regions carry their own,
        independent coefficients -- see ``make_lc``. Not used
        (and not required) when ``ld_mode="intensity_profile"``; its
        required length for every other mode is checked against
        ``ld_mode`` here.
    inc_star : float, optional
        Stellar inclination in degrees (default: 90.0).
        90deg = equator-on, 0deg = pole-on.
    mu_profile : array-like, optional
        Monotonically increasing mu grid points for
        ``ld_mode="intensity_profile"`` (default: [0, 1]).
    I_profile : array-like, shape (nwave, n_mu_pts), optional
        Quiet-photosphere specific intensity at each (wavelength, mu) grid
        point. Required when ``ld_mode="intensity_profile"``.
    ld_mode : str (default "quadratic")
        Limb-darkening law, shared by the quiet photosphere and every
        active region (each with its own coefficient values).
    ld_coeffs_active : jnp.ndarray, shape (nar, nwave, n_coeffs) or (nwave, n_coeffs), optional
        Per-active-region limb-darkening coefficients, same law as the
        quiet photosphere (``model["ld_mode"]``) but independent values.
        A (nwave, n_coeffs) array broadcasts to all active regions.
        Defaults to the quiet photosphere's own coefficients if omitted.
        Not used when ``ld_mode="intensity_profile"``.
    I_profile_active : jnp.ndarray, shape (nar, nwave, n_mu_pts) or (nwave, n_mu_pts), optional
        Per-active-region specific-intensity profile, used only when
        ``ld_mode="intensity_profile"``. Defaults to the quiet
        photosphere's own profile if omitted.
    plot_map_wavelength : float, optional
        Wavelength at which to plot the stellar map (see ``build_system``).
    oversample : int, optional
        Number of sub-exposures per phase point.  Each requested phase is
        spread into ``oversample`` uniformly spaced sub-phases spanning one
        phase step, and the resulting fluxes are averaged.  This mimics
        finite-exposure integration and smooths limb-crossing artefacts.
        Default: 1 (no oversampling).
    t0, period, a_over_rstar, inclination : float or array-like, shape (nplanet,), optional
        Transit-geometry parameters: mid-transit epoch, orbital period [days],
        semi-major axis/R*, and orbital inclination [rad]. All-or-nothing with
        ``k``. Each is a scalar or an array with a trailing ``(nplanet,)``
        axis -- exactly like ``ar_lat``/``ar_long``/etc. for active regions
        (see ``make_lc``) -- with ``nplanet`` inferred from ``t0``'s trailing
        axis. A scalar (or size-1 array) among the others broadcasts to every
        planet; giving more than one planet requires giving each of these
        four its own length-``nplanet`` array (or leaving it scalar to share
        one value across all planets).
    k : float or array-like, shape (nplanet,) or (nplanet, nwave), optional
        Planet-to-star radius ratio Rp/R*. A scalar (the same value for
        every planet and wavelength), an array of length ``nplanet`` (one
        achromatic value per planet), or an array of shape
        ``(nplanet, nwave)`` (a chromatic transit depth, independent per
        planet). For a single planet (``nplanet == 1``), a bare array of
        length ``nwave`` is also accepted as that planet's chromatic depth
        (legacy convention). Multiple overlapping planets combine their
        occultation multiplicatively (they are opaque, unlike active
        regions' additive contrast -- see ``make_lc``), so overlapping
        transits never drive flux negative.
        The planet's parameters are all-or-nothing, exactly like the AR parameters:
        give every one of them to evaluate a transit at those (possibly
        traced) values instead of the static ones ``model`` was built
        with, or omit all five to fall back to the model's static transit
        (or to no transit, if it doesn't have one) -- ``ValueError`` if
        only some are given.
    ecc, omega_peri : float or array-like, shape (nplanet,), optional
        Orbital eccentricity and argument of periastron [rad]. Only
        meaningful together with a transit; default to 0.0 (circular,
        non-precessing orbit). Scalar or ``(nplanet,)``, same broadcasting
        rule as ``t0``/``period``/etc.
    sp_orb : float or jnp.ndarray, shape (nplanet,), optional
        sky-projected spin-orbit angle, λ  [deg]
        Rotates the transit chord about the stellar
        centre, in the sky plane. Angle is relative to the
        stellar equator. Only meaningful together with a transit.
        Converted to radians internally before use --
        ``sajax.planet.planet_sky_position`` itself takes radians.
        Scalar or ``(nplanet,)``, same broadcasting rule as ``t0``/``period``/etc.
    ar_time_interp : "linear" or "cubic", optional (default "linear")
        Interpolation method for when time-varying active-region parameters
        need to be interpolated onto ``oversample``'d sub-exposures.
        Only matters when ``oversample > 1`` and at least one AR parameter
        is time-varying -- see ``make_lc`` for the full description.
    planet_mass : float or jnp.ndarray, shape (nplanet,), optional
        Planet mass(es) [M_Jup], >= 0. Required together with
        ``stellar_mass`` whenever a transit/orbit is attached (via
        ``build_system`` or this call's own ``t0``/``period``/
        ``a_over_rstar``/``inclination``/``k``). A scalar broadcasts
        to every planet; pass an explicit ``(nplanet,)``
        array for independent masses.
    stellar_mass : float, optional
        Stellar mass [M_sun], > 0. Required together with ``planet_mass``.
    gamma : float, optional (default 0.0)
        Systemic velocity offset [km/s], added to the total RV.
    verbose : bool, optional
        If True, print informational messages (LDC broadcasting, phase
        oversampling) while building the model. Default False.

    Returns
    -------
    (lc, rv, star_maps) tuple
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    ``lc``        - (ntimes, nwave) disc-integrated flux at each
                    wavelength bin, in the same units as
                    ``flux_quiet``/``flux_active`` (not normalised to the
                    quiet-star baseline -- divide by that yourself if you
                    want relative flux). If ``nwave == 1``, the wavelength
                    axis is dropped and this is shape (ntimes,).
    ``rv``        - (ntimes, nwave) radial velocity [km/s] at each
                    wavelength bin: ``gamma`` + the Keplerian reflex term
                    + the activity/Rossiter-McLaughlin term. If ``nwave == 1``,
                    the wavelength axis is dropped and this is shape (ntimes,).
    ``star_maps`` - (ntimes, n, n) stellar flux map per phase
                    (maps are from the *first* sub-exposure of each phase
                    when oversampling is active)
    """
    model = build_system(
        wavelength, flux_quiet, times=times, P_rot=P_rot,
        stellar_grid_size=stellar_grid_size, ve=ve,
        ld_coeffs=ld_coeffs, inc_star=inc_star, mu_profile=mu_profile,
        I_profile=I_profile, ld_mode=ld_mode,
        plot_map_wavelength=plot_map_wavelength,
        oversample=oversample, ar_time_interp=ar_time_interp,
        t0=t0, period=period, a_over_rstar=a_over_rstar,
        inclination=inclination, k=k, ecc=ecc, omega_peri=omega_peri,
        sp_orb=sp_orb,
        verbose=verbose,
    )

    flux_active_arr = np.atleast_1d(np.asarray(flux_active, dtype=np.float32))
    lc, rv, star_maps = make_lc_and_rv(
        model,
        jnp.asarray(flux_active_arr),
        jnp.asarray(np.atleast_1d(np.asarray(ar_lat,  dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_long, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_size, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_smoothness, dtype=np.float32))),
        None if ld_coeffs_active is None else jnp.asarray(np.asarray(ld_coeffs_active, dtype=np.float32)),
        None if I_profile_active  is None else jnp.asarray(np.asarray(I_profile_active,  dtype=np.float32)),
        planet_mass=planet_mass, stellar_mass=stellar_mass, gamma=gamma,
        transit_softness=transit_softness,
    )
    return np.array(lc), np.array(rv), np.array(star_maps)

