"""
core.py -- JAX-accelerated stellar active region light-curve engine.

This module is a complete rewrite of ``SAGE1/sage.py`` in JAX.

"""

from __future__ import annotations

import functools
import warnings
from typing import Literal, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import erfc
import interpax

from .geometry import rotate_active_region
from .planet import _compute_all_planets_mask, _all_planets_mask_from_d2, compute_multi_planet_sky_positions

_MIN_RESOLVED_PLANET_PX = 5.0

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
#  (see _pixel_flux_at_wavelength). Some limb-darkening laws (e.g. a user-supplied
# intensity_profile with I(mu=0)=0) can vanish at the limb, making the local
# F_quiet exactly 0 there; without a floor this causes NaN which corrupt the
# dimming surface D even though that pixel's contribution is supposed to be 0.
_FLUX_RATIO_EPS = 1e-12


# ---------------------------------------------------------------------------
# 0. Utilities
# ---------------------------------------------------------------------------

def flare_template(t, tpeak, fwhm, ampl=1.0):
    """Continuous flare template of Tovar Mendoza et al. (2022): the
    convolution of a Gaussian with a double exponential, evaluated at
    times ``t`` with peak time ``tpeak``, full width at half maximum
    ``fwhm`` (same units as ``t``), and peak amplitude ``ampl``.
    Coefficients are the published MCMC fit from
    https://github.com/lupitatovar/Llamaradas-Estelares -- with them the
    template's actual maximum is ~0.95 * ``ampl``, at ``t`` ~ ``tpeak``
    (matching the reference implementation, which is not renormalised
    to peak at exactly ``ampl`` either).

    Times more than 20 FWHM before the peak return exactly 0 (the true
    template is ~1e-70 there): the rise-side time argument is clamped so
    the ``exp * erfc`` product can never overflow to NaN - in value or in
    gradient - on long baselines.
    """
    A, B, C, D1, D2, f1 = (0.9687734504375167, -0.251299705922117,
                           0.22675974948468916, 0.15551880775110513,
                           1.2150539528490194, 0.12695865022878844)
    f2 = 1.0 - f1
    # Time in units of FWHM from the peak. Clamped on the rise side: at
    # tn = -20 the erfc factor has underflowed to exactly 0 while the exp
    # factor is still finite, so the product is exactly 0 there and for
    # all earlier times - and the clamp keeps exp from overflowing to
    # inf (NaN forward values and gradients) further out.
    tn = jnp.maximum((t - tpeak) / fwhm, -20.0)

    def _piece(D, f):
        return (0.5 * jnp.sqrt(jnp.pi) * A * C * f
                * jnp.exp(-D * tn + (B / C + D * C / 2.0) ** 2)
                * erfc((B - tn) / C + C * D / 2.0))

    return ampl * (_piece(D1, f1) + _piece(D2, f2))


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

    # mu = cos theta = sqrt(1 - (r/R)^2), clamped for float32 safety
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
    ``_pixel_flux_at_wavelength``). Thus, this function is purely geometric and
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
        # I(mu) = 1 - u*(1 - mu)
        result = 1.0 - ld_coeffs_wl[0] * (1.0 - mu_disc)
    elif ld_mode == "quadratic":
        # I(mu) = 1 - u1*(1-mu) - u2*(1-mu)^2
        result = (1.0
                  - ld_coeffs_wl[0] * (1.0 - mu_disc)
                  - ld_coeffs_wl[1] * (1.0 - mu_disc) ** 2)
    elif ld_mode == "power2":
        # I(mu) = 1 - a*(1 - mu^b)
        result = 1.0 - ld_coeffs_wl[0] * (1.0 - mu_disc ** ld_coeffs_wl[1])
    elif ld_mode == "kipping3":
        # I(mu) = 1 - c1*(1-mu^0.5) - c2*(1-mu) - c3*(1-mu^(3/2))
        result = (1.0
                  - ld_coeffs_wl[0] * (1.0 - mu_disc ** 0.5)
                  - ld_coeffs_wl[1] * (1.0 - mu_disc)
                  - ld_coeffs_wl[2] * (1.0 - mu_disc ** 1.5))
    else:  # "nonlinear4"  -- Claret (2000) four-parameter law
        # I(mu) = 1 - sum_{k=1}^{4} c_k*(1 - mu^(k/2))
        result = (1.0
                  - ld_coeffs_wl[0] * (1.0 - mu_disc ** 0.5)
                  - ld_coeffs_wl[1] * (1.0 - mu_disc)
                  - ld_coeffs_wl[2] * (1.0 - mu_disc ** 1.5)
                  - ld_coeffs_wl[3] * (1.0 - mu_disc ** 2.0))

    # Intensity can't be negative. Unphysical LDCs (e.g. u1+u2 > 1 for the
    # quadratic law) otherwise dip slightly negative near the limb, which
    # the _FLUX_RATIO_EPS floor in _pixel_flux_at_wavelength turns into a division
    # by (almost) zero -- a spurious contrast blowup rather than a graceful
    # degradation to zero flux.
    return jnp.maximum(result, 0.0)


# ---------------------------------------------------------------------------
# 4. Per-wavelength pixel flux (vmapped over the spectral axis, phase-free)
# ---------------------------------------------------------------------------

_DOT_PRECISION = jax.lax.Precision.HIGHEST  # true float32 GEMMs (GPU default is TF32)


def _doppler_shifted(wavelength_target, vel_col, wavelength_grid, spectra):
    """``spectra`` (nwave,) or (nar, nwave) resampled at every column's Doppler-shifted wavelength: (n,) or (nar, n).
    Velocity depends on x alone (see build_stellar_grid), so resample once per column, not per pixel."""
    query_wavelength_col = wavelength_target * (1.0 - vel_col)  # (n,)
    if spectra.ndim == 1:
        return jnp.interp(query_wavelength_col, wavelength_grid, spectra)
    return vmap(lambda spec: jnp.interp(query_wavelength_col, wavelength_grid, spec))(spectra)


def _quiet_pixel_flux_at_wavelength(
    # --- vmapped: one scalar/slice per wavelength ---
    wavelength_target:    float,
    ld_coeffs_quiet_wl:  jnp.ndarray, # (n_coeffs,)
    I_prof_quiet_wl:      jnp.ndarray, # (n_mu_pts,)
    # --- broadcast: shared across wavelengths ---
    wavelength_grid: jnp.ndarray, # (nwave,) full spectral axis
    flux_quiet:      jnp.ndarray, # (nwave,) full quiet-photosphere spectrum
    mu_disc:         jnp.ndarray, # (total_pixels,) grid of in-disc mu values
    mu_profile_pts:  jnp.ndarray, # (n_mu_pts,)
    col_idx:         jnp.ndarray, # (total_pixels,) int
    vel_col:         jnp.ndarray, # (n,)
    ld_mode:        LdMode,
) -> jnp.ndarray:
    """Quiet-photosphere flux per pixel at one wavelength (Doppler x LD), (total_pixels,)."""
    F_quiet_local = _doppler_shifted(wavelength_target, vel_col, wavelength_grid, flux_quiet)[col_idx]
    ldc_quiet = _evaluate_ldc(mu_disc, ld_coeffs_quiet_wl, I_prof_quiet_wl, mu_profile_pts, ld_mode)
    return F_quiet_local * ldc_quiet


def _pixel_flux_at_wavelength(
    # --- vmapped: one scalar/slice per wavelength ---
    wavelength_target:    float,
    ld_coeffs_quiet_wl:  jnp.ndarray, # (n_coeffs,)
    ld_coeffs_active_wl: jnp.ndarray, # (nar, n_coeffs)
    I_prof_quiet_wl:      jnp.ndarray, # (n_mu_pts,)
    I_prof_active_wl:     jnp.ndarray, # (nar, n_mu_pts)
    # --- broadcast: shared across wavelengths ---
    wavelength_grid: jnp.ndarray, # (nwave,) full spectral axis
    flux_quiet:      jnp.ndarray, # (nwave,) full quiet-photosphere spectrum
    flux_active:     jnp.ndarray, # (nar, nwave) full per-AR spectra
    mu_disc:         jnp.ndarray, # (total_pixels,) grid of in-disc mu values
    mu_profile_pts:  jnp.ndarray, # (n_mu_pts,)
    col_idx:         jnp.ndarray, # (total_pixels,) int
    vel_col:         jnp.ndarray, # (n,)
    ld_mode:        LdMode,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Phase-independent per-pixel flux terms at one wavelength (Doppler x LD).

    Returns
    -------
    F_quiet_px : (total_pixels,)      quiet flux per pixel
    D          : (nar, total_pixels)  F_quiet_px * (1 - C_a), flux removed per
                                      pixel by AR ``a`` at full coverage
    """
    F_quiet_px = _quiet_pixel_flux_at_wavelength(
        wavelength_target, ld_coeffs_quiet_wl, I_prof_quiet_wl,
        wavelength_grid=wavelength_grid, flux_quiet=flux_quiet, mu_disc=mu_disc,
        mu_profile_pts=mu_profile_pts, col_idx=col_idx, vel_col=vel_col, ld_mode=ld_mode,
    )
    F_active_local = _doppler_shifted(wavelength_target, vel_col, wavelength_grid, flux_active)[:, col_idx]  # (nar, total_pixels)

    # ---- Limb darkening (own law coefficients per AR) --------------------
    ldc_active = vmap(
        lambda coeffs, iprof: _evaluate_ldc(
            mu_disc, coeffs, iprof, mu_profile_pts, ld_mode,
        )
    )(ld_coeffs_active_wl, I_prof_active_wl)  # (nar, total_pixels)

    # ---- Physical flux (Doppler-shifted spectrum x limb darkening) ------
    F_active_px = F_active_local * ldc_active  # (nar, total_pixels)

    C_a = F_active_px / jnp.maximum(F_quiet_px[None, :], _FLUX_RATIO_EPS)  # (nar, total_pixels); floor avoids 0/0 NaN
    D   = F_quiet_px[None, :] * (1.0 - C_a)                                 # (nar, total_pixels)
    return F_quiet_px, D


_LD_BASIS_LAWS = ("linear", "quadratic", "kipping3", "nonlinear4")   # laws linear in their coefficients


def _ld_basis(mu: np.ndarray, ld_mode: LdMode) -> np.ndarray:
    """B_j(mu), (n_coeffs + 1, total_pixels), with I(mu) = B_0 + sum_j u_j B_j for the laws in _LD_BASIS_LAWS."""
    om = 1.0 - mu
    if ld_mode == "linear":
        B = [np.ones_like(mu), -om]
    elif ld_mode == "quadratic":
        B = [np.ones_like(mu), -om, -om ** 2]
    elif ld_mode == "kipping3":
        B = [np.ones_like(mu), -(1.0 - mu ** 0.5), -om, -(1.0 - mu ** 1.5)]
    else:
        B = [np.ones_like(mu), -(1.0 - mu ** 0.5), -om, -(1.0 - mu ** 1.5), -(1.0 - mu ** 2.0)]
    return np.stack(B).astype(np.float32)


def _column_sums(A, B, col_idx, n):
    """sum_{p in column c} A[..., p] B_j(mu_p) for every (j, c), flattened: (..., P) -> (..., (n_coeffs + 1) * n)."""
    H = jnp.stack([jnp.zeros(A.shape[:-1] + (n,), A.dtype).at[..., col_idx].add(A * Bj) for Bj in B], axis=-2)
    return H.reshape(A.shape[:-1] + (-1,))


def _column_terms_at_wavelength(
    wavelength_target:    float,
    ld_coeffs_quiet_wl:  jnp.ndarray, # (n_coeffs,)
    ld_coeffs_active_wl: jnp.ndarray, # (nar, n_coeffs)
    wavelength_grid: jnp.ndarray,
    flux_quiet:      jnp.ndarray,
    flux_active:     jnp.ndarray, # (nar, nwave)
    vel_col:         jnp.ndarray, # (n,)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Column factors of the pixel terms: F_quiet(p) = sum_j Gq[(j, c_p)] B_j(mu_p) and D_a likewise with GD[a];
    Gq is ((n_coeffs + 1) * n,), GD is (nar, (n_coeffs + 1) * n)."""
    F_quiet_col  = _doppler_shifted(wavelength_target, vel_col, wavelength_grid, flux_quiet)             # (n,)
    F_active_col = _doppler_shifted(wavelength_target, vel_col, wavelength_grid, flux_active)            # (nar, n)
    c_quiet  = jnp.concatenate([jnp.ones(1, ld_coeffs_quiet_wl.dtype), ld_coeffs_quiet_wl])             # (nb,)
    c_active = jnp.concatenate([jnp.ones((ld_coeffs_active_wl.shape[0], 1), ld_coeffs_active_wl.dtype), ld_coeffs_active_wl], axis=-1)
    Gq = c_quiet[:, None] * F_quiet_col[None, :]                                                         # (nb, n)
    GD = Gq[None] - c_active[:, :, None] * F_active_col[:, None, :]                                     # (nar, nb, n)
    return Gq.reshape(-1), GD.reshape(GD.shape[0], -1)


def _integrate_flux(
    ar_shapes:    Optional[jnp.ndarray], # (nphase, nar, total_pixels); None for a quiet star
    planet_mask:  Optional[jnp.ndarray], # None | (nphase, total_pixels) | (nphase, nwave, total_pixels)
    F_quiet_px:   jnp.ndarray,           # (nwave, total_pixels)
    D:            Optional[jnp.ndarray], # (nwave, nar, total_pixels); None for a quiet star
    total_pixels: int,
    nphase:       int,
    geometry:     dict,                          # precompute_geometry output (may be empty), see there for keys
    compute_map:  bool = False,
    map_idx:      Optional[jnp.ndarray] = None,  # wavelength index of the map (compute_map only)
    use_columns:  bool = False,                  # contract the geometry's column sums against Gq / GD
    Gq:           Optional[jnp.ndarray] = None,  # (nwave, K) column terms
    GD:           Optional[jnp.ndarray] = None,  # (nwave, nar, K)
) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Disc-integrate ``[F_quiet - sum_a D_a S_a] * [1 - M]`` over pixels for
    every (phase, wavelength), term by term, so the AR term is one GEMM and
    no (nphase, nwave, total_pixels) surface is built unless ``M`` is chromatic.
    ``planet_mask`` may cover only the phases ``geometry["transit_idx"]``.
    ``ar_shapes=None`` (quiet star) drops the AR terms altogether. With
    ``use_columns`` the quiet and spot terms (and the planet terms when the
    geometry holds ``H_M``) contract its column sums against ``Gq``/``GD``
    instead of pixel arrays, so ``F_quiet_px``/``D`` may be None.

    Returns
    -------
    lc        : (nphase, nwave)         flux normalised by total_pixels
    arted_map : (nphase, total_pixels)  per-pixel flux at ``map_idx``, or None
    """
    einsum = functools.partial(jnp.einsum, precision=_DOT_PRECISION)

    transit_idx = geometry.get("transit_idx")
    has_ar = ar_shapes is not None or (use_columns and "H_S" in geometry)
    quiet = einsum("wk,k->w", Gq, geometry["basis_sum"]) if use_columns else jnp.sum(F_quiet_px, axis=-1)  # (nwave,)
    lc    = jnp.broadcast_to(quiet[None, :], (nphase, quiet.shape[0]))
    if has_ar:  # spot term, (nphase, nwave)
        lc = lc - (einsum("nak,wak->nw", geometry["H_S"], GD) if use_columns else einsum("nap,wap->nw", ar_shapes, D))

    if use_columns and "H_M" in geometry:  # fixed k: planet terms from column sums too
        H_M, H_SM = geometry["H_M"], geometry.get("H_SM")
        occ = einsum("nk,wk->nw" if H_M.ndim == 2 else "nwk,wk->nw", H_M, Gq)
        if has_ar:
            occ = occ - einsum("nak,wak->nw" if H_SM.ndim == 3 else "nwak,wak->nw", H_SM, GD)
        lc = lc.at[transit_idx].add(-occ)
    elif planet_mask is not None:
        if not has_ar or transit_idx is None:
            S_in = ar_shapes
        else:
            S_in = geometry.get("ar_shapes_in", ar_shapes[transit_idx])
        if planet_mask.ndim == 2:  # achromatic mask (nphase_in, P): GEMMs
            occ = einsum("np,wp->nw", planet_mask, F_quiet_px)
            if has_ar:
                occ = occ - einsum("nap,wap->nw", S_in * planet_mask[:, None, :], D)
        else:  # chromatic mask (nphase_in, nwave, P): one fused pass over that surface
            unocculted = F_quiet_px[None, :, :]                           # (1, nwave, P)
            for a in range(ar_shapes.shape[1] if has_ar else 0):  # static loop keeps this a single fused expression
                unocculted = unocculted - S_in[:, a, None, :] * D[None, :, a, :]
            occulted = planet_mask * unocculted                           # (nphase_in, nwave, P)
            occ = einsum("nwp,p->nw", occulted, jnp.ones(occulted.shape[-1], occulted.dtype))  # dot beats reduce on XLA:CPU
        lc = lc - occ if transit_idx is None else lc.at[transit_idx].add(-occ)

    lc = lc / jnp.float32(total_pixels)

    if not compute_map:
        return lc, None

    arted = jnp.broadcast_to(F_quiet_px[map_idx][None, :], (nphase, F_quiet_px.shape[-1]))  # (nphase, P)
    if has_ar:
        arted = arted - einsum("nap,ap->np", ar_shapes, D[map_idx])
    if planet_mask is not None:
        M_map = planet_mask if planet_mask.ndim == 2 else planet_mask[:, map_idx]
        arted = arted * (1.0 - M_map) if transit_idx is None else arted.at[transit_idx].multiply(1.0 - M_map)
    return lc, arted


# ---------------------------------------------------------------------------
# 5. All-phases computation -- the factored pixel integral
# ---------------------------------------------------------------------------

def _ar_shapes_all(all_ar_carts, arsize_rads, ar_smoothness, x_disc, y_disc, star_pixel_rad):
    """AR shapes at every phase, (nphase, nar, total_pixels)."""
    return vmap(
        vmap(
            lambda cart, sr, sm: _compute_ar_shape(
                x_disc, y_disc, star_pixel_rad,
                cart[0], cart[1], cart[2], sr, sm,
            ),
            in_axes=(0, 0, 0),
        ),
        in_axes=(0, None, None),
    )(all_ar_carts, arsize_rads, ar_smoothness)


def _planet_mask_all(planet_xyz_all, k, k_achromatic, x_disc, y_disc, star_pixel_rad, transit_softness):
    """Planet mask at every phase: (nphase, P) for an achromatic k, else (nphase, nwave, P)."""
    if k_achromatic:
        k_planet = k[:, 0]                                            # (nplanet,)
        return vmap(
            lambda xyz: _compute_all_planets_mask(
                x_disc, y_disc, star_pixel_rad, xyz, k_planet, transit_softness,
            )
        )(planet_xyz_all)
    return vmap(
        lambda xyz: vmap(
            lambda k_wl: _compute_all_planets_mask(
                x_disc, y_disc, star_pixel_rad, xyz, k_wl, transit_softness,
            ),
            in_axes=1,
        )(k)
    )(planet_xyz_all)


def _planet_mask_from_geometry(geometry, k, k_achromatic, transit_softness):
    """In-transit planet mask from precomputed squared distances for a per-call (possibly traced) k."""
    d2, Z = geometry["d2_in"], geometry["z_in"]                       # (nphase_in, nplanet, P), (nphase_in, nplanet)
    one = lambda d2_t, z_t, k_t: _all_planets_mask_from_d2(d2_t, z_t, k_t, transit_softness)
    if k_achromatic:
        return vmap(one, in_axes=(0, 0, None))(d2, Z, k[:, 0])        # (nphase_in, P)
    return vmap(lambda d2_t, z_t: vmap(one, in_axes=(None, None, 1))(d2_t, z_t, k))(d2, Z)  # (nphase_in, nwave, P)


def _compute_all_phases(
    all_ar_carts:    Optional[jnp.ndarray],   # (nphase, nar, 3); None when ar_shapes is given
    planet_xyz_all:  Optional[jnp.ndarray],   # (nphase, nplanet, 3); None when planet_mask is given
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
    compute_map:         bool = False,
    has_planet:          bool = True,
    k_achromatic:        bool = False,
    has_ar:              bool = True,                   # False: quiet star, no AR terms at all
    nphase:              int,
    geometry:            dict = {},                     # precompute_geometry output (precomputed shapes, mask, column sums)
    planet_mask:         Optional[jnp.ndarray] = None,  # precomputed or per-call from the geometry, over transit_idx phases
) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Light curve at every phase and wavelength via ``_integrate_flux``.

    ``has_planet=False`` skips the occultation mask; ``k_achromatic=True``
    builds it once per phase from ``k[:, 0]`` instead of once per (phase,
    wavelength). Both are static, shape-derived choices. ``has_ar=False``
    skips the AR shapes and spot terms (quiet star). Precomputed arrays in
    ``geometry`` replace the per-call builds; its column sums are used unless
    the map is requested or the AR shapes are per-call, and then the per-pixel
    spectral terms are built only if a per-call planet mask still needs them.

    Returns
    -------
    lc_raw    : (nphase, nwave)
    star_maps : (nphase, n, n), or None when compute_map=False
    """
    ar_shapes = geometry.get("ar_shapes")
    use_columns = "basis_sum" in geometry and not compute_map and (ar_shapes is not None or not has_ar)
    Gq = GD = None
    if use_columns:
        Gq, GD = vmap(
            functools.partial(_column_terms_at_wavelength, wavelength_grid=wavelength,
                              flux_quiet=flux_quiet, flux_active=flux_active, vel_col=vel_col),
            in_axes=(0, 0, 1),
        )(wavelength, ld_coeffs_quiet, ld_coeffs_active)                 # (nwave, K), (nwave, nar, K)
    if use_columns and not (has_planet and "H_M" not in geometry):
        F_quiet_px = D = None                                           # nothing left to do per pixel
    elif not has_ar:
        ar_shapes = D = None
        F_quiet_px = vmap(
            functools.partial(
                _quiet_pixel_flux_at_wavelength,
                wavelength_grid=wavelength, flux_quiet=flux_quiet, mu_disc=mu_disc,
                mu_profile_pts=mu_profile_pts, col_idx=col_idx, vel_col=vel_col, ld_mode=ld_mode,
            )
        )(wavelength, ld_coeffs_quiet, I_profile_quiet)                 # (nwave, P)
    else:
        if ar_shapes is None:
            ar_shapes = _ar_shapes_all(all_ar_carts, arsize_rads, ar_smoothness, x_disc, y_disc, star_pixel_rad)
        F_quiet_px, D = vmap(  # once per wavelength; (nar, nwave, ...) inputs vmap on axis=1
            functools.partial(
                _pixel_flux_at_wavelength,
                wavelength_grid = wavelength,
                flux_quiet      = flux_quiet,
                flux_active     = flux_active,
                mu_disc         = mu_disc,
                mu_profile_pts  = mu_profile_pts,
                col_idx         = col_idx,
                vel_col         = vel_col,
                ld_mode         = ld_mode,
            ),
            in_axes=(0, 0, 1, 0, 1),
        )(wavelength, ld_coeffs_quiet, ld_coeffs_active, I_profile_quiet, I_profile_active)
        # F_quiet_px: (nwave, P);  D: (nwave, nar, P)

    if has_planet and planet_mask is None:
        planet_mask = _planet_mask_all(planet_xyz_all, k, k_achromatic, x_disc, y_disc, star_pixel_rad, transit_softness)

    map_idx = jnp.argmin(jnp.abs(wavelength - plot_map_wavelength)) if compute_map else None
    lc_raw, arted = _integrate_flux(
        ar_shapes, planet_mask, F_quiet_px, D, total_pixels, nphase, geometry,
        compute_map=compute_map, map_idx=map_idx, use_columns=use_columns, Gq=Gq, GD=GD,
    )

    if not compute_map:
        return lc_raw, None

    star_maps = (jnp.zeros((nphase, n * n), dtype=arted.dtype)
                 .at[:, flat_indices].set(arted)
                 .reshape(nphase, n, n))
    return lc_raw, star_maps


# ---------------------------------------------------------------------------
# 6. Single-phase computation -- thin wrapper (one phase of the above)
# ---------------------------------------------------------------------------

def _compute_single_phase(
    ar_cart_all:         jnp.ndarray,  # (nar, 3)
    planet_xyz:          jnp.ndarray,  # (nplanet, 3)
    *,
    k:                   jnp.ndarray,  # (nplanet, nwave) Rp / R*, one value per planet per wavelength
    compute_map:         bool = False,
    **kwargs,
) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    One phase of ``_compute_all_phases`` (same keywords, no phase axis).
    ``k`` is achromatic iff its wavelength axis has size 1. Pass
    ``jnp.array([[0., 0., -1e10]])`` and ``k = 0`` to disable the transit.

    Returns
    -------
    flux_per_wavelength : (nwave,)
    star_map            : (n, n) or None
    """
    lc, maps = _compute_all_phases(
        ar_cart_all[None], planet_xyz[None],
        k=k, compute_map=compute_map, nphase=1,
        has_planet=True, k_achromatic=(k.shape[1] == 1),
        **kwargs,
    )
    return lc[0], (None if maps is None else maps[0])


# ---------------------------------------------------------------------------
# 6b. All-phases computation -- time-varying active-region counterpart.
# ---------------------------------------------------------------------------

def _compute_all_phases_evolving(
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
    compute_map:        bool = False,
) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Time-varying counterpart to ``_compute_all_phases``. Used only when make_lc() is given at least
    one AR parameter with a time axis. Identical to _compute_all_phases except
    ar_lat/ar_long/arsize_rads/ ar_smoothness/flux_active are supplied per-phase and the AR's Cartesian
    position is rebuilt and rotated fresh every phase. This function's existence adds no cost to 
    _compute_all_phases or any caller that doesn't use it.

    ``compute_map=False`` skips the per-phase 2D map reconstruction inside
    ``_compute_single_phase`` to save on computation.

    Returns
    -------
    lc_raw    : (nphase, nwave)
    star_maps : (nphase, n, n), or None when compute_map=False
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
        return _compute_single_phase(
            cart_rot, planet_xyz_ph,
            wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_ph,
            ld_coeffs_quiet=ld_coeffs_quiet, ld_coeffs_active=ld_coeffs_active,
            I_profile_quiet=I_profile_quiet, I_profile_active=I_profile_active,
            mu_profile_pts=mu_profile_pts, x_disc=x_disc, y_disc=y_disc,
            mu_disc=mu_disc, col_idx=col_idx, vel_col=vel_col,
            star_pixel_rad=star_pixel_rad, total_pixels=total_pixels,
            arsize_rads=arsize_ph, ar_smoothness=smooth_ph, k=k, ld_mode=ld_mode,
            plot_map_wavelength=plot_map_wavelength, n=n, flat_indices=flat_indices,
            transit_softness=transit_softness, compute_map=compute_map,
        )

    _phase_vmap = vmap(_single_phase, in_axes=(0, 0, 0, 0, 0, 0, 0))
    return _phase_vmap(ar_lat_all, ar_long_all, arsize_rads_all, ar_smoothness_all,
                        flux_active_all, planet_xyz_all, phases_rot)


# ---------------------------------------------------------------------------
# 7. Public API -- two-stage design
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


def _k_is_achromatic(k, nplanet: int, nwave: int) -> bool:
    """True if ``k`` has no wavelength axis under ``_prepare_transit_k``'s shape rules (shape only; traced-safe)."""
    shape = jnp.shape(k)
    if len(shape) == 0 or nwave == 1:
        return True
    if len(shape) == 1:
        if shape[0] == 1:
            return True
        if nplanet == 1 and shape[0] == nwave:
            return False               # legacy single-planet chromatic depth
        return shape[0] == nplanet
    return False                       # explicit (nplanet, nwave)


def _check_ar_shape_params(ar_size, ar_smoothness, nar):
    """Broadcast a shared ``ar_smoothness`` to ``(nar,)``; validate concrete ``ar_size``/``ar_smoothness``."""
    if ar_smoothness.size == 1:
        ar_smoothness = jnp.broadcast_to(ar_smoothness, (nar,))
    elif ar_smoothness.shape != (nar,):
        raise ValueError(
            f"ar_smoothness shape mismatch: got shape {ar_smoothness.shape} "
            f"but expected a scalar or shape ({nar},)."
        )
    if not isinstance(ar_smoothness, jax.core.Tracer) and bool(jnp.any(ar_smoothness < 1)):
        raise ValueError(
            f"ar_smoothness must be >= 1 (got {ar_smoothness}); 1 is a "
            "Gaussian AR boundary and larger values sharpen it towards a "
            "top-hat, but values below 1 do not correspond to a physically "
            "meaningful AR shape."
        )
    if not isinstance(ar_size, jax.core.Tracer) and bool(jnp.any(ar_size < 0)):
        raise ValueError(
            f"ar_size must be >= 0 (got {ar_size}); ar_size is an "
            "angular radius in degrees and cannot be negative."
        )
    return ar_smoothness


def _ar_carts_all_phases(model, ar_lat, ar_long):
    """Rotated Cartesian AR positions at every (oversampled) phase, (nphase, nar, 3)."""
    spr, inc_star = model["star_pixel_rad"], model["inc_star"]
    ar_lat_rad, ar_long_rad = jnp.deg2rad(ar_lat), jnp.deg2rad(ar_long)
    ar_cart = jnp.stack([
        spr * jnp.sin(ar_long_rad) * jnp.cos(ar_lat_rad),
        spr * jnp.sin(ar_lat_rad),
        spr * jnp.cos(ar_long_rad) * jnp.cos(ar_lat_rad),
    ], axis=-1)   # (nar, 3)
    return vmap(lambda phase_deg: vmap(lambda cart: rotate_active_region(cart, phase_deg, inc_star))(ar_cart))(model["phases_rot"])


def _planet_orbit(model, t0, period, a_over_rstar, inclination, ecc, omega_peri, sp_orb):
    """Planet sky positions (nphase, nplanet, 3) and nplanet from per-call orbital parameters."""
    _defaults  = model["transit_params"]
    ecc_val        = ecc        if ecc        is not None else _defaults.get("ecc", 0.0)
    omega_peri_val = omega_peri if omega_peri is not None else _defaults.get("omega_peri", 0.0)
    sp_orb_deg     = sp_orb     if sp_orb     is not None else _defaults.get("sp_orb", 0.0)
    sp_orb_val     = jnp.deg2rad(sp_orb_deg)   # sp_orb is given in degrees; planet.py takes radians
    planet_xyz_all = compute_multi_planet_sky_positions(
        model["times_oversampled"], t0 - model["t_ref"], period, a_over_rstar, inclination,
        ecc_val, omega_peri_val, sp_orb_val,
    )   # t0 shifted by t_ref like times_oversampled was in build_system
    return planet_xyz_all, jnp.atleast_1d(jnp.asarray(t0)).shape[-1]


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
        model["k_achromatic"] = _k_is_achromatic(k, nplanet, nwave)
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


@jax.tree_util.register_pytree_node_class
class Geometry(dict):
    """``precompute_geometry`` output: a dict pytree whose Python scalars stay static under ``jit``."""

    def tree_flatten(self):
        keys = tuple(sorted(k for k, v in self.items() if isinstance(v, (jnp.ndarray, np.ndarray))))
        static = tuple(sorted((k, v) for k, v in self.items() if k not in keys))
        return tuple(self[k] for k in keys), (keys, static)

    @classmethod
    def tree_unflatten(cls, aux, children):
        keys, static = aux
        return cls({**dict(zip(keys, children)), **dict(static)})


def precompute_geometry(
    model: dict,
    *,
    ar_lat: Optional[np.ndarray] = None,
    ar_long: Optional[np.ndarray] = None,
    ar_size: Optional[np.ndarray] = None,
    ar_smoothness: Optional[np.ndarray] = None,
    t0: Optional[float | np.ndarray] = None,
    period: Optional[float | np.ndarray] = None,
    a_over_rstar: Optional[float | np.ndarray] = None,
    inclination: Optional[float | np.ndarray] = None,
    ecc: Optional[float | np.ndarray] = None,
    omega_peri: Optional[float | np.ndarray] = None,
    sp_orb: Optional[float | np.ndarray] = None,
    k: Optional[float | np.ndarray] = None,
    transit_softness: float = 0.0,
    k_max: float = 0.5,
    columns: bool = True,
) -> dict:
    """
    Precompute the geometry that stays fixed across ``make_lc`` calls, for the
    ``geometry=`` argument of ``make_lc``.

    Call it once, outside ``jit``, with *concrete* values; then pass the
    returned dict to ``make_lc(model, geometry=...)`` as an argument of the
    jitted function (not by closure). Everything it holds is built on the
    (oversampled) phase grid of ``model``:

    - ``ar_lat``/``ar_long``/``ar_size``/``ar_smoothness`` (all or none, static
      shapes only): the AR shapes, so ``make_lc`` then takes only
      ``flux_active`` for the active regions.
    - If ``model`` has a transit: the in-transit phase indices and the planet
      mask over those phases. With no orbital arguments this is the model's
      build-time transit (its ``k`` unless ``k`` is given here). With
      ``t0``/``period``/``a_over_rstar``/``inclination`` (all or none;
      ``ecc``/``omega_peri``/``sp_orb`` optional) and ``k``, that orbit; with
      ``k=None``, only the squared pixel-to-planet distances are stored so
      ``k`` is still passed (and may be traced) per call, with ``k_max``
      bounding it to size the in-transit window. ``transit_softness`` is
      fixed here.

    Memory: ``nphase * nar * total_pixels`` floats for the AR shapes and
    ``nphase_in * nplanet * total_pixels`` (times ``nwave`` for a chromatic
    ``k``) for the planet terms.

    ``columns`` (default True) additionally stores, for the limb-darkening laws
    that are linear in their coefficients (not ``power2``/``intensity_profile``),
    the sums of the precomputed shapes and mask over the pixels of each velocity
    column weighted by each basis function of mu. ``make_lc`` then contracts
    those small arrays instead of the pixel grid, exactly, as long as the
    limb-darkening coefficients keep the intensity non-negative everywhere
    (``make_lc`` clamps negative intensities on the pixel path only). With a
    fixed ``k`` the whole call is then independent of the grid size.

    Returns
    -------
    ``Geometry`` (a dict pytree) with ``ar_shapes``/``ar_shapes_in``/``nar`` and/or ``transit_idx``/
    ``nplanet``/``transit_softness`` plus ``planet_mask_in``/``k``/
    ``k_achromatic`` (fixed ``k``) or ``d2_in``/``z_in`` (free ``k``), and with
    ``columns`` the column sums ``basis_sum``/``H_S``/``H_M``/``H_SM``.
    """
    _args = dict(ar_lat=ar_lat, ar_long=ar_long, ar_size=ar_size, ar_smoothness=ar_smoothness,
                 t0=t0, period=period, a_over_rstar=a_over_rstar, inclination=inclination,
                 ecc=ecc, omega_peri=omega_peri, sp_orb=sp_orb, k=k)
    _traced = [n for n, v in _args.items() if isinstance(v, jax.core.Tracer)]
    if _traced:
        raise ValueError(
            f"precompute_geometry: {_traced} are traced; it takes concrete values only "
            "(call it once outside jit and pass the result to make_lc(geometry=...))."
        )
    x_disc, y_disc, spr = model["x_disc"], model["y_disc"], model["star_pixel_rad"]
    nwave = model["nwave"]
    geometry = Geometry()

    _ar_args = dict(ar_lat=ar_lat, ar_long=ar_long, ar_size=ar_size, ar_smoothness=ar_smoothness)
    _ar_given = [n for n, v in _ar_args.items() if v is not None]
    if _ar_given and len(_ar_given) != len(_ar_args):
        raise ValueError(
            f"precompute_geometry: partial active-region geometry given ({sorted(_ar_given)}); "
            f"provide all of {list(_ar_args)} or none."
        )
    if _ar_given:
        ar_lat, ar_long, ar_size, ar_smoothness = (jnp.atleast_1d(jnp.asarray(v)) for v in (ar_lat, ar_long, ar_size, ar_smoothness))
        if any(v.ndim != 1 for v in (ar_lat, ar_long, ar_size, ar_smoothness)):
            raise ValueError("precompute_geometry: time-varying active-region parameters cannot be precomputed.")
        nar = ar_lat.shape[0]
        ar_smoothness = _check_ar_shape_params(ar_size, ar_smoothness, nar)
        geometry["ar_shapes"] = _ar_shapes_all(_ar_carts_all_phases(model, ar_lat, ar_long), jnp.deg2rad(ar_size),
                                               ar_smoothness, x_disc, y_disc, spr)
        geometry["nar"] = nar

    _orbit_args = dict(t0=t0, period=period, a_over_rstar=a_over_rstar, inclination=inclination)
    _orbit_given = [n for n, v in _orbit_args.items() if v is not None]
    if _orbit_given and len(_orbit_given) != len(_orbit_args):
        raise ValueError(
            f"precompute_geometry: partial orbit given ({sorted(_orbit_given)}); "
            f"provide all of {list(_orbit_args)} or none (the model's build-time orbit)."
        )
    if (_orbit_given or k is not None) and not model.get("has_transit", False):
        raise ValueError("precompute_geometry: transit parameters were given, but this model has no transit attached.")
    if not model.get("has_transit", False):
        return _add_column_sums(model, geometry) if columns else geometry
    if _orbit_given:
        planet_xyz_all, nplanet = _planet_orbit(model, t0, period, a_over_rstar, inclination, ecc, omega_peri, sp_orb)
    else:
        if any(v is not None for v in (ecc, omega_peri, sp_orb)):
            raise ValueError("precompute_geometry: ecc/omega_peri/sp_orb need t0/period/a_over_rstar/inclination too.")
        planet_xyz_all, nplanet = model["planet_xyz"], model["nplanet"]
        k = model["k"] if k is None else k   # the model's static transit as-is, like make_lc without orbit arguments
    xyz = np.asarray(planet_xyz_all)                                  # (nphase, nplanet, 3)
    k_bound = float(np.max(np.asarray(k))) if k is not None else float(k_max)
    window = 1.0 + k_bound + 8.0 * float(transit_softness)             # sigmoid tail < 3e-4 beyond it
    in_transit = (xyz[..., 2] > 0.0) & (np.hypot(xyz[..., 0], xyz[..., 1]) < window)
    idx = np.flatnonzero(np.any(in_transit, axis=1)).astype(np.int32)
    geometry.update(transit_idx=idx, nplanet=nplanet, transit_softness=float(transit_softness))
    if "ar_shapes" in geometry:
        geometry["ar_shapes_in"] = geometry["ar_shapes"][idx]
    xyz_in = planet_xyz_all[idx]                                       # (nphase_in, nplanet, 3)
    if k is not None:
        k_achromatic = _k_is_achromatic(k, nplanet, nwave)
        k_dense = _prepare_transit_k(k, nplanet, nwave)
        geometry["planet_mask_in"] = _planet_mask_all(xyz_in, k_dense, k_achromatic, x_disc, y_disc, spr, float(transit_softness))
        geometry.update(k=k_dense, k_achromatic=k_achromatic)
    else:
        xn, yn = x_disc / spr, y_disc / spr
        geometry["d2_in"] = (xn - xyz_in[:, :, 0, None]) ** 2 + (yn - xyz_in[:, :, 1, None]) ** 2  # (nphase_in, nplanet, P)
        geometry["z_in"] = xyz_in[:, :, 2]
    return _add_column_sums(model, geometry) if columns else geometry


def _add_column_sums(model, geometry):
    """Column x LD-basis sums of the precomputed shapes / mask (see precompute_geometry); no-op for other laws."""
    if model["ld_mode"] not in _LD_BASIS_LAWS:
        return geometry
    B, col, n = _ld_basis(np.asarray(model["mu_disc"], np.float32), model["ld_mode"]), model["col_idx"], model["n"]
    sums = lambda A: _column_sums(A, B, col, n)
    geometry["basis_sum"] = sums(jnp.ones(model["total_pixels"], jnp.float32))
    if "ar_shapes" in geometry:
        geometry["H_S"] = sums(geometry["ar_shapes"])
    if "planet_mask_in" in geometry:
        M = geometry["planet_mask_in"]
        geometry["H_M"] = sums(M)
        if "ar_shapes_in" in geometry:
            S_in = geometry["ar_shapes_in"]
            geometry["H_SM"] = sums(S_in * M[:, None, :] if M.ndim == 2 else S_in[:, None, :, :] * M[:, :, None, :])
    return geometry


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
    compute_map: bool = False,
    geometry: Optional[Geometry] = None,
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
        regardless of the values passed in above. Set this > 0 to get a
        smooth, non-zero gradient for gradient-based retrieval of those
        parameters. Use ``default_transit_softness`` for the value.
    compute_map : bool, optional
        Default False. The per-phase ``(n, n)`` stellar pixel map
        (``star_maps`` below) is useful for plotting/diagnostics but constructing
        it roughly doubles this function's wall-clock cost. Thus, it is set to False
        by default. Users can set it to True when they want the map.
    geometry : dict, optional
        Output of ``precompute_geometry`` for the parts of the geometry that
        stay fixed across calls. With precomputed AR shapes, give
        ``flux_active`` only (no ``ar_lat``/``ar_long``/``ar_size``/
        ``ar_smoothness``); with a precomputed transit, give no orbital
        parameters, and ``k`` only if it was left free there.
        ``transit_softness`` then comes from ``precompute_geometry``.
        Time-varying active regions are not supported with ``geometry``.
        Column sums in the geometry (see ``precompute_geometry``) are used
        unless ``compute_map`` is set.

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
    geometry = geometry or {}
    geo_ar, geo_planet = "ar_shapes" in geometry, "transit_idx" in geometry

    # ---- AR parameters: all-or-nothing ------------------------------------
    _ar_args = dict(flux_active=flux_active, ar_lat=ar_lat, ar_long=ar_long,
                     ar_size=ar_size, ar_smoothness=ar_smoothness)
    _ar_given = {name: v for name, v in _ar_args.items() if v is not None}
    if geo_ar and set(_ar_given) != {"flux_active"}:
        raise ValueError(
            "make_lc: with precomputed AR shapes in `geometry`, give flux_active and "
            f"none of ar_lat/ar_long/ar_size/ar_smoothness (got {sorted(_ar_given)})."
        )
    if not geo_ar and _ar_given and len(_ar_given) != len(_ar_args):
        _missing = [name for name in _ar_args if name not in _ar_given]
        raise ValueError(
            "make_lc: partial active-region parameters given "
            f"({sorted(_ar_given)}); missing {_missing}. Provide all of "
            f"{list(_ar_args)} to add active region(s), or none for a quiet star."
        )
    has_ar = bool(_ar_given)   # a quiet star skips every AR term; the dummy AR below only feeds the shape checks
    if not _ar_given:
        flux_active   = model["flux_quiet"]
        ar_lat        = jnp.array([0.0])
        ar_long       = jnp.array([0.0])
        ar_size       = jnp.array([1.0])
        ar_smoothness = jnp.array([1.0])

    flux_active   = jnp.atleast_1d(jnp.asarray(flux_active))
    if geo_ar:
        nar = geometry["nar"]
        if flux_active.ndim == 3:
            raise ValueError("make_lc: time-varying flux_active cannot be combined with `geometry`.")
    else:
        ar_lat        = jnp.atleast_1d(jnp.asarray(ar_lat))
        ar_long       = jnp.atleast_1d(jnp.asarray(ar_long))
        ar_size       = jnp.atleast_1d(jnp.asarray(ar_size))
        ar_smoothness = jnp.atleast_1d(jnp.asarray(ar_smoothness))
        nar = ar_lat.shape[-1]   # trailing axis, whether or not a leading time axis is present
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
    ar_time_varying = not geo_ar and (
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

        if not geo_ar:
            ar_smoothness = _check_ar_shape_params(ar_size, ar_smoothness, nar)
        if not isinstance(flux_active, jax.core.Tracer) and bool(jnp.any(flux_active < 0)):
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

        if not isinstance(ar_size, jax.core.Tracer) and bool(jnp.any(ar_size < 0)):
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
        if not isinstance(ar_smoothness, jax.core.Tracer) and bool(jnp.any(ar_smoothness < 1)):
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
        if not isinstance(flux_active, jax.core.Tracer) and bool(jnp.any(flux_active < 0)):
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
        if model.get("verbose", False) and not isinstance(ar_size, jax.core.Tracer):
            n_size_clipped  = int(jnp.sum(ar_size < 0))
            n_smooth_clipped = int(jnp.sum(ar_smoothness < 1))
            n_flux_clipped  = int(jnp.sum(flux_active < 0))
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

    all_ar_carts = None if (geo_ar or ar_time_varying or not has_ar) else _ar_carts_all_phases(model, ar_lat, ar_long)

    # ---- Transit parameters: all-or-nothing (required 5), ecc/omega_peri
    # only meaningful alongside them -------------------------------------
    _transit_required = dict(t0=t0, period=period, a_over_rstar=a_over_rstar,
                               inclination=inclination, k=k)
    _transit_optional  = dict(ecc=ecc, omega_peri=omega_peri, sp_orb=sp_orb)
    _transit_given = {name: v for name, v in {**_transit_required, **_transit_optional}.items()
                       if v is not None}
    if geo_planet:
        if ar_time_varying:
            raise ValueError("make_lc: time-varying active regions cannot be combined with `geometry`.")
        if set(_transit_given) - {"k"}:
            raise ValueError(
                "make_lc: the orbit is fixed by `geometry`; give no "
                f"t0/period/a_over_rstar/inclination/ecc/omega_peri/sp_orb (got {sorted(_transit_given)})."
            )
        if ("planet_mask_in" in geometry) == (k is not None):
            raise ValueError("make_lc: give k exactly when precompute_geometry was called without it.")
        if transit_softness not in (0.0, geometry["transit_softness"]):
            raise ValueError("make_lc: transit_softness is fixed by `geometry`; pass the same value or omit it.")
        _transit_given = {}
    elif _transit_given:
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
    planet_mask = None
    if geo_planet:
        planet_xyz_all   = None
        nplanet          = geometry["nplanet"]
        transit_softness = geometry["transit_softness"]
        has_planet       = True
        if k is None:
            k_val, k_achromatic, planet_mask = geometry["k"], geometry["k_achromatic"], geometry["planet_mask_in"]
        else:
            k_val        = _prepare_transit_k(k, nplanet, nwave)
            k_achromatic = _k_is_achromatic(k, nplanet, nwave)
            planet_mask  = _planet_mask_from_geometry(geometry, k_val, k_achromatic, transit_softness)
    elif _transit_given:
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
        planet_xyz_all, nplanet = _planet_orbit(model, t0, period, a_over_rstar, inclination, ecc, omega_peri, sp_orb)
        k_val = k
        has_planet   = True
        k_achromatic = _k_is_achromatic(k, nplanet, nwave)
    elif model.get("has_transit", False):
        # Static, backwards-compatible path: positions baked at build time.
        planet_xyz_all = model["planet_xyz"]    # (nphase_compute, nplanet, 3)
        k_val          = model["k"]             # (nplanet, nwave), already dense
        nplanet        = model["nplanet"]
        has_planet     = True
        k_achromatic   = model.get("k_achromatic", False)
    else:
        # Dummy: planet permanently behind the star, zero-radius disc (time-varying path only).
        nphase_compute = model["phases_rot"].shape[0]
        nplanet        = 1
        planet_xyz_all = jnp.zeros((nphase_compute, 1, 3)).at[:, :, 2].set(-1e10)
        k_val          = 0.0
        transit_softness = 0.0
        has_planet     = False
        k_achromatic   = True

    # ---- Broadcast k to (nplanet, nwave): see _prepare_transit_k for the
    # full shape convention. A scalar means the same (achromatic) radius
    # ratio at every planet/wavelength; a genuinely chromatic and/or
    # per-planet depth is supported (the occultation mask is computed per
    # wavelength in _compute_all_phases unless _k_is_achromatic).
    # Idempotent (a no-op) on the already-dense static-path value above.
    k_val = _prepare_transit_k(k_val, nplanet, nwave)

    # ---- All-phases computation ------------------------------------------
    if not ar_time_varying:
        lc_raw, star_maps = _compute_all_phases(
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
            arsize_rads         = None if (geo_ar or not has_ar) else jnp.deg2rad(ar_size),
            ar_smoothness       = ar_smoothness,
            k                   = k_val,
            ld_mode             = model["ld_mode"],
            plot_map_wavelength = model["plot_map_wavelength"],
            n                   = model["n"],
            flat_indices        = model["flat_indices"],
            transit_softness    = transit_softness,
            compute_map         = compute_map,
            has_planet          = has_planet,
            k_achromatic        = k_achromatic,
            has_ar              = has_ar,
            nphase              = model["phases_rot"].shape[0],
            geometry            = geometry,
            planet_mask         = planet_mask,
        )
    else:
        lc_raw, star_maps = _compute_all_phases_evolving(
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
            compute_map         = compute_map,
        )

    # ---- Oversample averaging --------------------------------------------
    if oversample > 1:
        # lc_raw: (nphase_compute, nwave) -> (nphase_original, oversample, nwave) -> mean
        lc_raw = lc_raw.reshape(nphase_original, oversample, nwave).mean(axis=1)

        # star_maps: take only the first sub-exposure per original phase
        # (averaging 2D maps is expensive and rarely useful). None when
        # compute_map=False -- nothing to subsample.
        if star_maps is not None:
            star_maps = star_maps[::oversample]

    # ---- Single-wavelength convenience: drop the now-degenerate nwave
    # axis so single-channel callers get (nphase,) instead of (nphase, 1) ---
    if nwave == 1:
        lc_raw = lc_raw[..., 0]

    return lc_raw, star_maps


def default_transit_softness(model: dict, k: Optional[float] = None) -> float:
    """
    Recommended ``transit_softness`` for a model: half a pixel,
    ``0.5 / model["star_pixel_rad"]``.

    Parameters
    ----------
    model : dict
        As returned by ``build_system``.
    k : float or array_like, optional
        Approximate planet-to-star radius ratio. Only used to warn when the
        smallest planet spans too few pixels for the grid to resolve.

    Returns
    -------
    float
        ``transit_softness`` in units of R*.
    """
    star_pixel_rad = float(model["star_pixel_rad"])
    if k is not None:
        k_min = float(np.min(np.asarray(k, dtype=float)))
        if k_min * star_pixel_rad < _MIN_RESOLVED_PLANET_PX:
            warnings.warn(
                f"default_transit_softness: the planet spans only "
                f"{2.0 * k_min * star_pixel_rad:.1f} pixels (k={k_min:g}, "
                f"stellar_grid_size={star_pixel_rad:g}), so the pixel grid itself "
                f"-- not transit_softness -- dominates the transit depth error. "
                f"Raise stellar_grid_size for an accurate depth.",
                UserWarning, stacklevel=2,
            )
    return 0.5 / star_pixel_rad


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
    compute_map: bool = True,
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
    compute_map : bool, optional
        Default True (unlike ``make_lc``'s own default of False) -- a
        one-off call like this one is commonly used to plot/inspect the
        stellar map, so the map is returned by default here. Set False for
        a light-curve-only call if you don't need it.

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
    ``star_maps`` - (ntimes, n, n) stellar flux map per phase, or None
                    when compute_map=False
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
        compute_map=compute_map,
    )
    return np.array(lc), (None if star_maps is None else np.array(star_maps))

