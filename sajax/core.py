"""
core.py -- JAX-accelerated stellar active region light-curve engine.

This module is a complete rewrite of ``SAGE1/sage.py`` in JAX.

Key differences from the original NumPy/SciPy implementation
-------------------------------------------------------------
1. **No wavelength loop.**
   The original code iterated over wavelengths with a Python ``for`` loop.
   Here the entire spectral axis is handled by ``jax.vmap``, which maps
   the single-channel computation across all wavelengths in parallel.

2. **No phase loop.**
   The original code iterated over rotational phases with a Python loop.
   Here all phases are computed in a single ``jax.vmap`` call -- this is
   the main source of speedup over the original code.

3. **Contrast-surface active region model.**
   Each active region's contrast is described by a super-Gaussian function
   which can be used to control the region's smoothness. For a given pixel
   p, the flux at that pixel is given by:

       F_p/F_quiet = 1 - sum_a (1 - C_a) * exp(-(x_a/r_a)^(2*n_a))

   where ``x_a`` is the angular distance from active region ``a``'s centre,
   ``r_a``/``n_a`` are its angular radius and super-Gaussian order, and
   ``C_a = F_a/F_quiet`` is its spectral contrast. Components are summed,
   not selected by a winner-take-all rule, so overlapping active regions
   (e.g. an umbra sitting inside a penumbra) contribute simultaneously--
   the combined dip can be deeper than either component's own contrast.
   The super-gaussian goes from n_a=1 to inf, ranging from a pure gaussian
   curve, to a top-hat function.

4. **Per-active-region spectra and limb darkening.**
   Each active region carries its own spectrum and its own limb-darkening
   coefficients (same law as the quiet photosphere, different values),
   independent of every other active region and of the quiet photosphere.
   The contrast value used for the computation of F_p at each pixel and 
   wavelength is extracted directly from each active region's ratio of 
   F_a to F_quiet.

5. **Rotational broadening applied at the spectral level.**
   Rather than a first-order ``(1 + v/c)`` intensity scaling, each pixel's
   local radial velocity Doppler-shifts the spectrum itself before the
   contrast at the requested wavelength bin is extracted -- i.e. the
   spectrum is resampled at ``lambda * (1 - v/c)`` for that pixel's own
   velocity v. Because the stellar rotation axis is the y-axis in Carthesian
   coordinates, the velocity depends only on a pixel's y-coordinate, so all
   pixels in the same grid row share one velocity -- the (expensive) spectral
   resampling is done once per row (``n`` rows) rather than once per pixel
   (``~n^2``), then broadcast back out to pixels.

6. **No scatter-index active region placement.**
   The original code located active region pixels via integer scatter indices
   (fancy indexing with ``.astype(int)``), which is not differentiable
   and incompatible with ``jit``.  SAJAX instead computes an analytic
   angular-distance shape over the full pixel arrays using ``jnp.where``.

7. **No class state mutation.**
   The original ``sage_class.rotate_star()`` mutated ``self.phases_rot``
   inside a loop -- a latent bug.  SAJAX uses pure functions throughout.

8. **No astropy dependency for geometry.**
   Rotation matrices are implemented directly in JAX (see geometry.py).

9. **No transit-geometry parameters.**
   The original SAGE grid was sized using ``planet_pixel_size``,
   ``radiusratio``, and ``semimajor`` -- artifacts of its transit-fitting
   origin.  SAJAX replaces these with a single ``stellar_grid_size``
   parameter: the stellar radius in pixels.  No planet required.

10. **Pre-masked grid.**
    ``build_stellar_grid`` applies the stellar disc mask immediately and
    returns 1D arrays containing only the in-disc pixels.  No starmask is
    ever passed to JAX functions -- the mask is implicit in the data shape.
    The only 2D reconstruction happens at output time for ``star_maps``,
    using stored flat indices.

11. **Differentiable end-to-end.**
    All operations are JAX-native, so ``jax.grad`` / ``jax.jacobian``
    work on the full pipeline -- useful for gradient-based retrieval.

12. **Phase oversampling.**
    Real observations integrate photons over a finite exposure time.
    When an active region crosses the stellar limb, the discrete pixel
    grid can produce sharp discontinuities in the light curve.
    The ``oversample`` parameter (default 1, i.e. off) spreads each
    requested phase into multiple sub-exposures and averages the result,
    mimicking finite-exposure integration and smoothing limb-crossing
    artefacts.

JIT compilation
---------------
*Do NOT jit(evaluate_light_curve) directly* -- it contains Python-level
control flow on model metadata.  Instead, the inner _compute_all_phases
is the hot path and is safe to JIT via:

    from jax import jit
    _compute_all_phases_jit = jit(_compute_all_phases, static_argnames=[
        "star_pixel_rad", "total_pixels", "ldc_mode",
        "plot_map_wavelength", "n",
    ])
"""

from __future__ import annotations

import functools
from typing import Literal, Optional

import numpy as np
import jax.numpy as jnp
from jax import vmap

from .geometry import rotate_active_region
from .planet import _compute_planet_mask

# Type alias
LdcMode = Literal[
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


# ---------------------------------------------------------------------------
# 1a. Grid construction  (NumPy -- runs once per model configuration)
# ---------------------------------------------------------------------------

def build_stellar_grid(
    stellar_grid_size: int,
    ve: float,
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
        Stellar equatorial velocity [km/s].

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
    ``row_idx``       - (total_pixels,) int, 0..n-1 -- which grid row each
                        in-disc pixel belongs to.
    ``vel_row``       - (n,) Doppler factor Δv/c for each possible row.
                        The stellar rotation axis is the y-axis, so
                        rotational velocity depends only on a pixel's row
                        (its y-coordinate) -- every pixel in a row shares
                        one velocity, so it is never materialised per-pixel.
    """
    C = 299_792.458  # speed of light [km/s]

    star_pixel_rad = float(stellar_grid_size)

    # n = 2 * radius + 1 so the centre falls on a pixel (forces odd grid)
    n = 2 * int(stellar_grid_size) + 1

    coords = np.arange(n) - n // 2   # e.g. -R, ..., -1, 0, 1, ..., R
    xg, yg = np.meshgrid(coords, coords)   # (n, n) each

    r2     = xg ** 2 + yg ** 2
    starmask = r2 <= star_pixel_rad ** 2

    # Apply mask → 1D in-disc arrays
    flat_indices = np.flatnonzero(starmask)   # (total_pixels,)
    x_disc = xg.ravel()[flat_indices].astype(np.float32)
    y_disc = yg.ravel()[flat_indices].astype(np.float32)
    r_disc = np.sqrt(r2.ravel()[flat_indices]).astype(np.float32)

    # mu = cos θ = sqrt(1 - (r/R)²), clamped for float32 safety
    mu_disc = np.sqrt(
        np.clip(1.0 - (r_disc / star_pixel_rad) ** 2, 0.0, 1.0)
    ).astype(np.float32)

    # Row index: which of the n possible y-coordinates this pixel has.
    # coords[0] = -n//2, so row_idx = y + n//2 maps y in {-n//2, ..., n//2}
    # onto {0, ..., n-1}.
    row_idx = (y_disc + n // 2).astype(np.int32)

    # Per-row Doppler velocity factor:  Δv/c = (y / R_star) * (ve / c).
    # y increases upward → redshift on the receding limb. Computed once per
    # possible row (n values) rather than once per pixel: every pixel in a
    # row shares this velocity, since the rotation axis is the y-axis.
    vel_row = (coords / star_pixel_rad * (ve / C)).astype(np.float32)

    return dict(
        n             = n,
        star_pixel_rad= star_pixel_rad,
        total_pixels  = int(flat_indices.size),
        flat_indices  = flat_indices,          # kept in NumPy for scatter
        x             = x_disc,
        y             = y_disc,
        mu            = mu_disc,
        row_idx       = row_idx,
        vel_row       = vel_row,
    )


# ---------------------------------------------------------------------------
# 1b. Phase oversampling  (NumPy -- runs once in build_model)
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

    # Broadcast: (nphase, 1) + (1, oversample) → (nphase, oversample)
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
    of order ``ar_smoothness``:

      - ``ar_smoothness -> inf``  converges to a top-hat function.
      - ``ar_smoothness == 1``    is a Gaussian.

    This shape peaks at exactly 1 at the AR's centre and its amplitude within
    the light-curve formula is set separately by the AR's spectral contrast (see
    ``_flux_at_wavelength``). Thus, this function is purely geometric and
    carries no wavelength dependence.

    Uses the exact spherical "distance" variable ``x = 1 - cos(theta)``, with theta 
    the great circle distance, so this *holds even for large active regions*.

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

    # Cosine of great-circle distance via dot product on the unit sphere.
    cos_theta = (spx * x_disc + spy * y_disc + spz * z_disc) / (star_pixel_rad ** 2)

    # Exact spherical "distance" variable.
    x  = 1.0 - cos_theta
    x0 = jnp.maximum(1.0 - jnp.cos(arsize_rad), _AR_SHAPE_TINY)

    exponent = 2.0 * ar_smoothness

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
    ldc_coeffs_wl:  jnp.ndarray, # (n_coeffs,) one row for this wavelength
    I_prof_wl:      jnp.ndarray, # (n_mu_pts,) used only for "intensity_profile"
    mu_profile_pts: jnp.ndarray, # (n_mu_pts,) set of mu points of the user-provided intensity profile. used only for "intensity_profile"
    ldc_mode:       LdcMode,     # limb-darkening law to use
) -> jnp.ndarray:
    """
    Evaluate the limb-darkening law at each pixel for one wavelength bin.

    The same function is used for the quiet photosphere and for every
    active region -- each caller supplies its own ``ldc_coeffs_wl`` (and,
    for ``ldc_mode="intensity_profile"``, its own ``I_prof_wl``), but the
    functional law itself (``ldc_mode``) is shared by the whole star.

    Returns
    -------
    jnp.ndarray, shape (total_pixels,)
    """
    if ldc_mode == "intensity_profile":
        # Interpolate a user-supplied I(mu) profile.
        result = jnp.interp(mu_disc, mu_profile_pts, I_prof_wl,
                             left=0.0, right=0.0)
    elif ldc_mode == "linear":
        # I(μ) = 1 - u*(1 - μ)
        result = 1.0 - ldc_coeffs_wl[0] * (1.0 - mu_disc)
    elif ldc_mode == "quadratic":
        # I(μ) = 1 - u1*(1-μ) - u2*(1-μ)^2
        result = (1.0
                  - ldc_coeffs_wl[0] * (1.0 - mu_disc)
                  - ldc_coeffs_wl[1] * (1.0 - mu_disc) ** 2)
    elif ldc_mode == "power2":
        # I(μ) = 1 - a*(1 - μ^b)
        result = 1.0 - ldc_coeffs_wl[0] * (1.0 - mu_disc ** ldc_coeffs_wl[1])
    elif ldc_mode == "kipping3":
        # I(μ) = 1 - c1*(1-μ^0.5) - c2*(1-μ) - c3*(1-μ^(3/2))
        result = (1.0
                  - ldc_coeffs_wl[0] * (1.0 - mu_disc ** 0.5)
                  - ldc_coeffs_wl[1] * (1.0 - mu_disc)
                  - ldc_coeffs_wl[2] * (1.0 - mu_disc ** 1.5))
    else:  # "nonlinear4"  -- Claret (2000) four-parameter law
        # I(μ) = 1 - Σ_{k=1}^{4} c_k*(1 - μ^(k/2))
        result = (1.0
                  - ldc_coeffs_wl[0] * (1.0 - mu_disc ** 0.5)
                  - ldc_coeffs_wl[1] * (1.0 - mu_disc)
                  - ldc_coeffs_wl[2] * (1.0 - mu_disc ** 1.5)
                  - ldc_coeffs_wl[3] * (1.0 - mu_disc ** 2.0))

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
    ldc_coeffs_quiet_wl:  jnp.ndarray, # (n_coeffs,)
    ldc_coeffs_active_wl: jnp.ndarray, # (nar, n_coeffs)
    I_prof_quiet_wl:      jnp.ndarray, # (n_mu_pts,)
    I_prof_active_wl:     jnp.ndarray, # (nar, n_mu_pts)
    # --- broadcast: shared across wavelengths ---
    wavelength_grid: jnp.ndarray, # (nwave,) full spectral axis
    flux_quiet:      jnp.ndarray, # (nwave,) full quiet-photosphere spectrum
    flux_active:     jnp.ndarray, # (nar, nwave) full per-AR spectra
    mu_disc:         jnp.ndarray, # (total_pixels,) grid of in-disc mu values
    total_pixels:    int,
    ar_shapes:       jnp.ndarray, # (nar, total_pixels)
    planet_mask:     jnp.ndarray, # (total_pixels,)
    mu_profile_pts:  jnp.ndarray, # (n_mu_pts,)
    row_idx:         jnp.ndarray, # (total_pixels,) int
    vel_row:         jnp.ndarray, # (n,)
    ldc_mode:        LdcMode,
) -> tuple[float, float, jnp.ndarray]:
    """
    Compute disc-integrated flux for a single wavelength channel.

    Builds the dimensionless "contrast surface"::

        F_p/F_quiet = 1 - sum_a (1 - C_a) * ar_shapes[a]

    where ``C_a = F_a / F_quiet`` is active region ``a``'s spectral
    contrast, evaluated at each pixel's own Doppler-shifted wavelength and
    with its own limb-darkening law. Overlapping active regions sum, so
    e.g. an umbra sitting inside a penumbra contributes simultaneously (the
    combined dip exceeds either component's own contrast). The planet mask
    zeroes out occulted pixels.

    All arrays are 1D (in-disc pixels only) - no starmask needed.
    ``planet_mask`` is True for pixels occulted by the planet; those pixels
    contribute zero flux regardless of active-region status.

    Returns
    -------
    star_spec  : float            - un-active-region'ed integrated flux
    total_flux : float            - active-region'ed integrated flux
    arted_flux : (total_pixels,)  - per-pixel flux values (for map output)
    """
    # ---- Per-row Doppler-shifted spectral lookup -------------------------
    # The rotation axis is the y-axis, so velocity depends only on grid row
    # (see build_stellar_grid): resample each spectrum once per row (n
    # values), not once per pixel, then broadcast out via row_idx.
    query_wavelength_row = wavelength_target * (1.0 - vel_row)  # (n,)

    F_quiet_row   = jnp.interp(query_wavelength_row, wavelength_grid, flux_quiet)  # (n,)
    F_quiet_local = F_quiet_row[row_idx]                                          # (total_pixels,)

    F_active_row = vmap(
        lambda spec: jnp.interp(query_wavelength_row, wavelength_grid, spec)
    )(flux_active)                                   # (nar, n)
    F_active_local = F_active_row[:, row_idx]        # (nar, total_pixels)

    # ---- Limb darkening (own law coefficients per AR and for quiet) -----
    ldc_quiet = _evaluate_ldc(
        mu_disc, ldc_coeffs_quiet_wl, I_prof_quiet_wl, mu_profile_pts, ldc_mode,
    )  # (total_pixels,)

    ldc_active = vmap(
        lambda coeffs, iprof: _evaluate_ldc(
            mu_disc, coeffs, iprof, mu_profile_pts, ldc_mode,
        )
    )(ldc_coeffs_active_wl, I_prof_active_wl)  # (nar, total_pixels)

    # ---- Physical flux (Doppler-shifted spectrum x limb darkening) ------
    F_quiet_px  = F_quiet_local * ldc_quiet    # (total_pixels,)
    F_active_px = F_active_local * ldc_active  # (nar, total_pixels)

    star_spec = jnp.sum(F_quiet_px) / total_pixels

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
    return star_spec, total_flux, arted_flux


# ---------------------------------------------------------------------------
# 5. Single-phase computation
# ---------------------------------------------------------------------------

def _compute_single_phase(
    ar_cart_all:         jnp.ndarray,  # (nar, 3)
    planet_xyz:          jnp.ndarray,  # (3,)
    *,
    wavelength:          jnp.ndarray,  # (nwave,)
    flux_quiet:          jnp.ndarray,  # (nwave,)
    flux_active:         jnp.ndarray,  # (nar, nwave)
    ldc_coeffs_quiet:    jnp.ndarray,  # (nwave, n_coeffs)
    ldc_coeffs_active:   jnp.ndarray,  # (nar, nwave, n_coeffs)
    I_profile_quiet:     jnp.ndarray,  # (nwave, n_mu_pts)
    I_profile_active:    jnp.ndarray,  # (nar, nwave, n_mu_pts)
    mu_profile_pts:      jnp.ndarray,  # (n_mu_pts,)
    x_disc:              jnp.ndarray,  # (total_pixels,)
    y_disc:              jnp.ndarray,  # (total_pixels,)
    mu_disc:             jnp.ndarray,  # (total_pixels,)
    row_idx:             jnp.ndarray,  # (total_pixels,)
    vel_row:             jnp.ndarray,  # (n_grid,)
    star_pixel_rad:      float,
    total_pixels:        int,
    arsize_rads:         jnp.ndarray,  # (nar,)
    ar_smoothness:       jnp.ndarray,  # (nar,)
    k:                   float,        # Rp / R*
    ldc_mode:            LdcMode,
    plot_map_wavelength: float,
    n:                   int,         # full grid side (for map scatter)
    flat_indices:        jnp.ndarray, # (total_pixels,) scatter indices
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full spectral computation for one rotational phase, including optional
    pixel-level planet occultation.
    ``planet_xyz`` is the planet's sky-plane position (X, Y, Z) in stellar
    radii.  Pass ``jnp.array([0., 0., -1e10])`` and ``k=0.0`` to disable
    the transit mask (no performance cost -- the mask is all-False).

    Returns
    -------
    flux_per_wavelength   : (nwave,)  normalised flux at each wavelength bin
    contamination_factor  : (nwave,)
    star_map              : (n, n)  flux map at plot_map_wavelength
    """
    # ---- active region shapes: (nar, total_pixels) -----------------------
    ar_shapes = vmap(
        lambda cart, sr, sm: _compute_ar_shape(
            x_disc, y_disc, star_pixel_rad,
            cart[0], cart[1], cart[2], sr, sm,
        )
    )(ar_cart_all, arsize_rads, ar_smoothness)

    # ---- Planet mask: (total_pixels,)  ----------------------------------
    planet_mask = _compute_planet_mask(
        x_disc, y_disc, star_pixel_rad,
        planet_xyz[0], planet_xyz[1], planet_xyz[2], k,
    )

    # ---- vmap over wavelengths ----
    # ldc_coeffs_active/I_profile_active have the wavelength axis second
    # (nar, nwave, ...), so they vmap on axis=1; everything else vmaps on
    # its leading (wavelength) axis.
    _flux_vmap = vmap(
        functools.partial(
            _flux_at_wavelength,
            wavelength_grid = wavelength,
            flux_quiet      = flux_quiet,
            flux_active     = flux_active,
            mu_disc         = mu_disc,
            total_pixels    = total_pixels,
            ar_shapes       = ar_shapes,
            planet_mask     = planet_mask,
            mu_profile_pts  = mu_profile_pts,
            row_idx         = row_idx,
            vel_row         = vel_row,
            ldc_mode        = ldc_mode,
        ),
        in_axes=(0, 0, 1, 0, 1),
    )

    star_specs, bin_fluxes, flux_discs = _flux_vmap(
        wavelength,
        ldc_coeffs_quiet,
        ldc_coeffs_active,
        I_profile_quiet,
        I_profile_active,
    )

    # ---- Per-wavelength normalised flux and contamination factor --------
    flux_per_wavelength   = bin_fluxes / jnp.where(
        star_specs == 0.0, jnp.nan, star_specs
    )
    contamination_factor = star_specs / jnp.where(
        bin_fluxes == 0.0, jnp.nan, bin_fluxes
    )

    # ---- Reconstruct 2D map at plot_map_wavelength ----------------------
    map_idx   = jnp.argmin(jnp.abs(wavelength - plot_map_wavelength))
    flux_1d   = flux_discs[map_idx]   # (total_pixels,)
    star_map  = jnp.zeros(n * n).at[flat_indices].set(flux_1d).reshape(n, n)

    return flux_per_wavelength, contamination_factor, star_map


# ---------------------------------------------------------------------------
# 6. All-phases computation -- vmapped over the phase axis
# ---------------------------------------------------------------------------

def _compute_all_phases(
    all_ar_carts:    jnp.ndarray,   # (nphase, nar, 3)
    planet_xyz_all:  jnp.ndarray,   # (nphase, 3)
    *,
    wavelength:          jnp.ndarray,
    flux_quiet:          jnp.ndarray,
    flux_active:         jnp.ndarray,
    ldc_coeffs_quiet:    jnp.ndarray, # (nwave, n_coeffs)
    ldc_coeffs_active:   jnp.ndarray, # (nar, nwave, n_coeffs)
    I_profile_quiet:     jnp.ndarray,
    I_profile_active:    jnp.ndarray,
    mu_profile_pts:      jnp.ndarray,
    x_disc:              jnp.ndarray,
    y_disc:              jnp.ndarray,
    mu_disc:             jnp.ndarray,
    row_idx:             jnp.ndarray,
    vel_row:             jnp.ndarray,
    star_pixel_rad:      float,
    total_pixels:        int,
    arsize_rads:         jnp.ndarray,
    ar_smoothness:       jnp.ndarray,
    k:                   float,       # Rp / R★
    ldc_mode:            LdcMode,
    plot_map_wavelength: float,
    n:                   int,
    flat_indices:        jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    vmap ``_compute_single_phase`` over the phase axis.

    ``planet_xyz_all`` contains the planet (X, Y, Z) position at each
    (oversampled) phase.  Pass ``jnp.full((nphase, 3), [0, 0, -1e10])``
    and ``k=0.0`` to disable transit (no performance overhead).

    Returns
    -------
    lc_raw    : (nphase, nwave)
    epsilon   : (nphase, nwave)
    star_maps : (nphase, n, n)
    """
    _phase_vmap = vmap(
        functools.partial(
            _compute_single_phase,
            wavelength          = wavelength,
            flux_quiet          = flux_quiet,
            flux_active         = flux_active,
            ldc_coeffs_quiet    = ldc_coeffs_quiet,
            ldc_coeffs_active   = ldc_coeffs_active,
            I_profile_quiet     = I_profile_quiet,
            I_profile_active    = I_profile_active,
            mu_profile_pts      = mu_profile_pts,
            x_disc              = x_disc,
            y_disc              = y_disc,
            mu_disc             = mu_disc,
            row_idx             = row_idx,
            vel_row             = vel_row,
            star_pixel_rad      = star_pixel_rad,
            total_pixels        = total_pixels,
            arsize_rads         = arsize_rads,
            ar_smoothness       = ar_smoothness,
            k                   = k,
            ldc_mode            = ldc_mode,
            plot_map_wavelength = plot_map_wavelength,
            n                   = n,
            flat_indices        = flat_indices,
        ),
        in_axes=(0,0), # vmap over both ar_carts and planet_xyz
    )
    return _phase_vmap(all_ar_carts, planet_xyz_all)


# ---------------------------------------------------------------------------
# 7. Public API -- two-stage design
#
#   Stage 1:  build_model()         - NumPy, call once before sampling.
#                                     Pre-builds the grid and all static
#                                     arrays that are fixed across MCMC steps.
#
#   Stage 2:  evaluate_light_curve() - Pure JAX, call at every MCMC step.
#                                      Accepts JAX arrays / tracers so it is
#                                      fully compatible with jit, vmap, and
#                                      gradient-based samplers.
#
#   compute_light_curve()            - Convenience wrapper that calls both
#                                      stages in sequence.  Useful for
#                                      one-off calls outside MCMC.
# ---------------------------------------------------------------------------

def _prepare_ldc_coeffs(raw, ldc_mode: LdcMode, nwave: int, label: str) -> np.ndarray:
    """
    Validate and broadcast a set of LDC coefficients to shape (nwave, n_coeffs).

    Used for the quiet photosphere's coefficients in ``build_model``.
    Each element of ``raw`` may be a scalar (broadcast across wavelength)
    or an array of length ``nwave``.
    """
    if ldc_mode == "intensity_profile":
        return np.zeros((nwave, 1), dtype=np.float32)

    if ldc_mode not in _N_COEFFS:
        raise ValueError(
            f"unknown ldc_mode '{ldc_mode}'. "
            f"Must be one of {list(_N_COEFFS.keys()) + ['intensity_profile']}."
        )
    n_coeffs = _N_COEFFS[ldc_mode]

    if raw is None:
        raise ValueError(
            f"{label} must be provided for ldc_mode='{ldc_mode}'. "
            f"Expected {n_coeffs} coefficient(s)."
        )
    raw = list(raw) if not isinstance(raw, (list, tuple)) else list(raw)
    if len(raw) != n_coeffs:
        raise ValueError(
            f"{label}: ldc_mode='{ldc_mode}' expects {n_coeffs} "
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

    if all_scalar:
        coeff_str = ", ".join(f"{float(c[0]):.4f}" for c in coeff_arrays)
        print(
            f"{label}: scalar LDCs provided for '{ldc_mode}' law "
            f"([{coeff_str}]) - broadcasting across all {nwave} wavelength bins."
        )
    else:
        print(
            f"{label}: per-wavelength LDCs provided for '{ldc_mode}' law "
            f"({n_coeffs} coefficient(s), {nwave} wavelength bins)."
        )

    return coeffs


def build_model(
    wavelength: np.ndarray,
    flux_quiet: np.ndarray,
    params: dict,
    phases_rot: np.ndarray,
    stellar_grid_size: int,
    ve: float,
    ldc_mode: LdcMode = "quadratic",
    plot_map_wavelength: Optional[float] = None,
    oversample: int = 1,
) -> dict:
    """
    Pre-build all static model arrays.  Call this **once** before MCMC.

    Everything that does not change between MCMC steps is computed here in
    NumPy and stored in the returned model dict.  The only quantities that
    vary per step -- ``flux_active``, ``ar_lat``, ``ar_long``, ``ar_size``,
    ``ar_smoothness``, and each active region's own limb-darkening
    coefficients -- are intentionally excluded and passed to
    ``evaluate_light_curve`` instead (they may of course still be held
    fixed at every step if you don't want to sample/optimize them).

    Parameters
    ----------
    wavelength : array_like, shape (nwave,)
    flux_quiet : array_like, shape (nwave,)
        Quiet-photosphere spectrum.
    params : dict
        Quiet-photosphere model parameters. Recognised keys:

        ``inc_star`` : float, optional
            Stellar inclination in degrees (default: 90.0).
            90° = equator-on, 0° = pole-on.

        ``ldc_coeffs`` : list of float or list of array(nwave,)
            Quiet-photosphere limb-darkening coefficients for the chosen
            ``ldc_mode``:
            - ``"linear"``:     [u]
            - ``"quadratic"``:  [u1, u2]
            - ``"power2"``:     [c, alpha]
            - ``"kipping3"``:   [c1, c2, c3]
            - ``"nonlinear4"``: [c1, c2, c3, c4]
            Each element may be a scalar (broadcast to all wavelengths)
            or an array of length ``nwave``. Active regions carry their
            own, independent coefficients -- see ``evaluate_light_curve``.
            For ``"quadratic"`` mode only, ``u1`` and ``u2`` are also
            accepted as separate keys (legacy interface).

        ``mu_profile`` : array-like, optional
            Monotonically increasing mu grid points for
            ``ldc_mode="intensity_profile"`` (default: [0, 1]).

        ``I_profile`` : array-like, shape (nwave, n_mu_pts), optional
            Quiet-photosphere specific intensity at each (wavelength, mu)
            grid point. Required when ``ldc_mode="intensity_profile"``.
    phases_rot : array_like, shape (nphase,)
    stellar_grid_size : int
    ve : float
    ldc_mode : str
        Limb-darkening law, shared by the quiet photosphere and every
        active region (each with its own coefficient values).
    plot_map_wavelength : float, optional
    oversample : int, optional
        Number of sub-exposures per phase point.  Each requested phase is
        spread into ``oversample`` uniformly spaced sub-phases spanning one
        phase step, and the resulting fluxes are averaged.  This mimics
        finite-exposure integration and smooths limb-crossing artefacts.
        Default: 1 (no oversampling).

    Returns
    -------
    dict  - pass directly to ``evaluate_light_curve``
    """
    # Validate oversample
    if not isinstance(oversample, int) or oversample < 1:
        raise ValueError(
            f"oversample must be an integer >= 1, got {oversample}."
        )

    wavelength = np.asarray(wavelength, dtype=np.float32)
    flux_quiet = np.asarray(flux_quiet,  dtype=np.float32)
    phases_rot = np.atleast_1d(np.asarray(phases_rot, dtype=np.float32))

    nwave  = len(wavelength)
    nphase = len(phases_rot)  # original number of phases (before oversampling)

    # ---- Phase oversampling -------------------------------------------------
    if oversample > 1:
        phases_oversampled = _make_oversampled_phases(phases_rot, oversample)
        nphase_compute = len(phases_oversampled)
        print(
            f"build_model: oversampling enabled - {oversample} sub-exposures "
            f"per phase ({nphase} phases → {nphase_compute} sub-phases)."
        )
    else:
        phases_oversampled = phases_rot
        nphase_compute = nphase

    inc_star       = float(params.get("inc_star", 90.0))
    mu_profile_pts = np.asarray(params.get("mu_profile", [0.0, 1.0]),
                                dtype=np.float32)
    if not np.all(np.diff(mu_profile_pts) > 0):
        raise ValueError(
            "build_model: 'mu_profile' must be strictly increasing. "
            f"Got: {mu_profile_pts}"
        )
    I_profile = np.asarray(
        params.get("I_profile",
                   np.ones((nwave, len(mu_profile_pts)), dtype=np.float32)),
        dtype=np.float32,
    )

    # Accept either the unified "ldc_coeffs" key or legacy "u1"/"u2" for quadratic
    raw = params.get("ldc_coeffs", None)
    if raw is None and ldc_mode == "quadratic":
        raw = [params.get("u1", 0.0), params.get("u2", 0.0)]
    ldc_coeffs = _prepare_ldc_coeffs(raw, ldc_mode, nwave, label="build_model: quiet ldc_coeffs")

    grid = build_stellar_grid(stellar_grid_size, ve)

    if plot_map_wavelength is None:
        plot_map_wavelength = float(wavelength[nwave // 2])

    return dict(
        # spectral
        wavelength          = jnp.asarray(wavelength),
        flux_quiet          = jnp.asarray(flux_quiet),
        ldc_coeffs          = jnp.asarray(ldc_coeffs),
        I_profile           = jnp.asarray(I_profile),
        mu_profile_pts      = jnp.asarray(mu_profile_pts),
        # grid
        x_disc              = jnp.asarray(grid["x"]),
        y_disc              = jnp.asarray(grid["y"]),
        mu_disc             = jnp.asarray(grid["mu"]),
        row_idx             = jnp.asarray(grid["row_idx"]),
        vel_row             = jnp.asarray(grid["vel_row"]),
        star_pixel_rad      = grid["star_pixel_rad"],
        total_pixels        = grid["total_pixels"],
        n                   = grid["n"],
        flat_indices        = jnp.asarray(grid["flat_indices"]),
        phases_rot          = jnp.asarray(phases_oversampled),
        oversample          = oversample,
        nphase_original     = nphase,
        inc_star            = inc_star,
        ldc_mode            = ldc_mode,
        plot_map_wavelength = float(plot_map_wavelength),
        nwave               = nwave,
        nphase              = nphase_compute,
    )


def evaluate_light_curve(
    model: dict,
    flux_active: jnp.ndarray,
    ar_lat: jnp.ndarray,
    ar_long: jnp.ndarray,
    ar_size: jnp.ndarray,
    ar_smoothness: jnp.ndarray,
    ldc_coeffs_active: Optional[jnp.ndarray] = None,
    I_profile_active: Optional[jnp.ndarray] = None,
) -> dict:
    """
    Evaluate the light curve for a given set of active region parameters.

    This function is **pure JAX** -- all inputs may be JAX arrays or tracers,
    making it fully compatible with ``jit``, ``vmap``, and gradient-based
    samplers such as ``emcee_jax`` or ``blackjax``.

    When the model was built with ``oversample > 1``, the computation runs
    on the oversampled phase grid and the results are averaged back to the
    original phase grid before returning.

    Parameters
    ----------
    model : dict
        Pre-built model dict returned by ``build_model``.
    flux_active : jnp.ndarray, shape (nar, nwave) or (nwave,)
        Per-active-region flux spectrum.
        - If (nar, nwave): each active region gets its own spectrum.
        - If (nwave,):     broadcasts to all active regions.
    ar_lat : jnp.ndarray, shape (nar,)
        active region latitudes in degrees. Must be in [-90, 90].
    ar_long : jnp.ndarray, shape (nar,)
        active region longitudes in degrees. Must be in [0, 360).
    ar_size : jnp.ndarray, shape (nar,)
        active region angular radii in degrees ("sigma" of each AR's
        super-Gaussian).
    ar_smoothness : jnp.ndarray, shape (nar,) or scalar
        Super-Gaussian order controlling the sharpness of each AR's
        boundary (see ``_compute_ar_shape``). ``1`` is a true Gaussian;
        larger values sharpen the edge, converging to a hard-edged cap as
        ``ar_smoothness -> inf``. A scalar (or size-1 array) is broadcast
        to all active regions.
    ldc_coeffs_active : jnp.ndarray, shape (nar, nwave, n_coeffs) or (nwave, n_coeffs), optional
        Per-active-region limb-darkening coefficients, same law as the
        quiet photosphere (``model["ldc_mode"]``) but independent values.
        A (nwave, n_coeffs) array broadcasts to all active regions.
        Defaults to the quiet photosphere's own coefficients if omitted.
        Not used when ``ldc_mode="intensity_profile"``.
    I_profile_active : jnp.ndarray, shape (nar, nwave, n_mu_pts) or (nwave, n_mu_pts), optional
        Per-active-region specific-intensity profile, used only when
        ``ldc_mode="intensity_profile"``. Defaults to the quiet
        photosphere's own profile if omitted.

    Returns
    -------
    dict with keys
    ~~~~~~~~~~~~~~
    ``lc``        - (nphase_original, nwave) normalised flux at each wavelength bin
    ``epsilon``   - (nphase_original, nwave) contamination factor ε(λ)
    ``star_maps`` - (nphase_original, n, n) stellar flux map per phase
                    (maps are from the *first* sub-exposure of each phase
                    when oversampling is active)
    """
    flux_active   = jnp.atleast_1d(jnp.asarray(flux_active))
    ar_lat        = jnp.atleast_1d(jnp.asarray(ar_lat))
    ar_long       = jnp.atleast_1d(jnp.asarray(ar_long))
    ar_size       = jnp.atleast_1d(jnp.asarray(ar_size))
    ar_smoothness = jnp.atleast_1d(jnp.asarray(ar_smoothness))

    # Determine number of active regions
    nar   = ar_lat.size
    nwave = model["nwave"]

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

    # Broadcast a shared ar_smoothness (scalar or size-1) to all ARs
    if ar_smoothness.size == 1:
        ar_smoothness = jnp.broadcast_to(ar_smoothness, (nar,))
    elif ar_smoothness.shape != (nar,):
        raise ValueError(
            f"ar_smoothness shape mismatch: got shape {ar_smoothness.shape} "
            f"but expected a scalar or shape ({nar},)."
        )

    ldc_mode = model["ldc_mode"]
    n_coeffs = 1 if ldc_mode == "intensity_profile" else _N_COEFFS[ldc_mode]

    # ---- Per-AR limb-darkening coefficients: default to the quiet
    # photosphere's own, otherwise broadcast (nwave, n_coeffs) -> (nar, ...)
    if ldc_coeffs_active is None:
        ldc_coeffs_active = jnp.broadcast_to(
            model["ldc_coeffs"][None, :, :], (nar, nwave, n_coeffs)
        )
    else:
        ldc_coeffs_active = jnp.asarray(ldc_coeffs_active)
        if ldc_coeffs_active.ndim == 2:
            if ldc_coeffs_active.shape != (nwave, n_coeffs):
                raise ValueError(
                    f"ldc_coeffs_active shape mismatch: got {ldc_coeffs_active.shape} "
                    f"but expected ({nwave}, {n_coeffs})."
                )
            ldc_coeffs_active = jnp.broadcast_to(
                ldc_coeffs_active[None, :, :], (nar, nwave, n_coeffs)
            )
        elif ldc_coeffs_active.shape != (nar, nwave, n_coeffs):
            raise ValueError(
                f"ldc_coeffs_active shape mismatch: got {ldc_coeffs_active.shape} "
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

    spr        = model["star_pixel_rad"]
    inc_star   = model["inc_star"]
    oversample = model["oversample"]
    nphase_original = model["nphase_original"]

    # ---- active region Cartesian coordinates (JAX) --------------------------------
    ar_lat_rad  = jnp.deg2rad(ar_lat)
    ar_long_rad = jnp.deg2rad(ar_long)

    ar_cart = jnp.stack([
        spr * jnp.sin(ar_long_rad) * jnp.cos(ar_lat_rad),
        spr * jnp.sin(ar_lat_rad),
        spr * jnp.cos(ar_long_rad) * jnp.cos(ar_lat_rad),
    ], axis=-1)   # (nar, 3)

    # ---- Rotate all active regions for all phases (JAX) ---------------------------
    # phases_rot in the model is already oversampled if oversample > 1
    def _rotate_ars_at_phase(phase_deg):
        return vmap(
            lambda cart: rotate_active_region(cart, phase_deg, inc_star)
        )(ar_cart)

    all_ar_carts = vmap(_rotate_ars_at_phase)(
        model["phases_rot"]
    )   # (nphase_compute, nar, 3)

   # ---- Planet positions (if a transit model is present) ---------------
    if model.get("has_transit", False):
       planet_xyz_all = model["planet_xyz"]    # (nphase_compute, 3)
       k_val          = float(model["k"])
    else:
       # Dummy: planet permanently behind the star, zero-radius disc.
       nphase_compute = model["phases_rot"].shape[0]
       planet_xyz_all = jnp.zeros((nphase_compute, 3)).at[:, 2].set(-1e10)
       k_val          = 0.0

    # ---- All-phases computation ------------------------------------------
    lc_raw, epsilon, star_maps = _compute_all_phases(
        all_ar_carts,
        planet_xyz_all,
        wavelength          = model["wavelength"],
        flux_quiet          = model["flux_quiet"],
        flux_active         = flux_active,
        ldc_coeffs_quiet    = model["ldc_coeffs"],
        ldc_coeffs_active   = ldc_coeffs_active,
        I_profile_quiet     = model["I_profile"],
        I_profile_active    = I_profile_active,
        mu_profile_pts      = model["mu_profile_pts"],
        x_disc              = model["x_disc"],
        y_disc              = model["y_disc"],
        mu_disc             = model["mu_disc"],
        row_idx             = model["row_idx"],
        vel_row             = model["vel_row"],
        star_pixel_rad      = spr,
        total_pixels        = model["total_pixels"],
        arsize_rads         = jnp.deg2rad(ar_size),
        ar_smoothness       = ar_smoothness,
        k                   = k_val,
        ldc_mode            = model["ldc_mode"],
        plot_map_wavelength = model["plot_map_wavelength"],
        n                   = model["n"],
        flat_indices        = model["flat_indices"],
    )

    # ---- Oversample averaging --------------------------------------------
    if oversample > 1:
        # lc_raw: (nphase_compute, nwave) → (nphase_original, oversample, nwave) → mean
        lc_raw = lc_raw.reshape(nphase_original, oversample, nwave).mean(axis=1)

        # epsilon: (nphase_compute, nwave) → (nphase_original, oversample, nwave) → mean
        epsilon = epsilon.reshape(nphase_original, oversample, nwave).mean(axis=1)

        # star_maps: take only the first sub-exposure per original phase
        # (averaging 2D maps is expensive and rarely useful)
        star_maps = star_maps[::oversample]

    return {
        "lc"        : lc_raw,
        "epsilon"   : epsilon,
        "star_maps" : star_maps,
    }


def compute_light_curve(
    wavelength: np.ndarray,
    flux_quiet: np.ndarray,
    flux_active: np.ndarray,
    params: dict,
    ar_lat: np.ndarray,
    ar_long: np.ndarray,
    ar_size: np.ndarray,
    ar_smoothness: np.ndarray,
    phases_rot: np.ndarray,
    stellar_grid_size: int,
    ve: float,
    ldc_mode: LdcMode = "quadratic",
    ldc_coeffs_active: Optional[np.ndarray] = None,
    I_profile_active: Optional[np.ndarray] = None,
    plot_map_wavelength: Optional[float] = None,
    oversample: int = 1,
) -> dict:
    """
    Convenience wrapper: build model and evaluate in one call.

    Equivalent to::

        model  = build_model(wavelength, flux_quiet, params, phases_rot,
                             stellar_grid_size, ve, ldc_mode,
                             plot_map_wavelength, oversample)
        result = evaluate_light_curve(model, flux_active, ar_lat, ar_long,
                                      ar_size, ar_smoothness,
                                      ldc_coeffs_active, I_profile_active)

    Use ``build_model`` + ``evaluate_light_curve`` directly when running
    MCMC so the grid is built only once.

    Parameters
    ----------
    wavelength : array_like, shape (nwave,)
    flux_quiet : array_like, shape (nwave,)
    flux_active : array_like, shape (nar, nwave) or (nwave,)
    params : dict
    ar_lat : array_like, shape (nar,)
    ar_long : array_like, shape (nar,)
    ar_size : array_like, shape (nar,)
    ar_smoothness : array_like, shape (nar,) or scalar
        Super-Gaussian order controlling AR boundary sharpness (see
        ``evaluate_light_curve``). If scalar, it is shared across all ARs.
    phases_rot : array_like, shape (nphase,)
    stellar_grid_size : int
    ve : float
    ldc_mode : str
    ldc_coeffs_active : array_like, optional
        Per-AR limb-darkening coefficients (see ``evaluate_light_curve``).
    I_profile_active : array_like, optional
        Per-AR specific-intensity profile (see ``evaluate_light_curve``).
    plot_map_wavelength : float, optional
    oversample : int, optional
        Number of sub-exposures per phase point (default: 1).

    Returns
    -------
    dict with keys ``lc``, ``epsilon``, ``star_maps`` as NumPy arrays.
    """
    model  = build_model(
        wavelength, flux_quiet, params, phases_rot, stellar_grid_size,
        ve, ldc_mode, plot_map_wavelength, oversample,
    )

    flux_active_arr = np.atleast_1d(np.asarray(flux_active, dtype=np.float32))
    result = evaluate_light_curve(
        model,
        jnp.asarray(flux_active_arr),
        jnp.asarray(np.atleast_1d(np.asarray(ar_lat,  dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_long, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_size, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_smoothness, dtype=np.float32))),
        None if ldc_coeffs_active is None else jnp.asarray(np.asarray(ldc_coeffs_active, dtype=np.float32)),
        None if I_profile_active  is None else jnp.asarray(np.asarray(I_profile_active,  dtype=np.float32)),
    )
    return {
        "lc"        : np.array(result["lc"]),
        "epsilon"   : np.array(result["epsilon"]),
        "star_maps" : np.array(result["star_maps"]),
    }

def build_combined_model(
    wavelength:         np.ndarray,
    flux_quiet:         np.ndarray,
    params:             dict,
    times:              np.ndarray,
    P_rot:              float,
    transit_params:     dict,
    stellar_grid_size:  int,
    ve:                 float,
    ldc_mode:           LdcMode            = "quadratic",
    plot_map_wavelength: Optional[float]   = None,
    oversample:         int                = 1,
) -> dict:
    """
    Build a combined stellar-activity + planetary-transit sajax model.

    This is the entry point for modelling **active-region crossing events**:
    the planet mask is applied at the individual pixel level, so if the planet
    occults a starspot or facula the resulting anomaly in the light curve is
    computed correctly.

    Compared to multiplying independent stellar and transit light curves, this
    function correctly handles:
      • Planet occulting a spot (spot-crossing anomaly).
      • Planet occulting a facula (facula-crossing anomaly).
      • The varying limb-darkening depth of the transit as a function of
        the stellar surface brightness profile.

    Parameters
    ----------
    wavelength         : (nwave,)  wavelength array  [nm]
    flux_quiet         : (nwave,)  quiet-star flux spectrum
    params             : stellar model params dict (same as ``build_model``)
    times              : (ntime,)  absolute observation times  [days]
    P_rot              : stellar rotation period  [days]
    transit_params     : dict with keys (all required unless noted):
        ``t0``            - mid-transit epoch  [days]
        ``period``        - orbital period  [days]
        ``a_over_rstar``  - semi-major axis/R*  [dimensionless]
        ``inclination``   - orbital inclination  [rad]
        ``k``             - planet-to-star radius ratio  Rp/R* [dimensionless]
        ``ecc``           - eccentricity  [dimensionless] (default 0.0)
        ``omega_peri``    - argument of periastron  [rad]  (default 0.0)
    stellar_grid_size  : stellar radius in pixels
    ve                 : equatorial velocity  [km/s]
    ldc_mode           : limb-darkening law  (same options as ``build_model``)
    plot_map_wavelength: wavelength for 2D map output  [nm]
    oversample         : sub-exposure count per phase point  (default 1)

    Returns
    -------
    model dict - pass directly to ``evaluate_light_curve``

    Notes
    -----
    The oversampling is applied *consistently* to both the stellar rotation
    phase grid and the orbital time grid.  For each original time t_i with
    phase step dt, ``oversample`` sub-times are generated spanning
    [t_i - dt/2, t_i + dt/2), exactly mirroring ``_make_oversampled_phases``.
    """
    from .planet import build_transit_model   # local import avoids circular dep.

    times_arr  = np.asarray(times, dtype=np.float64)
    phases_rot = (times_arr / P_rot * 360.0) % 360.0

    # ---- Build the base stellar model (handles phase oversampling) ----------
    model = build_model(
        wavelength, flux_quiet, params, phases_rot, stellar_grid_size,
        ve, ldc_mode, plot_map_wavelength, oversample,
    )

    # ---- Compute oversampled TIMES to match the oversampled phases ----------
    # We work in absolute time (not phases) so the planet's orbital position
    # is computed correctly regardless of phase wrapping.
    if oversample > 1:
        n_t = len(times_arr)
        dt  = (times_arr[1] - times_arr[0]) if n_t > 1 else P_rot
        # Same offset scheme used in _make_oversampled_phases -- results align.
        offsets = np.linspace(-dt / 2.0, dt / 2.0, oversample, endpoint=False)
        offsets += dt / (2.0 * oversample)                          # centre sub-bins
        times_oversampled = (
            times_arr[:, None] + offsets[None, :]
        ).ravel().astype(np.float32)
    else:
        times_oversampled = times_arr.astype(np.float32)

    # ---- Build transit model (planet positions at oversampled times) ---------
    tp = transit_params
    transit = build_transit_model(
        times      = times_oversampled,
        t0         = float(tp["t0"]),
        period     = float(tp["period"]),
        a_over_rstar = float(tp["a_over_rstar"]),
        inclination  = float(tp["inclination"]),
        ecc          = float(tp.get("ecc",        0.0)),
        omega_peri   = float(tp.get("omega_peri", 0.0)),
        k            = float(tp["k"]),
    )

    # ---- Attach transit data to the model dict ------------------------------
    model["planet_xyz"]  = transit["planet_xyz"]   # (nphase_compute, 3)
    model["k"]           = transit["k"]
    model["has_transit"] = True
    model["P_rot"]       = P_rot
    model["times"]       = times_arr

    return model

def compute_combined_light_curve(
    wavelength:         np.ndarray,
    flux_quiet:         np.ndarray,
    flux_active:        np.ndarray,
    params:             dict,
    ar_lat:             np.ndarray,
    ar_long:            np.ndarray,
    ar_size:            np.ndarray,
    ar_smoothness:      np.ndarray,
    times:              np.ndarray,
    P_rot:              float,
    transit_params:     dict,
    stellar_grid_size:  int,
    ve:                 float,
    ldc_mode:           LdcMode            = "quadratic",
    ldc_coeffs_active:  Optional[np.ndarray] = None,
    I_profile_active:   Optional[np.ndarray] = None,
    plot_map_wavelength: Optional[float]   = None,
    oversample:         int                = 1,
) -> dict:
    """
    Convenience wrapper: build a combined stellar + transit model and
    evaluate it in one call.

    Equivalent to::

        model  = build_combined_model(wavelength, flux_quiet, params, times, P_rot,
                             transit_params, stellar_grid_size, ve, ldc_mode,
                             plot_map_wavelength, oversample)

        result = evaluate_light_curve(model, flux_active, ar_lat, ar_long,
                                      ar_size, ar_smoothness,
                                      ldc_coeffs_active, I_profile_active)

    Use ``build_model`` + ``evaluate_light_curve`` directly when running
    MCMC so the grid is built only once.

    Parameters
    ----------
    (All parameters match ``build_combined_model`` and
    ``evaluate_light_curve``.  See their docstrings for details.)

    transit_params : dict
        ``t0``, ``period``, ``a_over_rstar``, ``inclination``, ``k``,
        and optionally ``ecc``, ``omega_peri``.

    Returns
    -------
    dict with keys ``lc``, ``epsilon``, ``star_maps``  (same as
    ``compute_light_curve``).

    """

    model = build_combined_model(
        wavelength, flux_quiet, params, times, P_rot, transit_params,
        stellar_grid_size, ve, ldc_mode,
        plot_map_wavelength, oversample,
    )

    flux_active_arr = np.atleast_1d(np.asarray(flux_active, dtype=np.float32))
    result = evaluate_light_curve(
        model,
        jnp.asarray(flux_active_arr),
        jnp.asarray(np.atleast_1d(np.asarray(ar_lat,  dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_long, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_size, dtype=np.float32))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_smoothness, dtype=np.float32))),
        None if ldc_coeffs_active is None else jnp.asarray(np.asarray(ldc_coeffs_active, dtype=np.float32)),
        None if I_profile_active  is None else jnp.asarray(np.asarray(I_profile_active,  dtype=np.float32)),
    )
    return {
        "lc"        : np.array(result["lc"]),
        "epsilon"   : np.array(result["epsilon"]),
        "star_maps" : np.array(result["star_maps"]),
    }
