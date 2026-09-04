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
   velocity v. The stellar rotation axis is the y-axis in Carthesian
   coordinates, so the sky-projected line-of-sight velocity is
   ``v_z = -(ve/R) * sin(i_star) * x``: it depends only on a pixel's
   x-coordinate, and all pixels in the same grid column share one velocity --
   the (expensive) spectral resampling is done once per column (``n``
   columns) rather than once per pixel (``~n^2``), then broadcast back out to
   pixels.

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
*Do NOT jit(make_lc) directly* -- it contains Python-level
control flow on model metadata.  Instead, the inner _compute_all_phases
is the hot path and is safe to JIT via::

    from jax import jit
    _compute_all_phases_jit = jit(_compute_all_phases, static_argnames=[
        "star_pixel_rad", "total_pixels", "ld_mode",
        "plot_map_wavelength", "n", "transit_softness"
    ])
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
from .planet import _compute_all_planets_mask, compute_multi_planet_sky_positions

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
    the ``exp * erfc`` product can never overflow to NaN — in value or in
    gradient — on long baselines.
    """
    A, B, C, D1, D2, f1 = (0.9687734504375167, -0.251299705922117,
                           0.22675974948468916, 0.15551880775110513,
                           1.2150539528490194, 0.12695865022878844)
    f2 = 1.0 - f1
    # Time in units of FWHM from the peak. Clamped on the rise side: at
    # tn = -20 the erfc factor has underflowed to exactly 0 while the exp
    # factor is still finite, so the product is exactly 0 there and for
    # all earlier times — and the clamp keeps exp from overflowing to
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

    # Apply mask → 1D in-disc arrays
    flat_indices = np.flatnonzero(starmask)   # (total_pixels,)
    x_disc = xg.ravel()[flat_indices].astype(np.float32)
    y_disc = yg.ravel()[flat_indices].astype(np.float32)
    r_disc = np.sqrt(r2.ravel()[flat_indices]).astype(np.float32)

    # mu = cos θ = sqrt(1 - (r/R)²), clamped for float32 safety
    mu_disc = np.sqrt(
        np.clip(1.0 - (r_disc / star_pixel_rad) ** 2, 0.0, 1.0)
    ).astype(np.float32)

    # coords[0] = -n//2, so col_idx = x + n//2 maps x onto {0, ..., n-1}.
    col_idx = (x_disc + n // 2).astype(np.int32)

    # Sky-frame spin axis (0, sin i, cos i) → v_z = -(ve/R)*sin(i)*x, a function of x alone; +x recedes.
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

def _compute_single_phase(
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
    pixel-level planet occultation.
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
    star_map              : (n, n)  flux map at plot_map_wavelength
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

    # ---- Reconstruct 2D map at plot_map_wavelength ----------------------
    map_idx   = jnp.argmin(jnp.abs(wavelength - plot_map_wavelength))
    flux_1d   = flux_discs[map_idx]   # (total_pixels,)
    star_map  = jnp.zeros(n * n).at[flat_indices].set(flux_1d).reshape(n, n)

    return flux_per_wavelength, star_map


# ---------------------------------------------------------------------------
# 6. All-phases computation -- vmapped over the phase axis
# ---------------------------------------------------------------------------

def _compute_all_phases(
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
    k:                   jnp.ndarray,  # (nplanet, nwave) Rp / R★, one value per planet per wavelength
    ld_mode:            LdMode,
    plot_map_wavelength: float,
    n:                   int,
    flat_indices:        jnp.ndarray,
    transit_softness:    float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    vmap ``_compute_single_phase`` over the phase axis.

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
            _compute_single_phase,
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Time-varying counterpart to ``_compute_all_phases``. Used only when make_lc() is given at least
    one AR parameter with a time axis. Identical to _compute_all_phases except
    ar_lat/ar_long/arsize_rads/ ar_smoothness/flux_active are supplied per-phase and the AR's Cartesian
    position is rebuilt and rotated fresh every phase. This function's existence adds no cost to 
    _compute_all_phases or any caller that doesn't use it.

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
            transit_softness=transit_softness,
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
    them, since a bare phase array wraps every 360° and can't recover the
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
        90° = equator-on, 0° = pole-on.
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
                f"per phase ({nphase} phases → {nphase_compute} sub-phases)."
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
        planet_xyz_all = compute_multi_planet_sky_positions(
            model["times_oversampled"], t0 - model["t_ref"], period, a_over_rstar, inclination,
            ecc_val, omega_peri_val, sp_orb_val,
        )   # (nphase_compute, nplanet, 3)
        nplanet = jnp.atleast_1d(jnp.asarray(t0)).shape[-1]
        k_val = k
    elif model.get("has_transit", False):
        # Static, backwards-compatible path: positions baked at build time.
        planet_xyz_all = model["planet_xyz"]    # (nphase_compute, nplanet, 3)
        k_val          = model["k"]             # (nplanet, nwave), already dense
        nplanet        = model["nplanet"]
    else:
        # Dummy: planet permanently behind the star, zero-radius disc.
        nphase_compute = model["phases_rot"].shape[0]
        nplanet        = 1
        planet_xyz_all = jnp.zeros((nphase_compute, 1, 3)).at[:, :, 2].set(-1e10)
        k_val          = 0.0

    # ---- Broadcast k to (nplanet, nwave): see _prepare_transit_k for the
    # full shape convention. A scalar means the same (achromatic) radius
    # ratio at every planet/wavelength; a genuinely chromatic and/or
    # per-planet depth is supported (the occultation mask is computed per
    # wavelength -- see _flux_at_wavelength -- specifically to support this).
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
        )

    # ---- Oversample averaging --------------------------------------------
    if oversample > 1:
        # lc_raw: (nphase_compute, nwave) → (nphase_original, oversample, nwave) → mean
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
        90° = equator-on, 0° = pole-on.
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

