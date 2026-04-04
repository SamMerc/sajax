"""
core.py — JAX-accelerated stellar active region light-curve engine.

This module is a complete rewrite of ``SAGE1/sage.py`` in JAX.

Key differences from the original NumPy/SciPy implementation
-------------------------------------------------------------
1. **No wavelength loop.**
   The original code iterated over wavelengths with a Python ``for`` loop.
   Here the entire spectral axis is handled by ``jax.vmap``, which maps
   the single-channel computation across all wavelengths in parallel.

2. **No phase loop.**
   The original code iterated over rotational phases with a Python loop.
   Here all phases are computed in a single ``jax.vmap`` call — this is
   the main source of speedup over the original code.

3. **No scatter-index active region placement.**
   The original code located active region pixels via integer scatter indices
   (fancy indexing with ``.astype(int)``), which is not differentiable
   and incompatible with ``jit``.  SAJAX instead computes an analytic
   angular-distance mask over the full pixel arrays using ``jnp.where``,
   which is fully vectorised and differentiable.

4. **No class state mutation.**
   The original ``sage_class.rotate_star()`` mutated ``self.phases_rot``
   inside a loop — a latent bug.  SAJAX uses pure functions throughout.

5. **No astropy dependency for geometry.**
   Rotation matrices are implemented directly in JAX (see geometry.py).

6. **No transit-geometry parameters.**
   The original SAGE grid was sized using ``planet_pixel_size``,
   ``radiusratio``, and ``semimajor`` — artifacts of its transit-fitting
   origin.  SAJAX replaces these with a single ``stellar_grid_size``
   parameter: the stellar radius in pixels.  No planet required.

7. **Pre-masked grid.**
   ``build_stellar_grid`` applies the stellar disc mask immediately and
   returns 1D arrays containing only the in-disc pixels.  No starmask is
   ever passed to JAX functions — the mask is implicit in the data shape.
   The only 2D reconstruction happens at output time for ``star_maps``,
   using stored flat indices.

8. **Differentiable end-to-end.**
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
LdcMode = Literal[
    "linear",          # 1 coeff  : u
    "quadratic",       # 2 coeffs : u1, u2
    "power2",          # 2 coeffs : c, alpha
    "kipping3",        # 3 coeffs : c1, c2, c3
    "nonlinear4",      # 4 coeffs : c1, c2, c3, c4
    "intensity_profile",
]

# Number of LDC coefficients expected per law (used for validation in build_model)
_N_COEFFS: dict[str, int] = {
    "linear":     1,
    "quadratic":  2,
    "power2":     2,
    "kipping3":   3,
    "nonlinear4": 4,
}


# ---------------------------------------------------------------------------
# 1. Grid construction  (NumPy — runs once per model configuration)
# ---------------------------------------------------------------------------

def build_stellar_grid(
    stellar_grid_size: int,
    ve: float,
) -> dict:
    """
    Pre-compute the static stellar pixel grid, masked to the stellar disc.

    The mask is applied here once so that all downstream JAX functions
    receive 1D arrays containing only the in-disc pixels — no starmask
    is ever passed around.

    Parameters
    ----------
    stellar_grid_size : int
        Stellar radius in pixels.  This is the single resolution knob:
        higher values give a finer grid at the cost of n² memory and
        compute.  Values of 100-300 are typical.
    ve : float
        Stellar equatorial velocity [km/s].

    Returns
    -------
    dict with keys
    ~~~~~~~~~~~~~~
    ``n``             — full grid side length (always odd)
    ``star_pixel_rad``— stellar radius in pixels (= stellar_grid_size)
    ``total_pixels``  — number of in-disc pixels
    ``flat_indices``  — (total_pixels,) int  indices into the flattened
                        (n, n) grid; used to reconstruct 2D maps at output
    ``x``             — (total_pixels,) x pixel coordinates  [in-disc only]
    ``y``             — (total_pixels,) y pixel coordinates  [in-disc only]
    ``mu``            — (total_pixels,) limb-darkening cos θ [in-disc only]
    ``vel``           — (total_pixels,) Doppler factor Δv/c  [in-disc only]
    """
    C = 299_792.458  # speed of light [km/s]

    star_pixel_rad = float(stellar_grid_size)

    # n = 2 * radius + 1 so the centre falls on a pixel (odd grid)
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

    # Doppler velocity factor:  Δv/c = (y / R_star) * (ve / c)
    # y increases upward → redshift on the receding limb
    vel_disc = (y_disc / star_pixel_rad * (ve / C)).astype(np.float32)

    return dict(
        n             = n,
        star_pixel_rad= star_pixel_rad,
        total_pixels  = int(flat_indices.size),
        flat_indices  = flat_indices,          # kept in NumPy for scatter
        x             = x_disc,
        y             = y_disc,
        mu            = mu_disc,
        vel           = vel_disc,
    )


# ---------------------------------------------------------------------------
# 2. active region mask  (JAX — operates on 1D in-disc arrays)
# ---------------------------------------------------------------------------

def _compute_ar_mask(
    x_disc: jnp.ndarray,       # (total_pixels,)
    y_disc: jnp.ndarray,       # (total_pixels,)
    star_pixel_rad: float,
    spx: float,
    spy: float,
    spz: float,
    arsize_rad: float,
) -> jnp.ndarray:
    """
    Boolean mask over in-disc pixels: True where the pixel is inside the active region.

    Because ``x_disc`` and ``y_disc`` already contain only stellar-disc
    pixels, there is no need for an in-star check here.

    Parameters
    ----------
    x_disc, y_disc : (total_pixels,)
        Pixel coordinates of in-disc pixels.
    star_pixel_rad : float
        Stellar radius in pixels.
    spx, spy, spz : float
        active region centre Cartesian coordinates (after rotation + inclination).
    arsize_rad : float
        active region angular radius in radians.

    Returns
    -------
    jnp.ndarray, shape (total_pixels,), dtype bool
    """
    # z-coordinate of each pixel on the stellar sphere
    r2     = x_disc ** 2 + y_disc ** 2
    z_disc = jnp.sqrt(jnp.maximum(star_pixel_rad ** 2 - r2, 0.0))

    # active region centre in spherical coords
    arlong_rot = jnp.arctan2(spx, spz)
    arlat_rot  = jnp.arccos(jnp.clip(spy / star_pixel_rad, -1.0, 1.0))

    # Each pixel in spherical coords
    longi = jnp.arctan2(x_disc, z_disc)
    lati  = jnp.arccos(jnp.clip(y_disc / star_pixel_rad, -1.0, 1.0))

    # Great-circle distance between active region centre and each pixel
    delta_lon = jnp.abs(arlong_rot - longi)
    d_sigma   = jnp.arccos(jnp.clip(
        jnp.cos(arlat_rot) * jnp.cos(lati)
        + jnp.sin(arlat_rot) * jnp.sin(lati) * jnp.cos(delta_lon),
        -1.0, 1.0,
    ))

    # active regions with spz < 0 are on the far side of the star — not visible
    return (d_sigma <= arsize_rad) & (spz >= 0.0)


# ---------------------------------------------------------------------------
# 3. Single-wavelength flux  (vmapped over the spectral axis)
# ---------------------------------------------------------------------------

def _flux_at_wavelength(
    # --- vmapped: one scalar/slice per wavelength ---
    flux_quiet_wl: float,
    flux_active_wl: float,
    ldc_coeffs_wl: jnp.ndarray,  # (n_coeffs,)  — one row of ldc_coeffs
    I_prof_wl: jnp.ndarray,      # (n_mu_pts,)
    # --- broadcast: shared across wavelengths ---
    mu_disc: jnp.ndarray,         # (total_pixels,)
    vel_disc: jnp.ndarray,        # (total_pixels,)
    total_pixels: int,
    ar_masks: jnp.ndarray,      # (nar, total_pixels)
    mu_profile_pts: jnp.ndarray,  # (n_mu_pts,)
    ldc_mode: LdcMode,
) -> tuple[float, float, jnp.ndarray]:
    """
    Compute disc-integrated flux for a single wavelength channel.

    All arrays are 1D (in-disc pixels only) — no starmask needed.

    Returns
    -------
    star_spec  : float            — unarted integrated flux
    total_flux : float            — arted  integrated flux
    flux_disc  : (total_pixels,)  — per-pixel flux values (for map output)
    """
    # ---- Limb darkening -------------------------------------------------
    if ldc_mode == "intensity_profile":
        # Interpolate a user-supplied I(mu) profile.
        ldc = jnp.interp(mu_disc, mu_profile_pts, I_prof_wl,
                         left=0.0, right=0.0)
    elif ldc_mode == "linear":
        # I(μ) = 1 - u*(1 - μ)
        ldc = 1.0 - ldc_coeffs_wl[0] * (1.0 - mu_disc)
    elif ldc_mode == "quadratic":
        # I(μ) = 1 - u1*(1-μ) - u2*(1-μ)²
        ldc = (1.0
               - ldc_coeffs_wl[0] * (1.0 - mu_disc)
               - ldc_coeffs_wl[1] * (1.0 - mu_disc) ** 2)
    elif ldc_mode == "power2":
        # I(μ) = 1 - c*(1 - μ^α)
        ldc = 1.0 - ldc_coeffs_wl[0] * (1.0 - mu_disc ** ldc_coeffs_wl[1])
    elif ldc_mode == "kipping3":
        # I(μ) = 1 - c1*(1-μ^½) - c2*(1-μ) - c3*(1-μ^(3/2))
        ldc = (1.0
               - ldc_coeffs_wl[0] * (1.0 - mu_disc ** 0.5)
               - ldc_coeffs_wl[1] * (1.0 - mu_disc)
               - ldc_coeffs_wl[2] * (1.0 - mu_disc ** 1.5))
    else:  # "nonlinear4"  — Claret (2000) four-parameter law
        # I(μ) = 1 - Σ_{k=1}^{4} c_k*(1 - μ^(k/2))
        ldc = (1.0
               - ldc_coeffs_wl[0] * (1.0 - mu_disc ** 0.5)
               - ldc_coeffs_wl[1] * (1.0 - mu_disc)
               - ldc_coeffs_wl[2] * (1.0 - mu_disc ** 1.5)
               - ldc_coeffs_wl[3] * (1.0 - mu_disc ** 2.0))

    # ---- Unarted grid -------------------------------------------------
    hot_flux  = flux_quiet_wl  * (1.0 + vel_disc) * ldc   # (total_pixels,)
    star_spec = jnp.sum(hot_flux) / total_pixels

    # ---- Cold (active region) grid -----------------------------------------------
    cold_flux = flux_active_wl * (1.0 + vel_disc) * ldc   # (total_pixels,)

    # Replace hot pixels with cold pixels for each active region
    def _apply_one_ar(carry: jnp.ndarray,
                        ar_mask: jnp.ndarray) -> tuple:
        return jnp.where(ar_mask, cold_flux, carry), None

    arted_flux, _ = jax.lax.scan(_apply_one_ar, hot_flux, ar_masks)

    total_flux = jnp.sum(arted_flux) / total_pixels
    return star_spec, total_flux, arted_flux


# ---------------------------------------------------------------------------
# 4. Single-phase computation
# ---------------------------------------------------------------------------

def _compute_single_phase(
    ar_cart_all: jnp.ndarray,       # (nar, 3)
    *,
    wavelength: jnp.ndarray,          # (nwave,)
    flux_quiet_interp: jnp.ndarray,     # (nwave,)
    flux_active_interp: jnp.ndarray,    # (nwave,)
    ldc_coeffs: jnp.ndarray,          # (nwave, n_coeffs)
    I_profile: jnp.ndarray,           # (nwave, n_mu_pts)
    mu_profile_pts: jnp.ndarray,      # (n_mu_pts,)
    x_disc: jnp.ndarray,              # (total_pixels,)
    y_disc: jnp.ndarray,              # (total_pixels,)
    mu_disc: jnp.ndarray,             # (total_pixels,)
    vel_disc: jnp.ndarray,            # (total_pixels,)
    star_pixel_rad: float,
    total_pixels: int,
    arsize_rads: jnp.ndarray,       # (nar,)
    ldc_mode: LdcMode,
    plot_map_wavelength: float,
    n: int,                           # full grid side (for map scatter)
    flat_indices: jnp.ndarray,        # (total_pixels,) scatter indices
) -> tuple[float, jnp.ndarray, jnp.ndarray]:
    """
    Full spectral computation for one rotational phase.

    Returns
    -------
    flux_norm            : float
    contamination_factor : (nwave,)
    star_map             : (n, n)  flux map at plot_map_wavelength
    """
    # ---- active region masks: (nar, total_pixels) ------------------------------
    ar_masks = vmap(
        lambda cart, sr: _compute_ar_mask(
            x_disc, y_disc, star_pixel_rad,
            cart[0], cart[1], cart[2], sr,
        )
    )(ar_cart_all, arsize_rads)

    # ---- vmap over wavelengths ------------------------------------------
    _flux_vmap = vmap(
        functools.partial(
            _flux_at_wavelength,
            mu_disc        = mu_disc,
            vel_disc       = vel_disc,
            total_pixels   = total_pixels,
            ar_masks     = ar_masks,
            mu_profile_pts = mu_profile_pts,
            ldc_mode       = ldc_mode,
        ),
        in_axes=(0, 0, 0, 0),
    )

    star_specs, bin_fluxes, flux_discs = _flux_vmap(
        flux_quiet_interp,
        flux_active_interp,
        ldc_coeffs,
        I_profile,
    )

    # ---- Broadband flux and contamination factor ------------------------
    flux_norm            = jnp.sum(bin_fluxes) / jnp.sum(star_specs)
    contamination_factor = star_specs / jnp.where(
        bin_fluxes == 0.0, 1.0, bin_fluxes
    )

    # ---- Reconstruct 2D map at plot_map_wavelength ----------------------
    map_idx   = jnp.argmin(jnp.abs(wavelength - plot_map_wavelength))
    flux_1d   = flux_discs[map_idx]                         # (total_pixels,)
    star_map  = jnp.zeros(n * n).at[flat_indices].set(flux_1d).reshape(n, n)

    return flux_norm, contamination_factor, star_map


# ---------------------------------------------------------------------------
# 5. All-phases computation — vmapped over the phase axis
# ---------------------------------------------------------------------------

def _compute_all_phases(
    all_ar_carts: jnp.ndarray,   # (nphase, nar, 3)
    *,
    wavelength: jnp.ndarray,
    flux_quiet_interp: jnp.ndarray,
    flux_active_interp: jnp.ndarray,
    ldc_coeffs: jnp.ndarray,       # (nwave, n_coeffs)
    I_profile: jnp.ndarray,
    mu_profile_pts: jnp.ndarray,
    x_disc: jnp.ndarray,
    y_disc: jnp.ndarray,
    mu_disc: jnp.ndarray,
    vel_disc: jnp.ndarray,
    star_pixel_rad: float,
    total_pixels: int,
    arsize_rads: jnp.ndarray,
    ldc_mode: LdcMode,
    plot_map_wavelength: float,
    n: int,
    flat_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    vmap ``_compute_single_phase`` over the phase axis.

    Returns
    -------
    lc_raw    : (nphase,)
    epsilon   : (nphase, nwave)
    star_maps : (nphase, n, n)
    """
    _phase_vmap = vmap(
        functools.partial(
            _compute_single_phase,
            wavelength          = wavelength,
            flux_quiet_interp     = flux_quiet_interp,
            flux_active_interp    = flux_active_interp,
            ldc_coeffs          = ldc_coeffs,
            I_profile           = I_profile,
            mu_profile_pts      = mu_profile_pts,
            x_disc              = x_disc,
            y_disc              = y_disc,
            mu_disc             = mu_disc,
            vel_disc            = vel_disc,
            star_pixel_rad      = star_pixel_rad,
            total_pixels        = total_pixels,
            arsize_rads       = arsize_rads,
            ldc_mode            = ldc_mode,
            plot_map_wavelength = plot_map_wavelength,
            n                   = n,
            flat_indices        = flat_indices,
        ),
        in_axes=0,
    )
    return _phase_vmap(all_ar_carts)


# ---------------------------------------------------------------------------
# 6. Public API — two-stage design
#
#   Stage 1:  build_model()         — NumPy, call once before sampling.
#                                     Pre-builds the grid and all static
#                                     arrays that are fixed across MCMC steps.
#
#   Stage 2:  evaluate_light_curve() — Pure JAX, call at every MCMC step.
#                                      Accepts JAX arrays / tracers so it is
#                                      fully compatible with jit, vmap, and
#                                      gradient-based samplers.
#
#   compute_light_curve()            — Convenience wrapper that calls both
#                                      stages in sequence.  Useful for
#                                      one-off calls outside MCMC.
# ---------------------------------------------------------------------------

def build_model(
    wavelength: np.ndarray,
    flux_quiet: np.ndarray,
    params: dict,
    phases_rot: np.ndarray,
    stellar_grid_size: int,
    ve: float,
    ldc_mode: LdcMode = "quadratic",
    plot_map_wavelength: Optional[float] = None,
) -> dict:
    """
    Pre-build all static model arrays.  Call this **once** before MCMC.

    Everything that does not change between MCMC steps is computed here in
    NumPy and stored in the returned model dict.  The only quantities that
    vary per step — ``flux_active``, ``ar_lat``, ``ar_long``,
    ``ar_size`` — are intentionally excluded and passed to
    ``evaluate_light_curve`` instead.

    Parameters
    ----------
    wavelength : array_like, shape (nwave,)
    flux_quiet : array_like, shape (nwave,)
    params : dict
        ``u1``, ``u2``, ``inc_star``, ``mu_profile``, ``I_profile``
    phases_rot : array_like, shape (nphase,)
    stellar_grid_size : int
    ve : float
    ldc_mode : str
    plot_map_wavelength : float, optional

    Returns
    -------
    dict  — pass directly to ``evaluate_light_curve``
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux_quiet   = np.asarray(flux_quiet,   dtype=np.float64)
    phases_rot = np.atleast_1d(np.asarray(phases_rot, dtype=np.float64))

    nwave  = len(wavelength)
    nphase = len(phases_rot)

    inc_star       = float(params.get("inc_star", 90.0))
    mu_profile_pts = np.asarray(params.get("mu_profile", [0.0, 1.0]),
                                dtype=np.float64)
    I_profile = np.asarray(
        params.get("I_profile",
                   np.ones((nwave, len(mu_profile_pts)), dtype=np.float64)),
        dtype=np.float64,
    )

    if ldc_mode == "intensity_profile":
        # Intensity-profile mode does not use analytical LDC coefficients
        ldc_coeffs = np.zeros((nwave, 1), dtype=np.float64)
    else:
        n_coeffs = _N_COEFFS[ldc_mode]

        # Accept either the unified "ldc_coeffs" key or legacy "u1"/"u2" for quadratic
        raw = params.get("ldc_coeffs", None)
        if raw is None and ldc_mode == "quadratic":
            raw = [params.get("u1", 0.0), params.get("u2", 0.0)]
        if raw is None:
            raise ValueError(
                f"build_model: params must contain 'ldc_coeffs' for "
                f"ldc_mode='{ldc_mode}'. Expected {n_coeffs} coefficient(s)."
            )

        raw = list(raw) if not isinstance(raw, (list, tuple)) else list(raw)
        if len(raw) != n_coeffs:
            raise ValueError(
                f"build_model: ldc_mode='{ldc_mode}' expects {n_coeffs} "
                f"coefficient(s) but {len(raw)} were provided."
            )

        # Build (nwave, n_coeffs) — each element of raw is scalar or (nwave,)
        coeff_arrays = []
        all_scalar   = True
        for i, coeff in enumerate(raw):
            c = np.asarray(coeff, dtype=np.float64)
            if c.ndim == 0:
                coeff_arrays.append(np.full(nwave, float(c)))
            else:
                if len(c) != nwave:
                    raise ValueError(
                        f"build_model: ldc_coeffs[{i}] has length {len(c)} "
                        f"but wavelength grid has {nwave} bins. They must match."
                    )
                coeff_arrays.append(c)
                all_scalar = False

        ldc_coeffs = np.stack(coeff_arrays, axis=1)  # (nwave, n_coeffs)

        if all_scalar:
            coeff_str = ", ".join(f"{float(c[0]):.4f}" for c in coeff_arrays)
            print(
                f"build_model: scalar LDCs provided for '{ldc_mode}' law "
                f"([{coeff_str}]) — broadcasting across all {nwave} wavelength bins."
            )
        else:
            print(
                f"build_model: per-wavelength LDCs provided for '{ldc_mode}' law "
                f"({n_coeffs} coefficient(s), {nwave} wavelength bins)."
            )

    grid = build_stellar_grid(stellar_grid_size, ve)

    if plot_map_wavelength is None:
        plot_map_wavelength = float(wavelength[nwave // 2])

    return dict(
        # spectral
        wavelength          = jnp.asarray(wavelength),
        flux_quiet            = jnp.asarray(flux_quiet),
        ldc_coeffs          = jnp.asarray(ldc_coeffs),
        I_profile           = jnp.asarray(I_profile),
        mu_profile_pts      = jnp.asarray(mu_profile_pts),
        # grid
        x_disc              = jnp.asarray(grid["x"]),
        y_disc              = jnp.asarray(grid["y"]),
        mu_disc             = jnp.asarray(grid["mu"]),
        vel_disc            = jnp.asarray(grid["vel"]),
        star_pixel_rad      = grid["star_pixel_rad"],
        total_pixels        = grid["total_pixels"],
        n                   = grid["n"],
        flat_indices        = jnp.asarray(grid["flat_indices"]),
        # phases
        phases_rot          = jnp.asarray(phases_rot),
        # metadata
        inc_star            = inc_star,
        ldc_mode            = ldc_mode,
        plot_map_wavelength = float(plot_map_wavelength),
        nwave               = nwave,
        nphase              = nphase,
    )


def evaluate_light_curve(
    model: dict,
    flux_active: jnp.ndarray,
    ar_lat: jnp.ndarray,
    ar_long: jnp.ndarray,
    ar_size: jnp.ndarray,
) -> dict:
    """
    Evaluate the light curve for a given set of active region parameters.

    This function is **pure JAX** — all inputs may be JAX arrays or tracers,
    making it fully compatible with ``jit``, ``vmap``, and gradient-based
    samplers such as ``emcee_jax`` or ``blackjax``.

    Parameters
    ----------
    model : dict
        Pre-built model dict returned by ``build_model``.
    flux_active : jnp.ndarray, shape (nwave,)
        Active-region (active region) flux spectrum.  This is the only spectral
        quantity that varies between MCMC steps (via the contrast parameter).
    ar_lat : jnp.ndarray, shape (nar,)
        active region latitudes in degrees.
    ar_long : jnp.ndarray, shape (nar,)
        active region longitudes in degrees.
    ar_size : jnp.ndarray, shape (nar,)
        active region angular radii in degrees.

    Returns
    -------
    dict with keys
    ~~~~~~~~~~~~~~
    ``lc``        — (nphase,)       normalised broadband light curve
    ``epsilon``   — (nphase, nwave) contamination factor ε(λ)
    ``star_maps`` — (nphase, n, n)  stellar flux map per phase
    """
    flux_active  = jnp.atleast_1d(jnp.asarray(flux_active))
    ar_lat   = jnp.atleast_1d(jnp.asarray(ar_lat))
    ar_long  = jnp.atleast_1d(jnp.asarray(ar_long))
    ar_size  = jnp.atleast_1d(jnp.asarray(ar_size))

    spr        = model["star_pixel_rad"]
    inc_star   = model["inc_star"]

    # ---- active region Cartesian coordinates (JAX) --------------------------------
    ar_lat_rad  = jnp.deg2rad(ar_lat)
    ar_long_rad = jnp.deg2rad(ar_long)

    ar_cart = jnp.stack([
        spr * jnp.sin(ar_long_rad) * jnp.cos(ar_lat_rad),
        spr * jnp.sin(ar_lat_rad),
        spr * jnp.cos(ar_long_rad) * jnp.cos(ar_lat_rad),
    ], axis=-1)   # (nar, 3)

    # ---- Rotate all active regions for all phases (JAX) ---------------------------
    def _rotate_ars_at_phase(phase_deg):
        return vmap(
            lambda cart: rotate_active_region(cart, phase_deg, inc_star)
        )(ar_cart)

    all_ar_carts = vmap(_rotate_ars_at_phase)(
        model["phases_rot"]
    )   # (nphase, nar, 3)

    # ---- All-phases computation ------------------------------------------
    lc_raw, epsilon, star_maps = _compute_all_phases(
        all_ar_carts,
        wavelength          = model["wavelength"],
        flux_quiet_interp     = model["flux_quiet"],
        flux_active_interp    = flux_active,
        ldc_coeffs          = model["ldc_coeffs"],
        I_profile           = model["I_profile"],
        mu_profile_pts      = model["mu_profile_pts"],
        x_disc              = model["x_disc"],
        y_disc              = model["y_disc"],
        mu_disc             = model["mu_disc"],
        vel_disc            = model["vel_disc"],
        star_pixel_rad      = spr,
        total_pixels        = model["total_pixels"],
        arsize_rads       = jnp.deg2rad(ar_size),
        ldc_mode            = model["ldc_mode"],
        plot_map_wavelength = model["plot_map_wavelength"],
        n                   = model["n"],
        flat_indices        = model["flat_indices"],
    )

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
    phases_rot: np.ndarray,
    stellar_grid_size: int,
    ve: float,
    ldc_mode: LdcMode = "quadratic",
    plot_map_wavelength: Optional[float] = None,
) -> dict:
    """
    Convenience wrapper: build model and evaluate in one call.

    Equivalent to::

        model  = build_model(wavelength, flux_quiet, params, phases_rot,
                             stellar_grid_size, ve, ldc_mode,
                             plot_map_wavelength)
        result = evaluate_light_curve(model, flux_active,
                                      ar_lat, ar_long, ar_size)

    Use ``build_model`` + ``evaluate_light_curve`` directly when running
    MCMC so the grid is built only once.

    Returns
    -------
    dict with keys ``lc``, ``epsilon``, ``star_maps`` as NumPy arrays.
    """
    model  = build_model(
        wavelength, flux_quiet, params, phases_rot,
        stellar_grid_size, ve, ldc_mode, plot_map_wavelength,
    )
    result = evaluate_light_curve(
        model,
        jnp.asarray(np.asarray(flux_active,  dtype=np.float64)),
        jnp.asarray(np.atleast_1d(np.asarray(ar_lat,  dtype=np.float64))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_long, dtype=np.float64))),
        jnp.asarray(np.atleast_1d(np.asarray(ar_size, dtype=np.float64))),
    )
    return {
        "lc"        : np.array(result["lc"]),
        "epsilon"   : np.array(result["epsilon"]),
        "star_maps" : np.array(result["star_maps"]),
    }