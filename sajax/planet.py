"""
planet.py — Keplerian planet orbit and pixel-level transit geometry for sajax.

This module is a standalone companion to sajax/core.py.  It can be used
independently to compute transit light curves, or integrated with sajax
via ``build_system`` / ``quick_lc`` (defined in core.py, transit
parameters optional) to correctly model active-region crossing events —
i.e. cases where the planet occultes a starspot or facula during transit.

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

- **X** — sky-plane east-west (positive east)
- **Y** — sky-plane north-south (positive north, foreshortened by cos i)
- **Z** — line-of-sight toward observer (Z > 0 ⟹ planet in front of star)

All sky positions are in units of the stellar radius R*.

Minimum parameter set
---------------------

``t0``
   mid-transit epoch [days]
``period``
   orbital period [days]
``a_over_rstar``
   semimajor axis / R* (dimensionless). May be derived from stellar
   density via ``stellar_density_to_a_over_rstar()``.
``inclination``
   orbital inclination [rad] (90deg / pi/2 = perfect edge-on)
``ecc``
   orbital eccentricity [0, 1)
``omega_peri``
   argument of periastron [rad] (ω = 0deg → periapsis at ascending
   node; ω = 90deg → periapsis at inferior conjunction / transit
   centre for a circular orbit)
``sp_orb``
   sky-projected spin-orbit angle, λ [rad]. ``sp_orb`` rotates the
   transit chord about the stellar centre, in the sky plane. Angle is
   relative to the stellar equator.
``k``
   planet-to-star radius ratio Rp / R*

Limb darkening
--------------
The same LDC law stored in the sajax model dict is applied automatically
to occulted pixels — no separate transit LDC parameters are required.

Public API
----------
  ``_kepler(M, ecc)``                         — differentiable Kepler solver
  ``planet_sky_position(...)``                — single-epoch sky coords (X, Y, Z)
  ``compute_planet_sky_positions(...)``       — vectorised over an array of times
  ``compute_multi_planet_sky_positions(...)`` — vectorised over times AND planets
  ``_compute_planet_mask(...)``               — per-pixel occultation mask, one planet
  ``_compute_all_planets_mask(...)``          — per-pixel occultation mask, over multiple planets
  ``build_transit_model(...)``                — pre-compute positions for all times/planets
  ``stellar_density_to_a_over_rstar()``       — unit-conversion convenience
"""

from __future__ import annotations

import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap


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
    sp_orb: float = 0.0,
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
    sp_orb        : sky-projected spin-orbit angle, λ  [rad]
                    Rotates the transit chord about the stellar
                    centre, in the sky plane. Angle is relative to the 
                    stellar equator.

    Returns
    -------
    X, Y, Z : sky-plane coordinates in units of R*
        X  — east-west (positive east) at sp_orb = 0
        Y  — north-south projected  (= r sin(ω+f) cos i) at sp_orb = 0
        Z  — toward observer  (Z > 0 ⟹ transit;  Z < 0 ⟹ occultation)
        For nonzero sp_orb, (X, Y) are the sp_orb = 0 values rotated
        by λ.

    Notes
    -----
    The sky-plane separation from the stellar centre is sqrt(X^2 + Y^2)
    (rotation-invariant, so unaffected by spin-orbit angle).
    A transit (or occultation) event occurs when sqrt(X^2 + Y^2) < 1 + k,
    where k = Rp / R*.

    The spin-orbit angle rotation is applied last, after the orbital-mechanics
    (X, Y):

        X' = X cos(λ) - Y sin(λ)
        Y' = X sin(λ) + Y cos(λ)

    λ = 0 leaves (X, Y) untouched.
    λ = π/2 (polar transit) swaps the roles of X and Y, so a central
    (b = 0) transit chord that used to sweep through X = 0 now sweeps
    through Y = 0 instead.
    Positive λ rotates the orbit counterclockwise on the sky as seen by the observer.
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

    # ---- Spin-orbit angle: rotate (X, Y) relative to the (sky-Y-fixed)
    # stellar spin axis. Z (transit front/back) is unaffected. See the
    # docstring Notes for the rotation convention.
    cos_obl = jnp.cos(sp_orb)
    sin_obl = jnp.sin(sp_orb)
    X, Y = X * cos_obl - Y * sin_obl, X * sin_obl + Y * cos_obl

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
    sp_orb: float = 0.0,
) -> jnp.ndarray:
    """
    Vectorised wrapper around ``planet_sky_position``.

    Parameters
    ----------
    times : (ntime,) array of observation epochs
    sp_orb: sky-projected spin-orbit angle λ [rad] (default 0.0).
        See ``planet_sky_position``.

    Returns
    -------
    xyz : (ntime, 3) array  —  columns are [X, Y, Z] in units of R*
    """
    _pos = vmap(
        lambda t: jnp.stack(
            planet_sky_position(
                t, t0, period, a_over_rstar, inclination, ecc, omega_peri,
                sp_orb,
            )
        )
    )(jnp.asarray(times))   # (ntime, 3)
    return _pos


def compute_multi_planet_sky_positions(
    times: jnp.ndarray,
    t0: float | jnp.ndarray,
    period: float | jnp.ndarray,
    a_over_rstar: float | jnp.ndarray,
    inclination: float | jnp.ndarray,
    ecc: float | jnp.ndarray = 0.0,
    omega_peri: float | jnp.ndarray = 0.0,
    sp_orb: float | jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """
    Multi-planet counterpart of ``compute_planet_sky_positions``.

    Each orbital-element argument is scalar or shape ``(nplanet,)`` --
    exactly the "trailing axis" convention already used for active regions
    (``ar_lat``/``ar_long``/... in ``core.py``'s ``make_lc``). ``nplanet`` is
    inferred from ``t0``'s trailing axis; every other argument is broadcast
    to ``(nplanet,)`` if given as a scalar (or size-1 array), and otherwise
    must already have length ``nplanet``.

    Implemented as a thin ``vmap`` of ``compute_planet_sky_positions`` over
    the planet axis -- ``planet_sky_position``/``compute_planet_sky_positions``
    themselves are untouched, so this adds no risk to the underlying Kepler
    solver. For a single planet, ``compute_planet_sky_positions`` remains the
    simpler, preferred entry point -- this function is for ``nplanet > 1``
    (or for callers, like ``build_transit_model``, that want to support
    either uniformly).

    Parameters
    ----------
    times : (ntime,) array of observation epochs
    t0, period, a_over_rstar, inclination, ecc, omega_peri, sp_orb :
        Scalar or ``(nplanet,)``. See ``planet_sky_position`` for units and
        meaning. ``sp_orb`` is in radians, like ``planet_sky_position``.

    Returns
    -------
    xyz : (ntime, nplanet, 3) array -- columns are [X, Y, Z] in units of R*
    """
    t0_arr = jnp.atleast_1d(jnp.asarray(t0))
    nplanet = t0_arr.shape[-1]

    def _broadcast(name, value):
        arr = jnp.atleast_1d(jnp.asarray(value))
        if arr.shape[-1] == 1:
            return jnp.broadcast_to(arr, (nplanet,))
        if arr.shape[-1] != nplanet:
            raise ValueError(
                f"compute_multi_planet_sky_positions: '{name}' has length "
                f"{arr.shape[-1]} but nplanet (inferred from t0) is "
                f"{nplanet}. Every orbital-element argument must be a "
                "scalar or match t0's trailing axis length."
            )
        return arr

    period_arr       = _broadcast("period", period)
    a_over_rstar_arr = _broadcast("a_over_rstar", a_over_rstar)
    inclination_arr  = _broadcast("inclination", inclination)
    ecc_arr          = _broadcast("ecc", ecc)
    omega_peri_arr   = _broadcast("omega_peri", omega_peri)
    sp_orb_arr       = _broadcast("sp_orb", sp_orb)

    times_jax = jnp.asarray(times)
    _pos = vmap(
        lambda t0_p, period_p, a_p, inc_p, ecc_p, omega_p, sp_p:
            compute_planet_sky_positions(
                times_jax, t0_p, period_p, a_p, inc_p, ecc_p, omega_p, sp_p,
            )
    )(t0_arr, period_arr, a_over_rstar_arr, inclination_arr, ecc_arr,
      omega_peri_arr, sp_orb_arr)   # (nplanet, ntime, 3)

    return jnp.moveaxis(_pos, 0, 1)   # (ntime, nplanet, 3)


# ---------------------------------------------------------------------------
# 4. Per-pixel transit mask on the sajax stellar grid
# ---------------------------------------------------------------------------

_MASK_D_TINY = 1e-12  # floor under the sqrt so d(sqrt)/d(d2) doesn't blow up
                       # for a pixel landing exactly at the planet's centre.


def _compute_planet_mask(
    x_disc: jnp.ndarray,   # (total_pixels,)  pixel x coordinates
    y_disc: jnp.ndarray,   # (total_pixels,)  pixel y coordinates
    star_pixel_rad: float,
    X: jnp.ndarray,        # planet sky-plane x  [R*]
    Y: jnp.ndarray,        # planet sky-plane y  [R*]
    Z: jnp.ndarray,        # planet line-of-sight  [R*]  — Z > 0 ⟹ transit
    k: float,              # Rp / R*
    softness: float = 0.0, # transition width [R*]; 0.0 = exact hard edge
) -> jnp.ndarray:
    """
    Mask over in-disc pixels: non-zero where the pixel is occulted by the
    planet at this epoch.

    The mask is non-zero only when Z > 0 (planet in front of the star).
    Pixels inside the planet disc contribute zero flux; if those pixels
    coincide with an active region, the spot-crossing anomaly emerges
    automatically.

    By default (``softness=0.0``) this is the exact hard-edged disc used for
    physical light-curve simulation: on the fixed pixel grid, occulted flux
    is a function of ``k``/``X``/``Y`` (flat between the
    instants a pixel boundary crosses the disc edge), so its analytic
    derivative w.r.t. those parameters -- hence w.r.t. every transit-geometry
    parameter that moves the mask (``a_over_rstar``, ``inclination``, ``t0``,
    ``period``, ``ecc``, ``omega_peri``) -- is exactly 0 almost everywhere.
    This means that jax.grad gives NUTS/HMC no likelihood signal at all in
    those directions.

    Passing ``softness > 0`` replaces the hard threshold with a sigmoid of
    that transition width (in stellar radii), giving a smooth, non-zero
    gradient w.r.t. every transit-geometry parameter -- for gradient-based
    retrieval only. It biases the effective transit depth/duration slightly
    (a soft edge occults less than a hard one right at the boundary), so it
    is opt-in and defaults off; ``quick_lc`` / physical simulation
    is unaffected unless requested.

    Parameters
    ----------
    x_disc, y_disc  : in-disc pixel coordinates  [pixels]
    star_pixel_rad  : stellar radius in pixels
    X, Y            : planet sky position  [R*]
    Z               : planet line-of-sight position  [R*]
    k               : planet-to-star radius ratio
    softness        : sigmoid transition width [R*] (default 0.0: hard edge)

    Returns
    -------
    jnp.ndarray, shape (total_pixels,), dtype float32, values in [0, 1]
    (exactly {0, 1} when softness == 0.0)
    """
    # Normalise pixel coordinates to stellar radii
    xn = x_disc / star_pixel_rad
    yn = y_disc / star_pixel_rad

    # Squared sky-plane distance from planet centre to each pixel
    d2 = (xn - X) ** 2 + (yn - Y) ** 2

    if softness > 0.0:
        d = jnp.sqrt(jnp.maximum(d2, _MASK_D_TINY))
        disc_mask = jax.nn.sigmoid((k - d) / softness)
    else:
        # Hard disc mask: pixel is occulted iff it lies within the planet disc.
        disc_mask = (d2 < k ** 2).astype(jnp.float32)

    # Hard Z gate: planet in front of the star is topologically binary.
    z_gate = jnp.where(Z > 0.0, 1.0, 0.0)
    # Use jnp.where instead of `if k == 0.0` so this stays JAX-traceable when k
    # is a sampled parameter (tracer) inside a numpyro / JAX-jit context.
    return jnp.where(k > 0.0, disc_mask * z_gate, jnp.zeros_like(disc_mask))


def _compute_all_planets_mask(
    x_disc: jnp.ndarray,    # (total_pixels,)
    y_disc: jnp.ndarray,    # (total_pixels,)
    star_pixel_rad: float,
    planet_xyz: jnp.ndarray,  # (nplanet, 3)
    k: jnp.ndarray,           # (nplanet,)
    softness: float = 0.0,
) -> jnp.ndarray:
    """
    Combine per-planet occultation masks (see ``_compute_planet_mask``) over
    ``nplanet`` planets at a single epoch and wavelength.

    Planets are opaque occulters, not spectral-contrast modulators (unlike
    active regions, whose contrasts sum -- see core.py's module docstring),
    so the physically correct combination for possibly-overlapping planets
    is multiplicative: each planet's ``(1 - mask)`` is the fraction of a
    pixel's flux that *survives* that planet, and independent survival
    fractions multiply. This also keeps the combined result in [0, 1] for
    overlapping planets, unlike a sum of masks (which can exceed 1 and
    drive the "surviving flux" negative).

    ``nplanet == 1`` fast path
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    For a single planet, ``1 - prod([1 - mask]) == mask`` algebraically, but
    that identity is not free at trace time: the general path below always
    emits three real elementwise/reduction ops (subtract, product-reduce,
    subtract), and XLA's algebraic simplifier does not reliably cancel a
    ``1 - (1 - x)`` pattern when a ``reduce`` sits between the two
    subtracts, even when the reduce is over a size-1 axis -- confirmed via
    ``jax.jit(...).lower(...).compile().cost_analysis()``, which showed the
    general path costing 2 * total_pixels more flops than
    ``_compute_planet_mask`` alone at nplanet=1 (the two subtract passes
    survive compilation; only the reduce itself gets optimised away). Since
    ``planet_xyz.shape[0]`` is a static (trace-time) Python int -- JAX array
    shapes are always static even when the values themselves are traced --
    branching on it here is safe under jit/vmap/grad and costs nothing at
    trace time; it just avoids ever constructing the redundant ops for the
    common single-planet case, rather than hoping the compiler removes them.

    Parameters
    ----------
    x_disc, y_disc  : in-disc pixel coordinates  [pixels]
    star_pixel_rad  : stellar radius in pixels
    planet_xyz      : (nplanet, 3) planet sky positions [X, Y, Z] in R*
    k               : (nplanet,) planet-to-star radius ratios
    softness        : sigmoid transition width [R*] (default 0.0: hard edge)

    Returns
    -------
    jnp.ndarray, shape (total_pixels,), dtype float32, values in [0, 1]
    """
    if planet_xyz.shape[0] == 1:
        return _compute_planet_mask(
            x_disc, y_disc, star_pixel_rad,
            planet_xyz[0, 0], planet_xyz[0, 1], planet_xyz[0, 2], k[0],
            softness,
        )

    masks = vmap(
        lambda xyz, kk: _compute_planet_mask(
            x_disc, y_disc, star_pixel_rad, xyz[0], xyz[1], xyz[2], kk,
            softness,
        )
    )(planet_xyz, k)   # (nplanet, total_pixels)

    return 1.0 - jnp.prod(1.0 - masks, axis=0)


# ---------------------------------------------------------------------------
# 5. build_transit_model — pre-compute positions for all (oversampled) epochs
# ---------------------------------------------------------------------------

def _warn_if_precision_insufficient(times: np.ndarray) -> None:
    """
    Warn when float32 rounding of ``times`` could blur the sampling cadence.

    The mean anomaly (``planet_sky_position``) is formed from a cancelling
    subtraction ``time - t_peri``: at BJD scales (~2.4e6) float32's 24
    mantissa bits leave an absolute rounding error of a large fraction of a
    day, which can be comparable to or larger than the sampling cadence.
    ``build_transit_model`` does not shift ``times``/``t0`` itself i.e. this 
    warning fires whenever the *given* ``times`` are risky, regardless of whether
    the caller has already reduced them.
    """
    if jax.config.jax_enable_x64:
        return
    if isinstance(times, jax.core.Tracer):
        # Can't concretize a traced times array to diagnose it -- skip
        # rather than error, so build_transit_model stays traceable under
        # jit/vmap. This warning is advisory, not correctness-critical.
        return
    times64 = np.asarray(times, dtype=np.float64)
    unique_times = np.unique(times64)
    if unique_times.size < 2: # no two distinct epochs -- no cadence to compare against
        return
    cadence = np.min(np.diff(unique_times))
    rounding_error = np.max(np.abs(times64.astype(np.float32).astype(np.float64) - times64))
    if rounding_error > 0.1 * cadence: # warn if rounding error is 10% of mininum cadence
        warnings.warn(
            "SAJAX: `times` span values large enough (max |t| = "
            f"{np.max(np.abs(times64)):.6g}) that float32 rounding "
            f"(~{rounding_error:.3g} time units) is a significant fraction "
            f"of the minimum sampling cadence (~{cadence:.3g} time units). This can "
            "bias or wash out the transit shape. Enable double precision "
            "with `jax.config.update(\"jax_enable_x64\", True)` before "
            "using SAJAX, or subtract a reference epoch (e.g. "
            "`times - times.min()`, adjusting `t0` to match) before calling "
            "build_transit_model directly. Note that build_system already "
            "does this for you. See SAJAX documentation for more info.",
            stacklevel=3,
        )


def build_transit_model(
    times: np.ndarray,
    t0: float | np.ndarray,
    period: float | np.ndarray,
    a_over_rstar: float | np.ndarray,
    inclination: float | np.ndarray,
    ecc: float | np.ndarray        = 0.0,
    omega_peri: float | np.ndarray = 0.0,
    k: float | np.ndarray          = 0.1,
    sp_orb: float | np.ndarray     = 0.0,
) -> dict:
    """
    Pre-compute every planet's sky-plane position at every epoch in ``times``.

    The returned dict should be stored in the sajax model dict under the key
    ``"transit"``.  ``build_system()`` (in core.py) does this automatically
    when its transit parameters are given — end users typically do not need
    to call this function directly.

    Parameters
    ----------
    times         : (ntime,) array of observation epochs  [days]
                    Must be the **oversampled** time array when oversampling
                    is active (see ``build_system``).
    t0            : mid-transit epoch(s)  [days] -- scalar or (nplanet,).
                    ``nplanet`` is inferred from ``t0``'s trailing axis.
    period        : orbital period(s)  [days] -- scalar or (nplanet,)
    a_over_rstar  : semimajor axis / R*  (dimensionless) -- scalar or (nplanet,)
    inclination   : orbital inclination  [rad] -- scalar or (nplanet,)
    ecc           : eccentricity  (default: 0.0 = circular) -- scalar or (nplanet,)
    omega_peri    : argument of periastron  [rad]  (default: 0.0) -- scalar or (nplanet,)
    k             : planet-to-star radius ratio  Rp / R*  (default: 0.1).
                    The orbital position doesn't depend on k at all, so this
                    is stored as-is (not shape-validated against nplanet/nwave
                    here -- that happens one level up in core.py's
                    build_system/make_lc, which are the only callers that
                    know both nplanet and nwave) for later use by the
                    per-wavelength occultation mask in core.py.
    sp_orb        : sky-projected spin-orbit angle λ [rad] (default: 0.0) --
                    scalar or (nplanet,). See ``planet_sky_position``.

    Any scalar (or size-1 array) among t0/period/a_over_rstar/inclination/
    ecc/omega_peri/sp_orb broadcasts to all nplanet planets; a differently-
    sized array raises ``ValueError`` (see
    ``compute_multi_planet_sky_positions``).

    Numerical precision
    --------------------
    This function does **not** shift ``times``/``t0`` itself -- ``build_system``
    (in core.py) already subtracts a common reference epoch (``model["t_ref"]``)
    from both before ever casting either to a JAX array, so absolute BJD-scale
    epochs reaching this function via the normal two-stage API are already small.
    Direct/standalone callers passing raw BJD-scale ``times``/``t0`` are
    responsible for doing the same reduction themselves -- a warning fires
    (see ``_warn_if_precision_insufficient``) when this hasn't been done
    and float32 rounding is a meaningful fraction of the sampling cadence.

    Returns
    -------
    dict with keys
    ~~~~~~~~~~~~~~
    ``planet_xyz`` : (ntime, nplanet, 3) jnp.ndarray — each planet's (X, Y, Z) per epoch
    ``k``          : jnp.ndarray, as given — planet-to-star radius ratio
    ``nplanet``    : int — number of planets, inferred from t0's trailing axis
    """
    _warn_if_precision_insufficient(times)
    times_jax = jnp.asarray(times)

    xyz = compute_multi_planet_sky_positions(
        times_jax, t0, period, a_over_rstar, inclination, ecc, omega_peri,
        sp_orb,
    )   # (ntime, nplanet, 3)

    return dict(
        planet_xyz = xyz,
        k          = jnp.asarray(k, dtype=jnp.float32),
        nplanet    = int(jnp.atleast_1d(jnp.asarray(t0)).shape[-1]),
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