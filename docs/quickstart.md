# Quick Start

## Overview

SAJAX computes light curves that include stellar contamination from active regions (spots and faculae), and can optionally include a planetary transit.
It takes stellar, active region, and orbital parameters, as well as timing information to produce wavelength-resolved light curves.

## Basic Workflow

1. **Define spectra** — provide quiet-star and active-region flux as functions of wavelength
2. **Set stellar parameters** — inclination, rotation velocity, limb darkening
3. **Define active regions** *(optional)* — positions, sizes, and smoothness
4. **Define a transit** *(optional)* — orbital geometry and planet-to-star radius ratio
5. **Compute the light curve** — SAJAX returns disk-integrated flux at each requested time, per wavelength bin

### Two-stage API

- `build_system(...)` — call **once**. Builds the stellar grid and every array that stays fixed across model calls (spectra, grid, and, if given, the transit geometry).
- `make_lc(model, ...)` — call at **every step** to produce the light curve(s). Pure JAX - accepts tracers, so it's compatible with `jit`/`vmap`/gradient-based samplers.

`quick_lc(...)` wraps both stages in a single call — convenient for quick, one-off evaluations.

## Key Inputs

| Parameter | Description | Example |
|-----------|-------------|---------|
| `wavelength` | Wavelength grid [μm] | `jnp.linspace(0.3, 5.0, 200)` |
| `flux_quiet` | Quiet star spectrum | Model atmosphere or measured spectrum |
| `flux_active` | Active region spectrum | Cooler (dimmer) or hotter (brighter) than quiet star |
| `ld_coeffs`, `ld_mode` | Quiet-photosphere limb-darkening coefficients and law | `[0.5, 0.2] #for [u1, u2]`, `"quadratic"` |
| `inc_star` | Stellar inclination [deg] | `90.0` (equator-on) |
| `ve` | Stellar equatorial rotational velocity [km/s] | `2.0` |
| `ar_lat`, `ar_long` | Active region latitude/longitude [deg] | `20.0`, `0.0` |
| `ar_size` | Angular radius of active region [deg] | `10.0` |
| `ar_smoothness` | Super-Gaussian order of the AR edge (higher = sharper) | `4.0` |
| `times` | Absolute observation times [days] | `jnp.linspace(0, 10, 50, endpoint=False)` |
| `P_rot` | Stellar rotation period [days] | `10.0` |
| `stellar_grid_size` | Stellar grid radius [pixels] | `100` |
| `t0`, `period`, `a_over_rstar`, `inclination`, `k` | Transit geometry (all-or-nothing) — mid-transit epoch [days], orbital period [days], a/R\*, orbital inclination [rad], and Rp/R\* | `k=0.1` |
| `ecc`, `omega_peri` | Orbital eccentricity and argument of periastron [rad] *(optional, default circular)* | `0.0`, `0.0` |
| `sp_orb` | Sky-projected spin-orbit angle λ [deg] — rotates the transit chord relative to the stellar spin axis *(optional, default 0.0 = aligned)* | `90.0` (polar) |

`ar_lat`/`ar_long`/`ar_size`/`ar_smoothness`/`flux_active` are all-or-nothing: give every one of them to add active region(s), or omit all of them for a quiet star. Each carries a trailing `(nar,)` axis (`nar` inferred from `ar_lat`), so `ar_lat=[20.0, -20.0]` etc. adds two active regions.

`t0`/`period`/`a_over_rstar`/`inclination`/`k` are likewise all-or-nothing for a transit, and each is scalar or carries a trailing `(nplanet,)` axis (`nplanet` inferred from `t0`) — see case 3c below. `k` may be a scalar (the same value for every planet/wavelength), an array of shape `(nplanet,)` (one achromatic value per planet), or an array of shape `(nplanet, nwave)` (a chromatic transit depth, independent per planet); for a single planet, a bare `(nwave,)` array is also still accepted as that planet's chromatic depth.

## Key Outputs

`make_lc`/`quick_lc` return a `(lc, star_maps)` tuple:

| Output | Shape | Description |
|--------|-------|-------------|
| `lc` | `(n_times, nwave)`, or `(n_times,)` if `nwave == 1` | Light curve (wavelength-resolved when `nwave > 1`) in the same units as `flux_quiet`/`flux_active` |
| `star_maps` | `(n_times, n_px, n_px)` | 2D flux maps of the star (and transiting planet, if present) at each time |

## Limb-Darkening Modes

SAJAX supports multiple limb-darkening laws:

- `linear` — 1 coefficient
- `quadratic` — 2 coefficients (most common)
- `power2`, `kipping3` — alternative parameterizations
- `nonlinear4` — 4-coefficient law
- `intensity_profile` — full I(μ) profile

## Common Use Cases

### Case 1: Single active region

```python
import jax.numpy as jnp
from sajax import quick_lc

# Wavelength grid (e.g. in microns)
wavelength  = jnp.linspace(0.3, 5.0, 200)

# Flat spectra as a minimal example — replace with model atmospheres
flux_quiet    = jnp.ones_like(wavelength)
flux_active   = jnp.ones_like(wavelength) * 0.7   # active region is 30% dimmer

P_ROT = 10.0                                        # stellar rotation period [days]
times = jnp.linspace(0, P_ROT, 50, endpoint=False)   # one full rotation

lc, star_maps = quick_lc(
    wavelength         = wavelength,
    flux_quiet         = flux_quiet,
    flux_active        = flux_active,
    ld_coeffs          = [0.3, 0.1],       # quadratic law: [u1, u2]
    inc_star           = 90.0,             # stellar inclination [deg]  (equator-on)
    ar_lat             = 20.0,           # one active region at 20° latitude
    ar_long            = 0.0,
    ar_size            = 10.0,           # angular radius [deg]
    ar_smoothness      = 4.0,            # super-Gaussian edge order
    times              = times,
    P_rot              = P_ROT,
    stellar_grid_size  = 100,              # stellar radius in pixels
    ve                 = 2.0,              # equatorial velocity [km/s]
    ld_mode            = "quadratic",      # treatment of limb darkening
    plot_map_wavelength= 1.0,
)
```

### Case 2: Multiple active regions

Replace

```python
    ar_lat             = 20.0,           # one active region at 20° latitude
    ar_long            = 0.0,
    ar_size            = 10.0,           # angular radius [deg]
    ar_smoothness      = 4.0,
```

in the previous code, with

```python
    ar_lat             = [20.0, -45.0],          # two active regions at 20° and -45° latitude
    ar_long            = [0.0, 15.0],
    ar_size            = [10.0, 5.0],            # angular radius [deg]
    ar_smoothness      = [4.0, 1.0],             # sharp-edged spot, soft-edged facula
```

Active regions overlap additively — e.g. an umbra sitting inside a penumbra contributes on top of it, rather than one masking the other.

### Case 3: Planetary transit
#### a) Basic transit + stellar activity

Transit parameters are individual keyword arguments, given all together alongside the active-region ones above:

```python
lc, star_maps = quick_lc(
    wavelength         = wavelength,
    flux_quiet         = flux_quiet,
    flux_active        = flux_active,
    ld_coeffs          = [0.3, 0.1],
    inc_star           = 90.0,
    ar_lat             = 20.0,
    ar_long            = 0.0,
    ar_size            = 10.0,
    ar_smoothness      = 4.0,
    times              = times,
    P_rot              = P_ROT,
    stellar_grid_size  = 100,
    ve                 = 2.0,
    ld_mode            = "quadratic",
    t0                 = 5.0,     # mid-transit epoch [days]
    period             = 3.5,     # orbital period [days]
    a_over_rstar       = 15.0,    # a / R*
    inclination        = 1.55,    # orbital inclination [rad], ~edge-on
    k                  = 0.1,     # Rp / R*
)
```

By default the occultation mask has a hard edge, which gives `jax.grad` (almost) zero gradient with respect to the transit-geometry parameters. For gradient-based retrieval of `t0`/`period`/`a_over_rstar`/`inclination`/`k`/`ecc`/`omega_peri`/`sp_orb` (_e.g._, a gradient-descent MAP approach), pass `transit_softness > 0` to `make_lc` (not exposed on `quick_lc`) — see the `inference.ipynb` example notebook for a full walkthrough.

#### b) Misaligned transit (spin-orbit angle)

SAJAX fixes the stellar spin axis along the sky's north-south direction, so by default (`sp_orb=0.0`) the transit chord runs parallel to the projected stellar equator. Passing `sp_orb` rotates the chord about the stellar centre, letting you model misaligned or even polar configurations:

```python
lc, star_maps = quick_lc(
    wavelength         = wavelength,
    flux_quiet         = flux_quiet,
    flux_active        = flux_active,
    ld_coeffs          = [0.3, 0.1],
    inc_star           = 90.0,
    ar_lat             = 20.0,
    ar_long            = 0.0,
    ar_size            = 10.0,
    ar_smoothness      = 4.0,
    times              = times,
    P_rot              = P_ROT,
    stellar_grid_size  = 100,
    ve                 = 2.0,
    ld_mode            = "quadratic",
    t0                 = 5.0,
    period             = 3.5,
    a_over_rstar       = 15.0,
    inclination        = 1.55,
    k                  = 0.1,
    sp_orb             = 90.0,    # polar transit, 90° from aligned [deg]
)
```

Since the active region's latitude/longitude are unaffected by `sp_orb` — only the planet's trajectory rotates — a spot that produces a clear crossing anomaly at `sp_orb=0` can end up entirely missed by a polar (`sp_orb≈90`) chord, even though the transit depth and duration are unchanged. See `introduction.ipynb`'s Case 6 for a full side-by-side comparison (light curve + stellar-disc animation) of an aligned vs. inclined transit of the same spot.

#### c) Multiple planets

Every transit-geometry parameter (`t0`/`period`/`a_over_rstar`/`inclination`/`k`, plus the optional `ecc`/`omega_peri`/`sp_orb`) carries a trailing `(nplanet,)` axis, exactly like `ar_lat`/`ar_long`/etc. do for active regions — `nplanet` is inferred from `t0`. A scalar (or size-1 array) among the others broadcasts to every planet. Replace

```python
    t0                 = 5.0,
    period             = 3.5,
    a_over_rstar       = 15.0,
    inclination        = 1.55,
    k                  = 0.1,
```

in Case 3's snippet, with

```python
    t0                 = [5.0, 8.2],       # two planets' mid-transit epochs [days]
    period             = [3.5, 11.0],
    a_over_rstar       = [15.0, 22.0],
    inclination        = [1.55, 1.52],     # rad
    k                  = [0.1, 0.05],      # Rp/R* per planet
```

Unlike active regions — which are spectral-contrast modulators and so overlap **additively** (Case 2) — planets are opaque occulters, so overlapping transits combine **multiplicatively**: the fraction of a pixel's flux surviving `nplanet` planets is `prod(1 - mask_i)`, not `1 - sum(mask_i)`. This keeps the occulted flux fraction physically bounded in `[0, 1]` even when two planets' discs overlap on the stellar grid, rather than letting their masks double-subtract past full occultation.

### Case 5: Time-evolving active regions

`flux_active`/`ar_lat`/`ar_long`/`ar_size`/`ar_smoothness` may each independently carry an extra leading time axis (length = the number of `times` the model was built with) instead of their usual per-AR shape, to let that property evolve over the observations — a spot growing/decaying, drifting in latitude/longitude, or changing contrast:

```python
import numpy as np

ntime = len(times)
ar_size_evolving = np.linspace(5.0, 15.0, ntime)[:, None]   # (ntime, nar=1): a growing spot

lc, star_maps = quick_lc(
    wavelength         = wavelength,
    flux_quiet         = flux_quiet,
    flux_active        = flux_active,
    ld_coeffs          = [0.3, 0.1],
    inc_star           = 90.0,
    ar_lat             = 20.0,             # static: same shape as before
    ar_long            = 0.0,
    ar_size            = ar_size_evolving, # time-varying: (ntime, nar)
    ar_smoothness      = 4.0,
    times              = times,
    P_rot              = P_ROT,
    stellar_grid_size  = 100,
    ve                 = 2.0,
    ld_mode            = "quadratic",
    ar_time_interp     = "linear",         # or "cubic" -- see below
)
```

Only the parameters you want to evolve need the extra axis — the rest keep their usual constant-in-time shape. Values are given per `times` entry, *not* per oversampled sub-exposure. When `oversample > 1`, each evolving property is resolved onto the exact sub-exposure times through interpolation: `ar_time_interp="linear"` (the default) for piecewise-linear, or `"cubic"` for a C2 natural cubic spline (matching `scipy.interpolate.CubicSpline(bc_type="natural")`, but implemented in JAX via [`interpax`](https://github.com/f0uriest/interpax) so it stays differentiable). `ar_time_interp` is fixed at `build_system`/`quick_lc` time, like `ld_mode`. `ld_coeffs_active`/`I_profile_active` don't support time evolution and always stay fixed. Importantly, if none of the five parameters are given a time axis, `make_lc` takes the exact same code path as the static case, such that this feature adds no computational cost unless activated.

Each `times` entry's active-region values are used exactly as given — the forward model does **not** couple them across epochs (no smoothness/continuity is enforced between one entry and the next). This is deliberate: it keeps the model as a per-epoch evaluation, allowing users full control over the dynamics. It also means nothing stops a naive fit from letting an active region changing unphysically between epochs. If you're doing inference on a dynamic active region, it's up to you to keep that from happening — _e.g._ with priors on the epoch-to-epoch differences (or on physical rates, like a maximum drift speed) that prevent implausible jumps.

### Case 6: Flares

A flare is modeled with Case 5's time-evolving machinery: an active region whose *spectrum* varies in time. Combine two ingredients — a flare spectrum (e.g. a hot ~9,000 K blackbody sampled on the model's `wavelength` grid) and a time template. Users can create their own or use the one of [Tovar Mendoza et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022AJ....164...17T/abstract), available as `sajax.flare_template(t, tpeak, fwhm, ampl)` (peak time, FWHM in the units of `t`, peak amplitude; JAX-native and differentiable):

```python
from sajax import flare_template

template = flare_template(times, tpeak=0.1, fwhm=0.01)       # (ntime,)
flux_flaring = flux_quiet + template[:, None] * flux_flare   # (ntime, nwave)
```

Pass `flux_flaring` as that region's time-varying `flux_active` (shape `(ntime, nar, nwave)`, Case 5): its contrast rises and decays following the template, and outside the flare the contrast is exactly 1, so the region vanishes. Since flare emission is chromospheric rather than photospheric, we recommend users turn off limb darkening for that region (zero coefficients in `ld_coeffs_active`). Because the flare lives on the stellar surface, it is foreshortened toward the limb, carried by rotation, and occulted by a transiting planet crossing it. See `introduction.ipynb`'s Case 9 for a full worked example.

### Numerical precision for long baselines (absolute BJD times)

JAX defaults to `float32`. If you pass absolute epochs in BJD (e.g. `2460123.45`) `float32`'s 24 mantissa bits leave a rounding error (on the order of hours) significant enough to blur or bias the transit shape. To avoid this issue, `build_system` **automatically** subtracts a reference epoch (`model["t_ref"] = floor(times.min())`) so any downstream numerical work benefits from the smaller magnitude. If the rounding error is still a meaningful fraction of your sampling cadence *after* this automatic shift, SAJAX emits a warning, and `jax_enable_x64` remains available as a fallback for that case:

```python
import jax
jax.config.update("jax_enable_x64", True)

from sajax import build_system, make_lc   # import sajax only after this
...
```

## Next Steps

- 💾 **[Explore Tutorials](https://sajax.readthedocs.io/en/latest/examples/introduction.html)** — Check out full working examples with colorful plots, interesting use cases, and the full implementation of both an MCMC and a gradient-based retrieval!

- 📚 **{doc}`Read the API Reference <autoapi/index>`** — Learn about all available functions, classes, and parameters