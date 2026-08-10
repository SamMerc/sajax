# Usage Guide

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
| `stellar_grid_size` | Stellar grid resolution [pixels/side] | `100` |
| `t0`, `period`, `a_over_rstar`, `inclination`, `k` | Transit geometry (all-or-nothing) — mid-transit epoch [days], orbital period [days], a/R\*, orbital inclination [rad], and Rp/R\* | `k=0.1` |
| `ecc`, `omega_peri` | Orbital eccentricity and argument of periastron [rad] *(optional, default circular)* | `0.0`, `0.0` |

`ar_lat`/`ar_long`/`ar_size`/`ar_smoothness`/`flux_active` are all-or-nothing: give every one of them to add active region(s), or omit all of them for a quiet star. `t0`/`period`/`a_over_rstar`/`inclination`/`k` are likewise all-or-nothing for a transit. `k` may also be an array of shape `(nwave,)` for a chromatic transit depth.

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

### Case 3: Adding a planetary transit

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

By default the occultation mask has a hard edge, which gives `jax.grad` (almost) zero gradient with respect to the transit-geometry parameters. For gradient-based retrieval of `t0`/`period`/`a_over_rstar`/`inclination`/`k`/`ecc`/`omega_peri` (_e.g._, a gradient-descent MAP approach), pass `transit_softness > 0` to `make_lc` (not exposed on `quick_lc`) — see the `inference.ipynb` example notebook for a full walkthrough.

## Next Steps

- 💾 **[Explore Tutorials](https://sajax.readthedocs.io/en/latest/examples/quickstart.html)** — Check out full working examples with colorful plots, interesting use cases, and the full implementation of both an MCMC and a gradient-based retrieval!

- 📚 **[Read the API Reference](modules.html)** — Learn about all available functions, classes, and parameters