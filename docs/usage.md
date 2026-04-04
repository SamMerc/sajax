# Usage Guide

## Overview

SAJAX computes light curves that include stellar contamination from active regions (spots and faculae). 
It takes stellar spectra, active region properties, and rotation parameters to produce wavelength-dependent light curves.

## Basic Workflow

1. **Define spectra** — provide quiet star and active region flux as functions of wavelength
2. **Set stellar parameters** — inclination, rotation velocity, limb-darkening
3. **Define active regions** — spot/faculae positions, sizes, and temperatures
4. **Compute light curve** — SAJAX returns flux as a function of rotational phase

## Key Inputs

| Parameter | Description | Example |
|-----------|-------------|---------|
| `wavelength` | Wavelength grid [μm] | `np.linspace(0.3, 5.0, 200)` |
| `flux_quiet` | Quiet star spectrum | Model atmosphere or measured spectrum |
| `flux_active` | Active region spectrum | Cooler (dimmer) or Hotter (brighter) than quiet star |
| `ar_lat`, `ar_long` | Active region latitude/longitude [deg] | `[20.0]`, `[0.0]` |
| `ar_size` | Angular radius of active region [deg] | `[10.0]` |
| `phases_rot` | Rotation phases [deg] | `np.linspace(0, 360, 50)` |
| `inc_star` | Stellar inclination [deg] | `90.0` (equator-on) |
| `ldc_mode` | Limb-darkening law or Intensity Profile| `"quadratic"`, `"nonlinear4"`, etc. or `"intensity_profile"`|

## Key Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| `lc` | `(n_phases,)` | Broadband light curve (integrated over wavelength) |
| `epsilon` | `(n_phases, n_wavelengths)` | Contamination factor ε — how much active region contaminates each phase/wavelength |
| `star_maps` | `(n_phases, n_px, n_px)` | 2D intensity maps of the star at each phase |

## What is Contamination?

The contamination factor **ε** quantifies the flux deficit due to active regions:

$$\text{observed flux} = \text{true flux} \times (1 - \varepsilon)$$

Key insights:
- **ε = 0** → no contamination (quiet star)
- **ε > 0** → dimmer observed flux (active regions crossing disk)
- **ε wavelength-dependent** → different limb-darkening at different wavelengths
- **ε phase-dependent** → changes as star rotates

## Limb-Darkening Modes

SAJAX supports multiple limb-darkening laws:

- `linear` — 1 coefficient
- `quadratic` — 2 coefficients (most common)
- `power2`, `kipping3` — alternative parameterizations
- `nonlinear4` — 4-coefficient law
- `intensity_profile` — full I(μ) profile

## JAX Integration

SAJAX is fully differentiable — use JAX transforms for inference:

```python
import jax
from jax import jit, vmap, grad

# Compile for speed
compute_lc_jit = jit(compute_light_curve)

# Vectorize over active region parameters
compute_lc_batch = vmap(compute_light_curve, in_axes=(None, None, None, 0, ...))

# Compute gradients for inference
dlc_du = grad(lambda u: compute_light_curve(..., u1=u)["lc"].sum())
```

## Common Use Cases

### Case 1: Single active region

```python
import numpy as np
from sajax import compute_light_curve

# Wavelength grid (e.g. in microns)
wavelength  = np.linspace(0.3, 5.0, 200)

# Flat spectra as a minimal example — replace with model atmospheres
flux_quiet    = np.ones_like(wavelength)
flux_active   = np.ones_like(wavelength) * 0.7   # active region is 30% dimmer

params = dict(
    u1          = 0.3,      # quadratic LD coefficient
    u2          = 0.1,
    inc_star    = 90.0,     # stellar inclination [deg]  (equator-on)
)

result = compute_light_curve(
    wavelength         = wavelength,
    flux_quiet         = flux_quiet,
    flux_active        = flux_active,
    params             = params,
    ar_lat             = [20.0],           # one active region at 20° latitude
    ar_long            = [0.0],
    ar_size            = [10.0],           # angular radius [deg]
    phases_rot         = np.linspace(0, 360, 50, endpoint=False),
    planet_pixel_size  = 20,
    ve                 = 2.0,              # equatorial velocity [km/s]
    ldc_mode           = "Quadratic",      #treatment of limb darkening
    plot_map_wavelength= 1.0,
)

print(result["lc"])          # (50,)      broadband light curve
print(result["epsilon"])     # (50, 200)  contamination factor per phase/wavelength
print(result["star_maps"])   # (50, n, n) stellar flux maps
```

### Case 2: Multiple active regions

Replace 
```
    ar_lat             = [20.0],           # one active region at 20° latitude
    ar_long            = [0.0],
    ar_size            = [10.0],           # angular radius [deg]
```

in the previous code, with 
```
    ar_lat             = [20.0, -45.0],          # two active regions at 20° and -45° latitude
    ar_long            = [0.0, 15.0],
    ar_size            = [10.0, 5.0],           # angular radius [deg]
```

## Next Steps

Check Tutorials or the examples/ directory for a full working example with plots

Read the API Reference for all parameters

Explore JAX Integration for gradient-based inference