<p align="center">
  <img src="docs/logo.png" alt="SAJAX logo" width="300"/>
</p>

# SAJAX — Stellar Activity Grid for Exoplanets in JAX

![Tests](https://github.com/SamMerc/sajax/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/SamMerc/sajax/branch/main/graph/badge.svg)](https://codecov.io/gh/SamMerc/sajax)

SAJAX is a JAX-accelerated reimplementation of
[SAGE](https://github.com/chakrah/sage) ([Chakraborty et al. 2024](https://www.aanda.org/articles/aa/abs/2024/05/aa47727-23/aa47727-23.html)), a code
that models stellar contamination of exoplanet transmission spectra from
active regions (spots, faculae) on the stellar surface.

The key innovation over plain SAGE is that SAJAX vectorises the spectral
loop with `jax.vmap`, making it fast on both CPU and GPU without any
change to the calling code, and fully differentiable — enabling
gradient-based inference with tools like NumPyro or Optax.

---

## Features

- **Spectroscopic light curves** — provide a spectrum for the quiet star
  and the active region; SAJAX returns a light curve at every wavelength.
- **Multiple limb-darkening modes** — quadratic coefficients
  (`single`, `multi-color`) or a full intensity profile
  (`intensity_profile`).
- **Stellar rotation + inclination** — arbitrary rotational phases and
  stellar-axis inclinations are supported.
- **JAX-native** — `jit`, `vmap`, and `grad` work out of the box.
- **pip-installable** — clean, modern packaging.

---

## Installation

```bash
pip install sajax
```

Or in development mode from a local clone:

```bash
git clone https://github.com/SamMerc/sajax.git
cd sajax
pip install -e ".[dev]"
```

**Dependencies:** `numpy`, `jax`, `jaxlib`, `matplotlib`, `scipy`

For GPU support install the appropriate `jaxlib` wheel as described in
the [JAX installation guide](https://github.com/google/jax#installation).

---

## Quick start

```python
import numpy as np
from sajax import compute_light_curve

# Wavelength grid (e.g. in microns)
wavelength  = np.linspace(0.3, 5.0, 200)

# Flat spectra as a minimal example — replace with model atmospheres
flux_hot    = np.ones_like(wavelength)
flux_cold   = np.ones_like(wavelength) * 0.7   # spot is 30% dimmer

params = dict(
    radiusratio = 0.1,      # Rp / Rs
    semimajor   = 10.0,     # a / Rs
    u1          = 0.3,      # quadratic LD coefficient
    u2          = 0.1,
    inc_star    = 90.0,     # stellar inclination [deg]  (equator-on)
)

result = compute_light_curve(
    wavelength         = wavelength,
    flux_hot           = flux_hot,
    flux_cold          = flux_cold,
    params             = params,
    spot_lat           = [20.0],           # one spot at 20° latitude
    spot_long          = [0.0],
    spot_size          = [10.0],           # angular radius [deg]
    phases_rot         = np.linspace(0, 360, 50, endpoint=False),
    planet_pixel_size  = 20,
    ve                 = 2.0,              # equatorial velocity [km/s]
    ldc_mode           = "multi-color",
    plot_map_wavelength= 1.0,
)

print(result["lc"])          # (50,)  broadband light curve
print(result["epsilon"])     # (50, 200)  contamination factor per phase/wavelength
print(result["star_maps"])   # (50, n, n) stellar flux maps
```

For fuller examples see the `examples/` directory.

---

## Repository layout

```
sajax/
├── sajax/
│   ├── __init__.py          # public API
│   ├── core.py              # JAX light-curve engine
│   ├── geometry.py          # rotation matrices, coordinate transforms
│   └── interpolation.py     # spectral interpolation helpers
├── examples/
│   └── basic_lightcurve.ipynb
├── tests/
│   └── test_core.py
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Credits

SAJAX is a JAX port of SAGE, originally written by
Hritam Chakraborty (Université de Genève).
The physics and grid approach are described in Chakraborty et al. 2024.

---

## License

MIT
