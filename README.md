<p align="center">
  <img src="docs/_static/logo.png" width="300">
</p>

# SAJAX — Stellar Activity Grid for Exoplanets in JAX

![Tests](https://github.com/SamMerc/sajax/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/SamMerc/sajax/branch/main/graph/badge.svg)](https://codecov.io/gh/SamMerc/sajax)

SAJAX is a package that models contamination of exoplanet photometric and spectroscopic time series by active regions (spots and faculae) on the stellar surface. Its core functionality builds on [SAGE](https://github.com/chakrah/sage) ([Chakraborty et al. 2024](https://www.aanda.org/articles/aa/abs/2024/05/aa47727-23/aa47727-23.html)), and it draws additional inspiration from [ANTARESS](https://gitlab.unige.ch/spice_dune/antaress) ([Bourrier et al. 2024](https://www.aanda.org/articles/aa/full_html/2024/11/aa49203-24/aa49203-24.html)).

The main innovation over SAGE is that SAJAX vectorises the spatial and spectral loops with `jax.vmap`, making it fast on both CPU and GPU — with no change to the calling code — and fully differentiable, enabling gradient-based inference with tools like NumPyro or Optax.

The full documentation can be found at [sajax.readthedocs.io](http://sajax.readthedocs.io/)

## Installation

```bash
pip install sajax
```

Or in development mode from a local clone. `setup_env.sh` builds a
[uv](https://docs.astral.sh/uv/)-managed `.venv` pinned by `uv.lock`, so every
machine gets identical package versions:

```bash
git clone https://github.com/SamMerc/sajax.git
cd sajax
./setup_env.sh              # auto-detect: GPU if nvidia-smi sees a device
```

Useful flags: `--cpu` / `--gpu` to override the auto-detection (state `--gpu`
explicitly when provisioning on a GPU-less HPC login node), `--docs` to add the
sphinx dependencies, and `--check` to report what is already installed. 

Run commands in the environment with `uv run <cmd>` (e.g. `uv run pytest`), or
activate it with `source .venv/bin/activate`.

## Repository layout

```
sajax/
├── sajax/
│   ├── __init__.py          # public API
│   ├── core.py              # JAX light-curve and radial velocity engine
│   ├── planet.py            # planet orbital dynamics
│   └── geometry.py          # rotation matrices, coordinate transforms
├── docs/
│   └── examples/
│       ├── introduction_lc.ipynb
│       ├── introduction_rv.ipynb
│       ├── comparison.ipynb
│       ├── inference_lc.ipynb
│       ├── inference_rv.ipynb
│       └── inference_combined.ipynb
├── tests/
│   ├── test_core.py
│   └── test_planet.py
├── pyproject.toml
├── .gitignore
└── README.md
```
