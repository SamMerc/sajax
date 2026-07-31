<p align="center">
  <img src="docs/_static/logo.png" width="300">
</p>

# SAJAX — Stellar Activity Grid for Exoplanets in JAX

![Tests](https://github.com/SamMerc/sajax/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/SamMerc/sajax/branch/main/graph/badge.svg)](https://codecov.io/gh/SamMerc/sajax)

SAJAX is a JAX-accelerated reimplementation of
[SAGE](https://github.com/chakrah/sage) ([Chakraborty et al. 2024](https://www.aanda.org/articles/aa/abs/2024/05/aa47727-23/aa47727-23.html)), a code
that models stellar contamination of exoplanet transmission spectra from
active regions (ars, faculae) on the stellar surface.

The key innovation over plain SAGE is that SAJAX vectorises the spectral
loop with `jax.vmap`, making it fast on both CPU and GPU without any
change to the calling code, and fully differentiable — enabling
gradient-based inference with tools like NumPyro or Optax.

Documentation can be found at [sajax.readthedocs.io](http://sajax.readthedocs.io/)

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
sphinx dependencies, and `--check` to report what is already installed. The
CUDA extra is Linux x86_64 only and pulls in ~3 GB of NVIDIA runtime.

Run commands in the environment with `uv run <cmd>` (e.g. `uv run pytest`), or
activate it with `source .venv/bin/activate`.

## Repository layout

```
sajax/
├── sajax/
│   ├── __init__.py          # public API
│   ├── core.py              # JAX light-curve engine
│   ├── planet.py            # planet orbital dynamics
│   ├── geometry.py          # rotation matrices, coordinate transforms
├── docs/
│   ├── quickstart.ipynb
│   ├── comparison.ipynb
│   ├── inference.ipynb
├── tests/
│   ├── test_core.py
│   ├── test_planet.py
├── pyproject.toml
├── .gitignore
└── README.md
```
