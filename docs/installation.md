# Installation

To install *sajax* from pypi
    
```bash
pip install sajax
```

Or in development mode from a local clone, using the
[uv](https://docs.astral.sh/uv/)-managed environment pinned by `uv.lock`:

```bash
git clone https://github.com/SamMerc/sajax.git
cd sajax
./setup_env.sh              # auto-detect: GPU if nvidia-smi sees a device
```

`setup_env.sh --cpu` and `--gpu` override the auto-detection, `--docs` adds the
sphinx dependencies, and `--check` reports what is already installed. Use the
environment with `uv run <cmd>` or `source .venv/bin/activate`.

**Dependencies:** `numpy`, `jax`, `jaxlib`, `matplotlib`, `scipy`

GPU support comes from the `cuda` extra (`pip install "sajax[cuda]"`, or
`./setup_env.sh --gpu`).
