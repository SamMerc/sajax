# Installation

To install *sajax* from pypi
    
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