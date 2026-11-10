# Contributing to SAJAX

Thanks for your interest in SAJAX. Contributions of all kinds are welcome, including
bug reports, documentation fixes, new examples, performance work, and new
physics.

## Reporting bugs and requesting features

Open an [issue](https://github.com/SamMerc/sajax/issues). For a bug, please
include:

- what you ran (a minimal script or notebook cell that reproduces the problem),
- what you expected and what happened instead (full traceback if there is one),
- your SAJAX version (`python -c "import sajax; print(sajax.__version__)"`),
  Python version, and whether you are using a CPU or a GPU.

JAX-specific gotchas (shape errors under `vmap`, gradient flow problems, dtype
errors) are fair game — specify your JAX/jaxlib version.

## Running the tests

```bash
uv run pytest
```

Code coverage tests are configured in `pyproject.toml` and reported automatically. New
code should come with tests in `tests/`; please make sure the suite passes
before opening a pull request. 

## Pull requests

1. Fork the repository and create a branch off `main`
   (`git checkout -b my-feature`).
2. Make your change, with tests and appropriate documentation.
3. Run `uv run pytest` locally.
4. Open a pull request against `main` describing what the change does and
   why. Link any related issues.

Small, focused pull requests are much easier to review than large ones. If
you are planning a substantial change, it is worth opening an issue first to
discuss the approach.

## Style conventions

- Follow the style of the surrounding code: NumPy-style docstrings, explicit
  array shapes documented in the docstring, and type hints where they help.
- Keep the numerical core in pure JAX: no Python-level branching on traced
  values, no in-place mutation, and prefer `jnp` over `np` inside anything
  that may be `jit`-ed, `vmap`-ed, or differentiated.
- Any new public function should be importable from `sajax` and covered by a
  test that exercises it under `jax.jit` and, where relevant, `jax.grad`.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
