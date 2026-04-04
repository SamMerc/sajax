"""
tests/test_core.py — Smoke tests for the SAJAX core engine.

Run with:
    pytest tests/
"""

import numpy as np
import pytest

from sajax import compute_light_curve, build_stellar_grid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_spectra():
    """Flat spectra on a small wavelength grid — fast for tests."""
    wl       = np.linspace(0.5, 2.5, 30, dtype=np.float32)
    flux_quiet  = np.ones_like(wl)
    flux_active = np.full_like(wl, 0.7)
    return wl, flux_quiet, flux_active


@pytest.fixture
def base_params():
    return dict(
        ldc_coeffs = [0.3, 0.1],  # quadratic law: [u1, u2]
        inc_star   = 90.0,
    )


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def test_build_stellar_grid_shapes():
    grid = build_stellar_grid(
        stellar_grid_size=50,
        ve=2.0,
    )
    n = grid["n"]
    assert grid["n"] == 101
    assert grid["star_pixel_rad"] == 50
    assert grid["total_pixels"] > 0
    assert (len(grid["x"]) == grid["total_pixels"])
    assert (len(grid["y"]) == grid["total_pixels"])
    assert (len(grid["mu"]) == grid["total_pixels"])
    assert (len(grid["vel"]) == grid["total_pixels"])


def test_build_stellar_grid_mu_range():
    grid = build_stellar_grid(50, 2.0)
    mu = grid["mu"]
    assert float(mu.min()) >= 0.0
    assert float(mu.max()) <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Single-phase output shapes
# ---------------------------------------------------------------------------

def test_output_shapes_single_phase(flat_spectra, base_params):
    wl, flux_quiet, flux_active = flat_spectra
    result = compute_light_curve(
        wavelength         = wl,
        flux_quiet           = flux_quiet,
        flux_active          = flux_active,
        params             = base_params,
        ar_lat           = [20.0],
        ar_long          = [0.0],
        ar_size          = [10.0],
        phases_rot         = [0.0],
        stellar_grid_size  = 50,
        ve                 = 2.0,
        ldc_mode="quadratic",
    )
    assert result["lc"].shape == (1,)
    assert result["epsilon"].shape == (1, len(wl))
    assert result["star_maps"].ndim == 3


# ---------------------------------------------------------------------------
# Multi-phase output shapes
# ---------------------------------------------------------------------------

def test_output_shapes_multi_phase(flat_spectra, base_params):
    wl, flux_quiet, flux_active = flat_spectra
    phases = np.linspace(0, 360, 8, endpoint=False)
    result = compute_light_curve(
        wavelength         = wl,
        flux_quiet           = flux_quiet,
        flux_active          = flux_active,
        params             = base_params,
        ar_lat           = [20.0],
        ar_long          = [0.0],
        ar_size          = [10.0],
        phases_rot         = phases,
        stellar_grid_size  = 50,
        ve                 = 2.0,
    )
    assert result["lc"].shape == (8,)
    assert result["epsilon"].shape == (8, len(wl))


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------

def test_no_ar_flux_is_unity(flat_spectra, base_params):
    """With a vanishingly small active region the light curve should be ~1."""
    wl, flux_quiet, flux_active = flat_spectra
    result = compute_light_curve(
        wavelength         = wl,
        flux_quiet           = flux_quiet,
        flux_active          = flux_active,
        params             = base_params,
        ar_lat           = [0.0],
        ar_long          = [0.0],
        ar_size          = [0.001],   # effectively zero active region
        phases_rot         = [0.0],
        stellar_grid_size  = 50,
        ve                 = 0.0,
        ldc_mode="quadratic",
    )
    assert abs(float(result["lc"][0]) - 1.0) < 0.01


def test_cold_ar_dims_flux(flat_spectra, base_params):
    """A visible active region with cold spectrum should reduce the total flux."""
    wl, flux_quiet, flux_active = flat_spectra
    result = compute_light_curve(
        wavelength         = wl,
        flux_quiet           = flux_quiet,
        flux_active          = flux_active,
        params             = base_params,
        ar_lat           = [0.0],
        ar_long          = [0.0],
        ar_size          = [20.0],    # large visible active region
        phases_rot         = [0.0],
        stellar_grid_size  = 50,
        ve                 = 0.0,
        ldc_mode="quadratic",
    )
    assert float(result["lc"][0]) < 1.0


def test_multi_ar(flat_spectra, base_params):
    """Two active regions should work without error and return sensible shapes."""
    wl, flux_quiet, flux_active = flat_spectra
    result = compute_light_curve(
        wavelength         = wl,
        flux_quiet           = flux_quiet,
        flux_active          = flux_active,
        params             = base_params,
        ar_lat           = [20.0, -20.0],
        ar_long          = [0.0,  180.0],
        ar_size          = [10.0,  10.0],
        phases_rot         = np.linspace(0, 360, 6, endpoint=False),
        stellar_grid_size  = 50,
        ve                 = 2.0,
    )
    assert result["lc"].shape == (6,)


# ---------------------------------------------------------------------------
# LDC modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ldc_mode,ldc_coeffs", [
    ("linear",      [0.3]),
    ("quadratic",   [0.3, 0.1]),
    ("power2",      [0.4, 0.6]),
    ("kipping3",    [0.2, 0.3, 0.1]),
    ("nonlinear4",  [0.1, 0.2, 0.15, 0.05]),
])
def test_ldc_modes(flat_spectra, base_params, ldc_mode, ldc_coeffs):
    wl, flux_quiet, flux_active = flat_spectra
    params = {**base_params, "ldc_coeffs": ldc_coeffs}
    result = compute_light_curve(
        wavelength        = wl,
        flux_quiet          = flux_quiet,
        flux_active         = flux_active,
        params            = params,
        ar_lat          = [15.0],
        ar_long         = [0.0],
        ar_size         = [8.0],
        phases_rot        = [0.0, 90.0],
        stellar_grid_size = 50,
        ve                = 1.0,
        ldc_mode          = ldc_mode,
    )
    assert result["lc"].shape == (2,)
    assert np.all(np.isfinite(result["lc"]))
    assert np.all(np.isfinite(result["epsilon"]))