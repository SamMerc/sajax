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
    flux_hot  = np.ones_like(wl)
    flux_cold = np.full_like(wl, 0.7)
    return wl, flux_hot, flux_cold


@pytest.fixture
def base_params():
    return dict(
        u1       = 0.3,
        u2       = 0.1,
        inc_star = 90.0,
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
    wl, flux_hot, flux_cold = flat_spectra
    result = compute_light_curve(
        wavelength         = wl,
        flux_hot           = flux_hot,
        flux_cold          = flux_cold,
        params             = base_params,
        spot_lat           = [20.0],
        spot_long          = [0.0],
        spot_size          = [10.0],
        phases_rot         = [0.0],
        stellar_grid_size  = 50,
        ve                 = 2.0,
        ldc_mode           = "multi-color",
    )
    assert result["lc"].shape == (1,)
    assert result["epsilon"].shape == (1, len(wl))
    assert result["star_maps"].ndim == 3


# ---------------------------------------------------------------------------
# Multi-phase output shapes
# ---------------------------------------------------------------------------

def test_output_shapes_multi_phase(flat_spectra, base_params):
    wl, flux_hot, flux_cold = flat_spectra
    phases = np.linspace(0, 360, 8, endpoint=False)
    result = compute_light_curve(
        wavelength         = wl,
        flux_hot           = flux_hot,
        flux_cold          = flux_cold,
        params             = base_params,
        spot_lat           = [20.0],
        spot_long          = [0.0],
        spot_size          = [10.0],
        phases_rot         = phases,
        stellar_grid_size  = 50,
        ve                 = 2.0,
    )
    assert result["lc"].shape == (8,)
    assert result["epsilon"].shape == (8, len(wl))


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------

def test_no_spot_flux_is_unity(flat_spectra, base_params):
    """With a vanishingly small spot the light curve should be ~1."""
    wl, flux_hot, flux_cold = flat_spectra
    result = compute_light_curve(
        wavelength         = wl,
        flux_hot           = flux_hot,
        flux_cold          = flux_cold,
        params             = base_params,
        spot_lat           = [0.0],
        spot_long          = [0.0],
        spot_size          = [0.001],   # effectively zero spot
        phases_rot         = [0.0],
        stellar_grid_size  = 50,
        ve                 = 0.0,
        ldc_mode           = "single",
    )
    assert abs(float(result["lc"][0]) - 1.0) < 0.01


def test_cold_spot_dims_flux(flat_spectra, base_params):
    """A visible spot with cold spectrum should reduce the total flux."""
    wl, flux_hot, flux_cold = flat_spectra
    result = compute_light_curve(
        wavelength         = wl,
        flux_hot           = flux_hot,
        flux_cold          = flux_cold,
        params             = base_params,
        spot_lat           = [0.0],
        spot_long          = [0.0],
        spot_size          = [20.0],    # large visible spot
        phases_rot         = [0.0],
        stellar_grid_size  = 50,
        ve                 = 0.0,
        ldc_mode           = "single",
    )
    assert float(result["lc"][0]) < 1.0


def test_multi_spot(flat_spectra, base_params):
    """Two spots should work without error and return sensible shapes."""
    wl, flux_hot, flux_cold = flat_spectra
    result = compute_light_curve(
        wavelength         = wl,
        flux_hot           = flux_hot,
        flux_cold          = flux_cold,
        params             = base_params,
        spot_lat           = [20.0, -20.0],
        spot_long          = [0.0,  180.0],
        spot_size          = [10.0,  10.0],
        phases_rot         = np.linspace(0, 360, 6, endpoint=False),
        stellar_grid_size  = 50,
        ve                 = 2.0,
    )
    assert result["lc"].shape == (6,)


# ---------------------------------------------------------------------------
# LDC modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ldc_mode", ["single", "multi-color"])
def test_ldc_modes(flat_spectra, base_params, ldc_mode):
    wl, flux_hot, flux_cold = flat_spectra
    result = compute_light_curve(
        wavelength        = wl,
        flux_hot          = flux_hot,
        flux_cold         = flux_cold,
        params            = base_params,
        spot_lat          = [15.0],
        spot_long         = [0.0],
        spot_size         = [8.0],
        phases_rot        = [0.0, 90.0],
        stellar_grid_size = 50,
        ve                = 1.0,
        ldc_mode          = ldc_mode,
    )
    assert result["lc"].shape == (2,)
    assert np.all(np.isfinite(result["lc"]))
    assert np.all(np.isfinite(result["epsilon"]))