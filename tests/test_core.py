"""
tests/test_core.py — Tests for the SAJAX core engine.

Run with:
    pytest tests/
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sajax import compute_light_curve, build_stellar_grid
from sajax.core import (
    build_model,
    build_combined_model,
    evaluate_light_curve,
    compute_combined_light_curve,
    _compute_planet_mask,
    _compute_ar_shape,
    _compute_single_phase,
)
from sajax.geometry import rotate_active_region

C_KMS = 299_792.458  # speed of light [km/s]

# Default smoothness used throughout when a test doesn't care about the
# exact shape of the AR boundary -- sharp enough to be visually spot-like.
_SM = 20.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_spectra():
    """Flat spectra on a small wavelength grid — fast for tests.

    Uses float64 throughout to match build_model's internal dtype.
    """
    wl         = np.linspace(500.0, 600.0, 30, dtype=np.float64)
    flux_quiet = np.ones_like(wl)
    flux_active = np.full_like(wl, 0.7)
    return wl, flux_quiet, flux_active


@pytest.fixture
def base_params():
    return dict(
        ldc_coeffs=[0.3, 0.1],   # quadratic law: [u1, u2]
        inc_star=90.0,
    )


@pytest.fixture
def small_model(flat_spectra, base_params):
    """Pre-built model for tests that need the two-stage API."""
    wl, flux_quiet, _ = flat_spectra
    return build_model(
        wavelength=wl,
        flux_quiet=flux_quiet,
        params=base_params,
        phases_rot=np.linspace(0, 360, 8, endpoint=False),
        stellar_grid_size=50,
        ve=2.0,
        ldc_mode="quadratic",
    )


# ===================================================================
# Grid construction
# ===================================================================

class TestBuildStellarGrid:

    def test_shapes(self):
        grid = build_stellar_grid(stellar_grid_size=50, ve=2.0)
        assert grid["n"] == 101
        assert grid["star_pixel_rad"] == 50.0
        assert grid["total_pixels"] > 0
        assert len(grid["x"]) == grid["total_pixels"]
        assert len(grid["y"]) == grid["total_pixels"]
        assert len(grid["mu"]) == grid["total_pixels"]
        assert len(grid["row_idx"]) == grid["total_pixels"]
        assert len(grid["vel_row"]) == grid["n"]

    def test_mu_range(self):
        grid = build_stellar_grid(50, 2.0)
        mu = grid["mu"]
        assert float(mu.min()) >= 0.0
        assert float(mu.max()) <= 1.0 + 1e-5

    def test_mu_centre_pixel_is_one(self):
        """The central pixel should have mu ≈ 1 (disc centre)."""
        grid = build_stellar_grid(50, 2.0)
        centre_mask = (grid["x"] == 0) & (grid["y"] == 0)
        assert np.any(centre_mask), "Centre pixel not found in grid"
        assert abs(float(grid["mu"][centre_mask][0]) - 1.0) < 1e-5

    def test_vel_row_zero_when_ve_zero(self):
        """Doppler factor should be identically zero for non-rotating star."""
        grid = build_stellar_grid(50, ve=0.0)
        assert np.allclose(grid["vel_row"], 0.0, atol=1e-10)

    def test_row_idx_within_bounds(self):
        grid = build_stellar_grid(50, 2.0)
        assert np.all(grid["row_idx"] >= 0)
        assert np.all(grid["row_idx"] < grid["n"])

    def test_row_idx_reconstructs_per_pixel_velocity(self):
        """row_idx + vel_row must exactly reproduce y/spr*(ve/c) per pixel --
        this is the row-based Doppler optimization's core correctness
        property: it must be an exact reformulation, not an approximation."""
        ve = 30.0
        grid = build_stellar_grid(50, ve=ve)
        expected = grid["y"] / grid["star_pixel_rad"] * (ve / C_KMS)
        reconstructed = grid["vel_row"][grid["row_idx"]]
        np.testing.assert_allclose(reconstructed, expected, atol=1e-6)

    def test_flat_indices_within_bounds(self):
        grid = build_stellar_grid(50, 2.0)
        n = grid["n"]
        assert np.all(grid["flat_indices"] >= 0)
        assert np.all(grid["flat_indices"] < n * n)

    def test_dtype_consistency(self):
        grid = build_stellar_grid(50, 2.0)
        dtypes = {grid["x"].dtype, grid["y"].dtype, grid["mu"].dtype}
        assert len(dtypes) == 1, f"Grid arrays have mixed dtypes: {dtypes}"


# ===================================================================
# Single-phase and multi-phase output shapes
# ===================================================================

class TestOutputShapes:

    def test_single_phase(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[20.0],
            ar_long=[0.0],
            ar_size=[10.0],
            ar_smoothness=[_SM],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=2.0,
            ldc_mode="quadratic",
        )
        assert result["lc"].shape == (1, len(wl))
        assert result["epsilon"].shape == (1, len(wl))
        assert result["star_maps"].ndim == 3

    def test_multi_phase(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        phases = np.linspace(0, 360, 8, endpoint=False)
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[20.0],
            ar_long=[0.0],
            ar_size=[10.0],
            ar_smoothness=[_SM],
            phases_rot=phases,
            stellar_grid_size=50,
            ve=2.0,
        )
        assert result["lc"].shape == (8, len(wl))
        assert result["epsilon"].shape == (8, len(wl))
        assert result["star_maps"].shape[0] == 8


# ===================================================================
# Physical sanity checks
# ===================================================================

class TestPhysics:

    def test_no_ar_flux_is_unity(self, flat_spectra, base_params):
        """With a vanishingly small AR the light curve should be ~1."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[0.001], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="quadratic",
        )
        assert abs(float(result["lc"][0, 0]) - 1.0) < 0.01

    def test_cold_ar_dims_flux(self, flat_spectra, base_params):
        """A visible cold AR should reduce the total flux."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[20.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="quadratic",
        )
        assert float(result["lc"][0, 0]) < 1.0

    def test_hot_ar_brightens_flux(self, flat_spectra, base_params):
        """A facula (flux_active > flux_quiet) should increase total flux."""
        wl, flux_quiet, _ = flat_spectra
        flux_facula = np.full_like(wl, 1.3)
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_facula,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[20.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="quadratic",
        )
        assert float(result["lc"][0, 0]) > 1.0

    def test_far_side_ar_invisible(self, flat_spectra, base_params):
        """An AR on the far side of the star should not affect the flux."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0], ar_long=[180.0], ar_size=[15.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="quadratic",
        )
        assert abs(float(result["lc"][0, 0]) - 1.0) < 0.01

    def test_light_curve_is_periodic(self, flat_spectra, base_params):
        """LC at phase=0 should equal LC at phase=360."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[20.0], ar_long=[45.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=[0.0, 360.0], stellar_grid_size=50, ve=2.0, ldc_mode="quadratic",
        )
        np.testing.assert_allclose(result["lc"][0], result["lc"][1], rtol=1e-5)

    def test_epsilon_unity_without_ar(self, flat_spectra, base_params):
        """Contamination factor should be ~1 everywhere with no AR."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[0.001], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        np.testing.assert_allclose(
            result["epsilon"][0], 1.0, atol=0.01,
            err_msg="epsilon should be ~1 when no AR is visible",
        )

    def test_epsilon_gt_one_for_cold_ar(self, flat_spectra, base_params):
        """ε = F_quiet / F_spotted > 1 when the AR dims the star."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[20.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        assert np.all(result["epsilon"][0] > 1.0), (
            "epsilon should be > 1 for a cold AR (flux_active < flux_quiet)"
        )


# ===================================================================
# Multiple active regions
# ===================================================================

class TestMultiAR:

    def test_multi_ar_shapes(self, flat_spectra, base_params):
        """Two ARs should work and return correct shapes."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[20.0, -20.0], ar_long=[0.0, 180.0], ar_size=[10.0, 10.0],
            ar_smoothness=[_SM, _SM],
            phases_rot=np.linspace(0, 360, 6, endpoint=False),
            stellar_grid_size=50, ve=2.0,
        )
        assert result["lc"].shape == (6, len(wl))

    def test_per_ar_spectra(self, flat_spectra, base_params):
        """Each AR can have its own spectrum: flux_active shape (nar, nwave)."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        flux_active_multi = np.stack([
            np.full(nwave, 0.5),    # cold spot
            np.full(nwave, 0.9),    # mild spot
            np.full(nwave, 1.2),    # facula
        ])  # (3, nwave)

        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active_multi,
            params=base_params,
            ar_lat=[10.0, -10.0, 30.0], ar_long=[0.0, 60.0, 120.0],
            ar_size=[8.0, 8.0, 8.0], ar_smoothness=[_SM, _SM, _SM],
            phases_rot=[0.0, 90.0], stellar_grid_size=50, ve=1.0,
        )
        assert result["lc"].shape == (2, len(wl))
        assert np.all(np.isfinite(result["lc"]))


# ===================================================================
# AR overlap compounding (contrast-surface superposition, no more
# hottest/coldest-wins winner-take-all)
# ===================================================================

class TestAROverlapCompounding:
    """
    Overlapping active regions contribute simultaneously -- the contrast
    surface sums their deviations from 1 (see core.py's module docstring) --
    rather than one excluding the other. An umbra sitting inside a penumbra
    should therefore combine to something darker than either component's
    own contrast alone.
    """

    def test_same_centre_overlap_deeper_than_either_alone(
        self, flat_spectra, base_params
    ):
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        combined = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet,
            flux_active=np.stack([np.full(nwave, 0.3), np.full(nwave, 0.7)]),
            params=base_params,
            ar_lat=[0.0, 0.0], ar_long=[0.0, 0.0], ar_size=[15.0, 15.0],
            ar_smoothness=[50.0, 50.0],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        shallower_alone = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=np.full((1, nwave), 0.7),
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[15.0], ar_smoothness=[50.0],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        assert float(combined["lc"][0, 0]) < float(shallower_alone["lc"][0, 0]), (
            "overlapping components should compound, giving a deeper dip "
            "than either component alone"
        )

    def test_exact_contrast_surface_value_at_shared_centre(self):
        """
        With no limb darkening, the flux at the exact shared centre of two
        co-located ARs must equal 1 - [(1-C1)+(1-C2)] exactly.
        """
        wl = np.array([550.0])
        flux_quiet = np.array([1.0])
        params = dict(ldc_coeffs=[0.0, 0.0], inc_star=90.0)
        model = build_model(
            wavelength=wl, flux_quiet=flux_quiet, params=params,
            phases_rot=np.array([0.0]), stellar_grid_size=60, ve=0.0,
        )
        flux_active = jnp.array([[0.3], [0.7]])
        result = evaluate_light_curve(
            model, flux_active, jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]),
            jnp.array([15.0, 15.0]), jnp.array([50.0, 50.0]),
        )
        star_map = np.array(result["star_maps"][0])
        centre = star_map.shape[0] // 2
        expected = 1.0 - ((1 - 0.3) + (1 - 0.7))
        assert abs(star_map[centre, centre] - expected) < 1e-4

    def test_non_overlapping_ars_independent(self, flat_spectra, base_params):
        """Two well-separated ARs shouldn't measurably interact."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        combined = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet,
            flux_active=np.stack([np.full(nwave, 0.3), np.full(nwave, 1.5)]),
            params=base_params,
            ar_lat=[0.0, 0.0], ar_long=[0.0, 180.0], ar_size=[10.0, 10.0],
            ar_smoothness=[_SM, _SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        cold_alone = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=np.full((1, nwave), 0.3),
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        np.testing.assert_allclose(
            combined["lc"][0], cold_alone["lc"][0], rtol=1e-3,
            err_msg="far-side AR should not measurably affect the near-side AR",
        )


# ===================================================================
# LDC modes
# ===================================================================

class TestLDCModes:

    @pytest.mark.parametrize("ldc_mode,ldc_coeffs", [
        ("linear",      [0.3]),
        ("quadratic",   [0.3, 0.1]),
        ("power2",      [0.4, 0.6]),
        ("kipping3",    [0.2, 0.3, 0.1]),
        ("nonlinear4",  [0.1, 0.2, 0.15, 0.05]),
    ])
    def test_analytic_ldc_modes(
        self, flat_spectra, base_params, ldc_mode, ldc_coeffs
    ):
        wl, flux_quiet, flux_active = flat_spectra
        params = {**base_params, "ldc_coeffs": ldc_coeffs}
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=params,
            ar_lat=[15.0], ar_long=[0.0], ar_size=[8.0], ar_smoothness=[_SM],
            phases_rot=[0.0, 90.0], stellar_grid_size=50, ve=1.0, ldc_mode=ldc_mode,
        )
        assert result["lc"].shape == (2, len(wl))
        assert np.all(np.isfinite(result["lc"]))
        assert np.all(np.isfinite(result["epsilon"]))

    def test_intensity_profile_mode(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        nwave = len(wl)
        mu_pts = np.linspace(0.0, 1.0, 50)
        I_profile = np.tile(mu_pts, (nwave, 1))  # (nwave, 50)

        params = dict(inc_star=90.0, mu_profile=mu_pts, I_profile=I_profile)
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="intensity_profile",
        )
        assert result["lc"].shape == (1, nwave)
        assert np.all(np.isfinite(result["lc"]))

    def test_legacy_u1_u2_keys(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(inc_star=90.0, u1=0.3, u2=0.1)
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=params,
            ar_lat=[10.0], ar_long=[0.0], ar_size=[8.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="quadratic",
        )
        assert np.all(np.isfinite(result["lc"]))

    def test_per_wavelength_ldc(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        nwave = len(wl)
        params = dict(
            inc_star=90.0,
            ldc_coeffs=[np.linspace(0.2, 0.5, nwave), np.linspace(0.05, 0.2, nwave)],
        )
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=params,
            ar_lat=[10.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="quadratic",
        )
        assert np.all(np.isfinite(result["lc"]))

    def test_per_ar_ldc_differs_from_quiet(self, flat_spectra, base_params):
        """An AR with different LDC coefficients than quiet must give a
        different light curve than the default (quiet's own coefficients)."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        model = build_model(
            wavelength=wl, flux_quiet=flux_quiet, params=base_params,
            phases_rot=np.array([0.0]), stellar_grid_size=50, ve=0.0,
        )
        common = dict(
            flux_active=jnp.asarray(np.full((1, nwave), 0.5)),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([15.0]), ar_smoothness=jnp.array([_SM]),
        )
        default_result = evaluate_light_curve(model, **common)
        custom_ldc = jnp.asarray(np.tile([0.9, 0.05], (nwave, 1))[None, :, :])
        custom_result = evaluate_light_curve(model, **common, ldc_coeffs_active=custom_ldc)
        assert not np.allclose(default_result["lc"], custom_result["lc"])

    def test_default_ar_ldc_matches_quiet(self, flat_spectra, base_params):
        """Omitting ldc_coeffs_active must exactly match explicitly passing
        the quiet photosphere's own coefficients, broadcast to all ARs."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        model = build_model(
            wavelength=wl, flux_quiet=flux_quiet, params=base_params,
            phases_rot=np.array([0.0]), stellar_grid_size=50, ve=0.0,
        )
        common = dict(
            flux_active=jnp.asarray(np.full((1, nwave), 0.5)),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([15.0]), ar_smoothness=jnp.array([_SM]),
        )
        r_default = evaluate_light_curve(model, **common)
        explicit_quiet_ldc = jnp.broadcast_to(model["ldc_coeffs"][None, :, :], (1, nwave, 2))
        r_explicit = evaluate_light_curve(model, **common, ldc_coeffs_active=explicit_quiet_ldc)
        np.testing.assert_allclose(r_default["lc"], r_explicit["lc"], rtol=1e-6)


# ===================================================================
# Input validation
# ===================================================================

class TestInputValidation:

    def test_invalid_ldc_mode_raises_valueerror(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(inc_star=90.0, ldc_coeffs=[0.3])
        with pytest.raises(ValueError, match="ldc_mode"):
            build_model(
                wavelength=wl, flux_quiet=flux_quiet, params=params,
                phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="banana",
            )

    def test_wrong_number_of_ldc_coeffs_raises(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(inc_star=90.0, ldc_coeffs=[0.3, 0.1])
        with pytest.raises(ValueError, match="coefficient"):
            build_model(
                wavelength=wl, flux_quiet=flux_quiet, params=params,
                phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="nonlinear4",
            )

    def test_non_monotonic_mu_profile_raises(self, flat_spectra):
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        bad_mu = np.array([0.0, 0.5, 0.3, 1.0])
        params = dict(inc_star=90.0, mu_profile=bad_mu, I_profile=np.ones((nwave, len(bad_mu))))
        with pytest.raises(ValueError, match="mu_profile.*increasing"):
            build_model(
                wavelength=wl, flux_quiet=flux_quiet, params=params,
                phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="intensity_profile",
            )

    def test_flux_active_shape_mismatch_raises(self, small_model):
        wrong_flux = jnp.ones(5)
        with pytest.raises(ValueError, match="flux_active"):
            evaluate_light_curve(
                small_model, flux_active=wrong_flux,
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_flux_active_2d_shape_mismatch_raises(self, small_model):
        nwave = small_model["nwave"]
        wrong_flux = jnp.ones((5, nwave))
        with pytest.raises(ValueError, match="flux_active"):
            evaluate_light_curve(
                small_model, flux_active=wrong_flux,
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_ar_smoothness_shape_mismatch_raises(self, small_model):
        nwave = small_model["nwave"]
        with pytest.raises(ValueError, match="ar_smoothness"):
            evaluate_light_curve(
                small_model, flux_active=jnp.ones((2, nwave)),
                ar_lat=jnp.array([0.0, 10.0]), ar_long=jnp.array([0.0, 10.0]),
                ar_size=jnp.array([10.0, 10.0]),
                ar_smoothness=jnp.array([1.0, 2.0, 3.0]),  # wrong size
            )

    def test_ldc_coeffs_active_shape_mismatch_raises(self, small_model):
        nwave = small_model["nwave"]
        with pytest.raises(ValueError, match="ldc_coeffs_active"):
            evaluate_light_curve(
                small_model, flux_active=jnp.ones((1, nwave)),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
                ldc_coeffs_active=jnp.ones((1, nwave, 5)),  # wrong n_coeffs (quadratic expects 2)
            )

    def test_ldc_coeffs_wavelength_length_mismatch_raises(self, flat_spectra):
        wl, flux_quiet, _ = flat_spectra
        params = dict(inc_star=90.0, ldc_coeffs=[np.ones(5), np.ones(5)])
        with pytest.raises(ValueError, match="wavelength grid"):
            build_model(
                wavelength=wl, flux_quiet=flux_quiet, params=params,
                phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="quadratic",
            )

    def test_missing_ldc_coeffs_raises(self, flat_spectra):
        wl, flux_quiet, _ = flat_spectra
        params = dict(inc_star=90.0)
        with pytest.raises(ValueError, match="ldc_coeffs"):
            build_model(
                wavelength=wl, flux_quiet=flux_quiet, params=params,
                phases_rot=[0.0], stellar_grid_size=50, ve=0.0, ldc_mode="power2",
            )

    def test_invalid_oversample_raises(self, flat_spectra, base_params):
        wl, flux_quiet, _ = flat_spectra
        with pytest.raises(ValueError, match="oversample"):
            build_model(
                wavelength=wl, flux_quiet=flux_quiet, params=base_params,
                phases_rot=[0.0], stellar_grid_size=50, ve=0.0, oversample=0,
            )


# ===================================================================
# Contamination factor edge cases
# ===================================================================

class TestContaminationEdgeCases:

    def test_epsilon_finite_for_normal_ar(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[15.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        assert np.all(np.isfinite(result["epsilon"]))

    def test_epsilon_handles_totally_dark_ar(self, flat_spectra, base_params):
        """A totally dark AR covering almost the whole disc shouldn't crash;
        epsilon may be very large or nan where bin_flux -> 0, by design."""
        wl, flux_quiet, _ = flat_spectra
        flux_dark = np.zeros_like(wl)
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_dark,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[89.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        assert result["epsilon"].shape == (1, len(wl))


# ===================================================================
# Two-stage API (build_model + evaluate_light_curve)
# ===================================================================

class TestTwoStageAPI:

    def test_two_stage_matches_convenience(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        phases = np.linspace(0, 360, 6, endpoint=False)

        result_one = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[15.0], ar_long=[30.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=phases, stellar_grid_size=50, ve=2.0,
        )

        model = build_model(
            wavelength=wl, flux_quiet=flux_quiet, params=base_params,
            phases_rot=phases, stellar_grid_size=50, ve=2.0,
        )
        result_two = evaluate_light_curve(
            model,
            flux_active=jnp.asarray(flux_active),
            ar_lat=jnp.array([15.0]), ar_long=jnp.array([30.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )

        np.testing.assert_allclose(result_one["lc"], np.array(result_two["lc"]), rtol=1e-5)
        np.testing.assert_allclose(result_one["epsilon"], np.array(result_two["epsilon"]), rtol=1e-5)

    def test_evaluate_reusable_with_different_ar_params(self, small_model):
        nwave = small_model["nwave"]
        result_a = evaluate_light_curve(
            small_model, flux_active=jnp.ones(nwave) * 0.5,
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )
        result_b = evaluate_light_curve(
            small_model, flux_active=jnp.ones(nwave) * 0.9,
            ar_lat=jnp.array([45.0]), ar_long=jnp.array([90.0]),
            ar_size=jnp.array([5.0]), ar_smoothness=jnp.array([_SM]),
        )
        assert not np.allclose(np.array(result_a["lc"]), np.array(result_b["lc"]))


# ===================================================================
# Oversampling cases
# ===================================================================

def test_oversample_smooths_light_curve():
    wavelength = np.array([550.0])
    flux_quiet = np.array([1.0])
    flux_active = np.array([[0.7]])
    params = dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0)
    phases = np.linspace(0, 360, 500, endpoint=False)

    common = dict(
        wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_active,
        params=params,
        ar_lat=[20.0], ar_long=[5.0], ar_size=[11.0], ar_smoothness=[_SM],
        phases_rot=phases, stellar_grid_size=100, ve=2.0, ldc_mode="quadratic",
    )

    # Single wavelength bin here -- squeeze lc down to one value per phase.
    lc_no_os = np.asarray(compute_light_curve(**common, oversample=1)["lc"])[:, 0]
    lc_os3   = np.asarray(compute_light_curve(**common, oversample=3)["lc"])[:, 0]

    assert lc_no_os.shape == lc_os3.shape

    roughness_no_os = np.max(np.abs(np.diff(lc_no_os)))
    roughness_os3   = np.max(np.abs(np.diff(lc_os3)))

    assert roughness_os3 <= roughness_no_os, (
        f"Oversampled roughness ({roughness_os3:.6f}) should be <= "
        f"non-oversampled ({roughness_no_os:.6f})"
    )


def test_oversample_1_is_identity():
    wavelength = np.array([550.0])
    flux_quiet = np.array([1.0])
    flux_active = np.array([[0.7]])
    params = dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0)
    phases = np.linspace(0, 360, 100, endpoint=False)

    common = dict(
        wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_active,
        params=params,
        ar_lat=[20.0], ar_long=[5.0], ar_size=[11.0], ar_smoothness=[_SM],
        phases_rot=phases, stellar_grid_size=80, ve=2.0,
    )

    lc_default = compute_light_curve(**common)["lc"]
    lc_os1     = compute_light_curve(**common, oversample=1)["lc"]

    np.testing.assert_array_equal(lc_default, lc_os1)


def test_oversample_invalid_value():
    with pytest.raises(ValueError, match="oversample"):
        build_model(
            wavelength=np.array([550.0]), flux_quiet=np.array([1.0]),
            params=dict(ldc_coeffs=[0.4, 0.2]),
            phases_rot=np.linspace(0, 360, 10),
            stellar_grid_size=50, ve=2.0, oversample=0,
        )


def test_oversample_preserves_shape():
    wavelength = np.array([550.0])
    flux_quiet = np.array([1.0])
    flux_active = np.array([[0.7]])
    params = dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0)
    phases = np.linspace(0, 360, 50, endpoint=False)

    common = dict(
        wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_active,
        params=params,
        ar_lat=[20.0], ar_long=[5.0], ar_size=[11.0], ar_smoothness=[_SM],
        phases_rot=phases, stellar_grid_size=80, ve=2.0,
    )

    for os_factor in [1, 3, 5]:
        result = compute_light_curve(**common, oversample=os_factor)
        assert result["lc"].shape == (50, 1)
        assert result["epsilon"].shape[0] == 50
        assert result["star_maps"].shape[0] == 50


# ===================================================================
# Numerical edge cases
# ===================================================================

class TestNumericalEdgeCases:

    def test_ar_at_pole(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        for lat in [90.0, -90.0]:
            result = compute_light_curve(
                wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
                params=base_params,
                ar_lat=[lat], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
                phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
            )
            assert np.all(np.isfinite(result["lc"]))

    def test_ar_size_zero(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[0.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        np.testing.assert_allclose(
            result["lc"][0], 1.0, atol=1e-3,
            err_msg="Zero-size AR should have negligible effect on flux",
        )

    def test_ar_size_90_degrees(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[90.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        assert np.all(np.isfinite(result["lc"]))
        assert float(result["lc"][0, 0]) < 0.95

    def test_inclination_zero_pole_on(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(ldc_coeffs=[0.3, 0.1], inc_star=0.0)
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=[0.0, 90.0, 180.0], stellar_grid_size=50, ve=0.0,
        )
        assert np.all(np.isfinite(result["lc"]))

    def test_inclination_zero_constant_lc(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(ldc_coeffs=[0.3, 0.1], inc_star=0.0)
        phases = np.linspace(0, 360, 12, endpoint=False)
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=phases, stellar_grid_size=50, ve=0.0,
        )
        lc = result["lc"]
        np.testing.assert_allclose(
            lc, np.mean(lc), rtol=5e-3,
            err_msg="Pole-on view should produce a constant light curve",
        )

    def test_single_wavelength(self, base_params):
        wl = np.array([550.0])
        flux_quiet = np.array([1.0])
        flux_active = np.array([0.8])
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[10.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=[0.0], stellar_grid_size=50, ve=0.0,
        )
        assert result["lc"].shape == (1, 1)
        assert result["epsilon"].shape == (1, 1)
        assert np.all(np.isfinite(result["lc"]))


# ===================================================================
# Symmetry tests
# ===================================================================

class TestSymmetry:

    def test_equatorial_ar_symmetric_phases(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=[45.0, 315.0], stellar_grid_size=50, ve=0.0, ldc_mode="quadratic",
        )
        np.testing.assert_allclose(
            result["lc"][0], result["lc"][1], rtol=1e-4,
            err_msg="Equatorial AR should be symmetric about phase=0",
        )

    def test_north_south_symmetry_equator_on(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        phases = np.linspace(0, 360, 8, endpoint=False)

        result_north = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[30.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=phases, stellar_grid_size=50, ve=0.0,
        )
        result_south = compute_light_curve(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            params=base_params,
            ar_lat=[-30.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=phases, stellar_grid_size=50, ve=0.0,
        )
        np.testing.assert_allclose(
            result_north["lc"], result_south["lc"], rtol=1e-4,
            err_msg="N/S symmetric ARs should produce identical LCs at inc=90°",
        )


# ---------------------------------------------------------------------------
# Shared test configuration
# ---------------------------------------------------------------------------

WAVELENGTH   = np.array([550.0])
FLUX_QUIET   = np.array([1.0])
FLUX_SPOT    = np.array([0.7])
FLUX_FACULA  = np.array([1.1])

STELLAR_GRID = 60
VE           = 0.0

BASE_PARAMS = dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0)

TIMES = np.linspace(-0.15, 0.15, 200)
P_ROT = 25.0

TRANSIT_PARAMS = dict(
    t0           = 0.0,
    period       = 5.0,
    a_over_rstar = 10.0,
    inclination  = np.pi / 2.0,
    k            = 0.1,
    ecc          = 0.0,
    omega_peri   = 0.0,
)


@pytest.fixture(scope="module")
def combined_model():
    return build_combined_model(
        wavelength        = WAVELENGTH,
        flux_quiet        = FLUX_QUIET,
        params            = BASE_PARAMS,
        times             = TIMES,
        P_rot             = P_ROT,
        transit_params    = TRANSIT_PARAMS,
        stellar_grid_size = STELLAR_GRID,
        ve                = VE,
        ldc_mode          = "quadratic",
        oversample        = 1,
    )


@pytest.fixture(scope="module")
def stellar_only_model():
    phases = (TIMES / P_ROT * 360.0) % 360.0
    return build_model(
        wavelength        = WAVELENGTH,
        flux_quiet        = FLUX_QUIET,
        params            = BASE_PARAMS,
        phases_rot        = phases,
        stellar_grid_size = STELLAR_GRID,
        ve                = VE,
        ldc_mode          = "quadratic",
        oversample        = 1,
    )


def _combined_lc(transit_params=None, ar_lat=None, ar_long=None, ar_size=None,
                 ar_smoothness=None, flux_active=None, oversample=1, params=None):
    """Thin wrapper that fills in defaults to keep test bodies short."""
    return compute_combined_light_curve(
        wavelength        = WAVELENGTH,
        flux_quiet        = FLUX_QUIET,
        flux_active       = np.atleast_2d(flux_active if flux_active is not None else FLUX_QUIET),
        params            = params or BASE_PARAMS,
        ar_lat            = ar_lat  or [0.0],
        ar_long           = ar_long or [180.0],  # far side — invisible by default
        ar_size           = ar_size or [0.001],
        ar_smoothness     = ar_smoothness or [_SM],
        times             = TIMES,
        P_rot             = P_ROT,
        transit_params    = transit_params or TRANSIT_PARAMS,
        stellar_grid_size = STELLAR_GRID,
        ve                = VE,
        oversample        = oversample,
    )


# ===================================================================
# 1.  _compute_planet_mask — pixel-level occultation mask
# ===================================================================

class TestComputePlanetMask:
    """Unit tests for the pixel-level planet occultation mask (unaffected
    by the contrast-surface AR rewrite -- planet.py is a separate module)."""

    @pytest.fixture(autouse=True)
    def _grid(self):
        g = build_stellar_grid(50, 0.0)
        self.x   = jnp.asarray(g["x"])
        self.y   = jnp.asarray(g["y"])
        self.spr = g["star_pixel_rad"]

    def _mask(self, X=0.0, Y=0.0, Z=5.0, k=0.1):
        return _compute_planet_mask(self.x, self.y, self.spr, X, Y, Z, k)

    def test_shape_matches_disc(self):
        mask = self._mask()
        assert mask.shape == self.x.shape

    def test_dtype_is_float(self):
        assert self._mask().dtype == jnp.float32

    def test_all_false_when_planet_behind_star(self):
        assert not jnp.any(self._mask(Z=-1.0))

    def test_all_false_at_Z_exactly_zero(self):
        assert not jnp.any(self._mask(Z=0.0))

    def test_all_false_for_zero_radius_planet(self):
        assert not jnp.any(self._mask(k=0.0))

    def test_some_pixels_masked_at_disc_centre(self):
        assert jnp.any(self._mask(X=0.0, Y=0.0, Z=5.0, k=0.1))

    def test_masked_pixel_count_scales_with_k(self):
        n_small = int(jnp.sum(self._mask(k=0.05)))
        n_large = int(jnp.sum(self._mask(k=0.20)))
        assert n_large > n_small

    def test_planet_outside_disc_masks_nothing(self):
        assert not jnp.any(self._mask(X=3.0, Y=0.0, Z=5.0, k=0.1))

    def test_mask_centre_pixels_when_centred(self):
        centre = (np.hypot(np.array(self.x), np.array(self.y)) / self.spr) < 0.05
        mask = np.array(self._mask(X=0.0, Y=0.0, Z=5.0, k=0.15))
        if centre.any():
            assert mask[centre].all()


# ===================================================================
# 2.  build_combined_model — model dict structure
# ===================================================================

class TestBuildCombinedModel:

    def test_has_transit_flag_set(self, combined_model):
        assert combined_model.get("has_transit") is True

    def test_planet_xyz_key_present(self, combined_model):
        assert "planet_xyz" in combined_model

    def test_k_value_stored(self, combined_model):
        assert "k" in combined_model
        assert float(combined_model["k"]) == pytest.approx(TRANSIT_PARAMS["k"])

    def test_planet_xyz_shape(self, combined_model):
        xyz    = combined_model["planet_xyz"]
        nphase = combined_model["nphase"]
        assert xyz.shape == (nphase, 3)

    def test_xyz_third_column_Z(self, combined_model):
        xyz = np.array(combined_model["planet_xyz"])
        assert np.any(xyz[:, 2] > 0)

    def test_all_stellar_keys_preserved(self, combined_model):
        required = [
            "x_disc", "y_disc", "mu_disc", "row_idx", "vel_row",
            "star_pixel_rad", "total_pixels", "wavelength",
            "phases_rot", "ldc_coeffs", "flat_indices", "n",
        ]
        for key in required:
            assert key in combined_model, f"Stellar key missing: '{key}'"

    def test_oversample_inflates_nphase(self):
        oversample = 3
        model = build_combined_model(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, params=BASE_PARAMS,
            times=TIMES, P_rot=P_ROT, transit_params=TRANSIT_PARAMS,
            stellar_grid_size=STELLAR_GRID, ve=VE, oversample=oversample,
        )
        n_orig    = model["nphase_original"]
        n_compute = model["nphase"]
        assert n_compute == n_orig * oversample

    def test_planet_xyz_length_matches_nphase(self):
        oversample = 3
        model = build_combined_model(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, params=BASE_PARAMS,
            times=TIMES, P_rot=P_ROT, transit_params=TRANSIT_PARAMS,
            stellar_grid_size=STELLAR_GRID, ve=VE, oversample=oversample,
        )
        assert model["planet_xyz"].shape[0] == model["nphase"]


# ===================================================================
# 3.  Transit physics
# ===================================================================

class TestTransitPhysics:

    def test_output_shape_matches_times(self):
        result = _combined_lc()
        assert result["lc"].shape == (len(TIMES), 1)

    def test_lc_finite(self):
        assert np.all(np.isfinite(_combined_lc()["lc"]))

    def test_transit_produces_flux_dip(self):
        lc = _combined_lc()["lc"]
        assert float(np.min(lc)) < 1.0

    def test_transit_depth_scales_with_k(self):
        d_small = 1.0 - float(np.min(_combined_lc({**TRANSIT_PARAMS, "k": 0.05})["lc"]))
        d_large = 1.0 - float(np.min(_combined_lc({**TRANSIT_PARAMS, "k": 0.15})["lc"]))
        assert d_large > d_small

    def test_approximate_transit_depth_equals_k_squared(self):
        k = 0.1
        tp = {**TRANSIT_PARAMS, "k": k}
        params_no_ld = dict(ldc_coeffs=[0.0, 0.0], inc_star=90.0)
        lc = _combined_lc(tp, params=params_no_ld)["lc"]
        depth = 1.0 - float(np.min(lc))
        np.testing.assert_allclose(depth, k**2, rtol=0.15)

    def test_grazing_transit_shallower_than_central(self):
        k   = 0.1
        a   = TRANSIT_PARAMS["a_over_rstar"]
        inc_grazing = np.arccos(0.85 / a)

        d_central = 1.0 - float(np.min(_combined_lc({**TRANSIT_PARAMS, "k": k})["lc"]))
        d_grazing = 1.0 - float(np.min(_combined_lc(
            {**TRANSIT_PARAMS, "k": k, "inclination": inc_grazing})["lc"]))
        assert d_central > d_grazing

    def test_spot_crossing_produces_positive_bump(self):
        lc_spot = _combined_lc(
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], flux_active=FLUX_SPOT,
        )["lc"]
        lc_clean = _combined_lc(
            ar_lat=[0.0], ar_long=[180.0], ar_size=[10.0], flux_active=FLUX_SPOT,
        )["lc"]

        oot = np.abs(TIMES) > 0.12
        lc_spot_norm = lc_spot / np.median(lc_spot[oot])
        lc_clean_norm = lc_clean / np.median(lc_clean[oot])

        in_transit = np.abs(TIMES) < 0.04
        bump = float(np.min(lc_spot_norm[in_transit])) - float(np.min(lc_clean_norm[in_transit]))
        assert bump > 0

    def test_facula_crossing_produces_negative_anomaly(self):
        lc_fac = _combined_lc(
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], flux_active=FLUX_FACULA,
        )["lc"]
        lc_clean = _combined_lc(
            ar_lat=[0.0], ar_long=[180.0], ar_size=[10.0], flux_active=FLUX_FACULA,
        )["lc"]

        oot = np.abs(TIMES) > 0.12
        lc_fac_norm = lc_fac / np.median(lc_fac[oot])
        lc_clean_norm = lc_clean / np.median(lc_clean[oot])

        in_transit = np.abs(TIMES) < 0.04
        dip_delta = float(np.min(lc_fac_norm[in_transit])) - float(np.min(lc_clean_norm[in_transit]))
        assert dip_delta < 0

    def test_out_of_transit_matches_stellar_only(self, stellar_only_model):
        result_combined = _combined_lc(
            ar_lat=[0.0], ar_long=[180.0], ar_size=[0.001], flux_active=FLUX_QUIET,
        )
        result_stellar = evaluate_light_curve(
            stellar_only_model,
            flux_active=jnp.array(FLUX_QUIET),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([180.0]),
            ar_size=jnp.array([0.001]), ar_smoothness=jnp.array([_SM]),
        )
        oot = np.abs(TIMES) > 0.12
        np.testing.assert_allclose(
            result_combined["lc"][oot], np.array(result_stellar["lc"])[oot], rtol=1e-4,
        )

    def test_eccentric_orbit_centre_Z_positive(self):
        for ecc, omega in [(0.3, np.pi / 2.0), (0.5, np.pi / 4.0)]:
            tp = {**TRANSIT_PARAMS, "ecc": ecc, "omega_peri": omega}
            lc = _combined_lc(tp)["lc"]
            assert float(np.min(lc)) < 1.0

    def test_no_transit_when_fully_inclined(self):
        a = TRANSIT_PARAMS["a_over_rstar"]
        k = TRANSIT_PARAMS["k"]
        inc_no_transit = np.arccos(2.0 * (1.0 + k) / a)
        tp = {**TRANSIT_PARAMS, "inclination": inc_no_transit}
        lc = _combined_lc(tp)["lc"]
        np.testing.assert_allclose(lc, 1.0, atol=0.005)


# ===================================================================
# 4.  Oversampling — transit path
# ===================================================================

class TestTransitOversampling:

    def test_oversample_preserves_output_shape(self):
        for os in [1, 3, 5]:
            lc = _combined_lc(oversample=os)["lc"]
            assert lc.shape == (len(TIMES), 1)

    def test_oversampled_lc_is_finite(self):
        lc = _combined_lc(oversample=3)["lc"]
        assert np.all(np.isfinite(lc))

    def test_oversampled_transit_still_present(self):
        lc = _combined_lc(oversample=3)["lc"]
        assert float(np.min(lc)) < 1.0


# ===================================================================
# 5.  API consistency
# ===================================================================

class TestAPIConsistency:

    def test_stellar_only_model_lacks_transit_flag(self, stellar_only_model):
        assert not stellar_only_model.get("has_transit", False)

    def test_evaluate_stellar_only_still_works(self, stellar_only_model):
        result = evaluate_light_curve(
            stellar_only_model,
            flux_active=jnp.array(FLUX_SPOT),
            ar_lat=jnp.array([20.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )
        lc = np.array(result["lc"])
        assert np.all(np.isfinite(lc))
        assert lc.shape == (len(TIMES), 1)

    def test_compute_light_curve_api_unchanged(self):
        from sajax import compute_light_curve
        phases = (TIMES / P_ROT * 360.0) % 360.0
        result = compute_light_curve(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, flux_active=FLUX_SPOT,
            params=BASE_PARAMS,
            ar_lat=[20.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            phases_rot=phases,
            stellar_grid_size=STELLAR_GRID, ve=VE,
        )
        assert result["lc"].shape == (len(TIMES), 1)
        assert np.all(np.isfinite(result["lc"]))

    def test_no_transit_flag_gives_unity_transit_factor(self, stellar_only_model):
        result_stellar = evaluate_light_curve(
            stellar_only_model,
            flux_active=jnp.array(FLUX_QUIET),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([180.0]),
            ar_size=jnp.array([0.001]), ar_smoothness=jnp.array([_SM]),
        )
        lc = np.array(result_stellar["lc"])
        assert float(np.max(lc) - np.min(lc)) < 0.005


# ===================================================================
# Autodiff propagation
# ===================================================================

class TestAutodiff:
    """
    Verify that jax.grad produces finite, non-zero gradients through the
    full pipeline for every physically meaningful parameter -- AR size,
    lat, long, smoothness, and spectral contrast.
    """

    @pytest.fixture(scope="class")
    def grad_model(self):
        return build_model(
            wavelength=np.array([550.0]),
            flux_quiet=np.array([1.0]),
            params=dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0),
            phases_rot=np.array([0.0]),
            stellar_grid_size=50,
            ve=0.0,
        )

    @staticmethod
    def _lc_scalar(model, flux_active, ar_lat, ar_long, ar_size, ar_smoothness):
        return jnp.sum(evaluate_light_curve(
            model, flux_active=flux_active, ar_lat=ar_lat, ar_long=ar_long,
            ar_size=ar_size, ar_smoothness=ar_smoothness,
        )["lc"])

    def test_grad_wrt_flux_active(self, grad_model):
        fa = jnp.array([0.7])
        grad = jax.grad(
            lambda fa: self._lc_scalar(
                grad_model, fa, jnp.array([0.0]), jnp.array([0.0]),
                jnp.array([20.0]), jnp.array([_SM]),
            )
        )(fa)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(jnp.abs(grad) > 0)

    def test_grad_wrt_ar_size(self, grad_model):
        fa = jnp.array([0.7])
        grad = jax.grad(
            lambda sz: self._lc_scalar(
                grad_model, fa, jnp.array([0.0]), jnp.array([0.0]), sz, jnp.array([_SM]),
            )
        )(jnp.array([20.0]))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(jnp.abs(grad) > 0)

    def test_grad_wrt_ar_lat(self, grad_model):
        fa = jnp.array([0.7])
        grad = jax.grad(
            lambda lat: self._lc_scalar(
                grad_model, fa, lat, jnp.array([0.0]), jnp.array([20.0]), jnp.array([_SM]),
            )
        )(jnp.array([30.0]))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(jnp.abs(grad) > 0)

    def test_grad_wrt_ar_long(self, grad_model):
        fa = jnp.array([0.7])
        grad = jax.grad(
            lambda lng: self._lc_scalar(
                grad_model, fa, jnp.array([0.0]), lng, jnp.array([20.0]), jnp.array([_SM]),
            )
        )(jnp.array([45.0]))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(jnp.abs(grad) > 0)

    def test_grad_wrt_ar_smoothness(self, grad_model):
        fa = jnp.array([0.7])
        grad = jax.grad(
            lambda sm: self._lc_scalar(
                grad_model, fa, jnp.array([0.0]), jnp.array([0.0]), jnp.array([20.0]), sm,
            )
        )(jnp.array([_SM]))
        assert jnp.all(jnp.isfinite(grad))


# ===================================================================
# Gradient sign and FD-agreement tests for _compute_ar_shape
# ===================================================================

class TestARShapeGradients:
    """
    Verify that _compute_ar_shape's gradients have the physically correct
    sign and agree with central finite differences at a properly calibrated
    step size.

    Background
    ----------
    At ar_smoothness=20 the boundary transition is narrow enough that a
    finite-difference step of h=0.1 rad overshoots it, so the FD chord no
    longer approximates the local derivative (JAX's infinitesimal-limit
    gradient is correct; the FD estimate is not). h=1e-4 rad stays well
    inside the boundary's locally-linear regime.
    """

    _spr        = 50.0
    _arsize_deg = 20.0
    _smoothness = 20.0

    @property
    def _arsize_rad(self):
        return float(jnp.deg2rad(self._arsize_deg))

    def _boundary_pixel(self):
        s = float(jnp.sin(jnp.float32(self._arsize_rad)))
        return self._spr * s, 0.0

    def _shape_sum(self, arsize, spx=0.0, spy=0.0, spz=None, px=None, py=None,
                   smoothness=None):
        spz = spz if spz is not None else self._spr
        if px is None:
            px, py = self._boundary_pixel()
        sm = smoothness if smoothness is not None else self._smoothness
        return jnp.sum(_compute_ar_shape(
            jnp.array([px], dtype=jnp.float32),
            jnp.array([py], dtype=jnp.float32),
            self._spr, spx, spy, spz, arsize, sm,
        ))

    def test_arsize_gradient_positive_at_boundary(self):
        """Enlarging the AR must increase the shape value at its boundary pixel."""
        arsize = jnp.float32(self._arsize_rad)
        grad = float(jax.grad(self._shape_sum)(arsize))
        assert grad > 0

    def test_arsize_gradient_fd_agreement_with_small_h(self):
        """JAX autodiff agrees with FD at h=1e-4 rad, inside the boundary's linear regime."""
        arsize = jnp.float32(self._arsize_rad)
        h = jnp.float32(1e-4)

        grad_jax = float(jax.grad(self._shape_sum)(arsize))
        fd = float((self._shape_sum(arsize + h) - self._shape_sum(arsize - h)) / (2.0 * h))

        assert abs(fd) > 0.1 * abs(grad_jax)
        ratio = grad_jax / fd
        assert 0.5 <= ratio <= 2.0, (
            f"JAX ({grad_jax:.4g}) disagrees with FD ({fd:.4g}), ratio={ratio:.3f}."
        )

    def test_arsize_large_h_gives_wrong_result(self):
        """h=0.1 rad overshoots the boundary transition, disagreeing with JAX
        by more than an order of magnitude."""
        arsize = jnp.float32(self._arsize_rad)
        h_large = jnp.float32(0.1)

        grad_jax = float(jax.grad(self._shape_sum)(arsize))
        fd_large = float((self._shape_sum(arsize + h_large) - self._shape_sum(arsize - h_large))
                         / (2.0 * h_large))

        if abs(fd_large) < 1e-30:
            return
        ratio = abs(grad_jax / fd_large)
        sign_flip = float(np.sign(grad_jax)) != float(np.sign(fd_large))
        assert ratio < 0.1 or ratio > 10.0 or sign_flip

    def test_spz_gradient_positive_at_boundary_pixel(self):
        """
        Place the AR centre on the z-axis at spz = cos(arsize) * spr so that
        the disc-centre pixel sits exactly on the AR boundary. Increasing
        spz moves the AR toward the observer, raising cos_theta at the
        disc-centre pixel -> shape increases -> d(shape)/d(spz) > 0.
        """
        cos_a = float(jnp.cos(jnp.float32(self._arsize_rad)))
        spz0  = jnp.float32(cos_a * self._spr)

        grad = float(jax.grad(lambda sz: self._shape_sum(
            jnp.float32(self._arsize_rad),
            spx=0.0, spy=0.0, spz=sz, px=0.0, py=0.0,
        ))(spz0))
        assert grad > 0

    def test_dark_spot_flux_decreases_as_spot_grows(self, flat_spectra, base_params):
        """
        For a dark spot (flux_active < 1.0) centred on the disc, enlarging
        the spot must reduce the total normalised broadband flux.
        """
        wl, flux_quiet, flux_active = flat_spectra
        nwave = len(wl)
        assert float(flux_active[0]) < float(flux_quiet[0])

        model = build_model(
            wavelength=wl, flux_quiet=flux_quiet, params=base_params,
            phases_rot=np.array([0.0]), stellar_grid_size=int(self._spr),
            ve=0.0, ldc_mode="quadratic",
        )
        spr = float(model["star_pixel_rad"])
        ar_cart    = jnp.array([[0.0, 0.0, spr]], dtype=jnp.float32)
        planet_xyz = jnp.array([0.0, 0.0, -1e10], dtype=jnp.float32)
        flux_act   = jnp.array(flux_active[np.newaxis, :], dtype=jnp.float32)  # (1, nwave)
        n_coeffs   = 2
        n_mu       = model["mu_profile_pts"].shape[0]

        def flux_fn(arsize):
            fval, _, _ = _compute_single_phase(
                ar_cart, planet_xyz,
                wavelength          = model["wavelength"],
                flux_quiet          = model["flux_quiet"],
                flux_active         = flux_act,
                ldc_coeffs_quiet    = model["ldc_coeffs"],
                ldc_coeffs_active   = jnp.broadcast_to(model["ldc_coeffs"][None, :, :], (1, nwave, n_coeffs)),
                I_profile_quiet     = model["I_profile"],
                I_profile_active    = jnp.broadcast_to(model["I_profile"][None, :, :], (1, nwave, n_mu)),
                mu_profile_pts      = model["mu_profile_pts"],
                x_disc              = model["x_disc"],
                y_disc              = model["y_disc"],
                mu_disc             = model["mu_disc"],
                row_idx             = model["row_idx"],
                vel_row             = model["vel_row"],
                star_pixel_rad      = spr,
                total_pixels        = model["total_pixels"],
                arsize_rads         = jnp.array([arsize]),
                ar_smoothness       = jnp.array([_SM]),
                k                   = jnp.float32(0.0),
                ldc_mode            = model["ldc_mode"],
                plot_map_wavelength = model["plot_map_wavelength"],
                n                   = model["n"],
                flat_indices        = model["flat_indices"],
            )
            return jnp.sum(fval)

        arsize_val = jnp.float32(jnp.deg2rad(15.0))
        grad = float(jax.grad(flux_fn)(arsize_val))
        assert grad < 0


# ===================================================================
# FD-agreement tests for spot lat, long, size, and flux through the
# full evaluate_light_curve pipeline
# ===================================================================

class TestARParamGradientsFD:
    """
    Compare JAX autodiff to central finite differences for active-region
    latitude, longitude, and flux contrast, using the public
    ``evaluate_light_curve`` API. Step sizes below were empirically
    calibrated (not derived from a closed form, since the super-Gaussian's
    effective transition width doesn't have as simple a formula as the old
    sigmoid did) against a 20-degree AR at ar_smoothness=20 on a 50-pixel grid.
    """

    _SPR       = 50
    _ARSIZE    = 20.0   # degrees
    _H_LATLONG = 0.01   # degrees
    _H_FLUX    = 0.01

    @pytest.fixture(scope="class")
    def grad_model(self):
        return build_model(
            wavelength   = np.array([550.0]),
            flux_quiet   = np.array([1.0]),
            params       = dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0),
            phases_rot   = np.array([0.0]),
            stellar_grid_size = self._SPR,
            ve           = 0.0,
        )

    def _lc_sum(self, model, lat, long, flux, arsize=None, smoothness=None):
        arsize = jnp.float32(self._ARSIZE) if arsize is None else arsize
        smoothness = jnp.float32(_SM) if smoothness is None else smoothness
        return jnp.sum(evaluate_light_curve(
            model,
            flux_active   = jnp.array([flux]),
            ar_lat        = jnp.array([lat]),
            ar_long       = jnp.array([long]),
            ar_size       = jnp.array([arsize]),
            ar_smoothness = jnp.array([smoothness]),
        )["lc"])

    def _fd(self, f, x, h):
        return float((f(x + h) - f(x - h)) / (2.0 * h))

    def _check_fd_agreement(self, name, grad_jax, fd, h):
        assert abs(fd) > 0.1 * abs(grad_jax), (
            f"'{name}': FD ({fd:.3g}) is numerically degenerate vs JAX ({grad_jax:.3g})."
        )
        ratio = grad_jax / fd
        assert 0.5 <= ratio <= 2.0, (
            f"'{name}': JAX ({grad_jax:.4g}) disagrees with FD ({fd:.4g}), ratio={ratio:.3f}."
        )

    def test_lat_gradient_fd_agreement(self, grad_model):
        lat0  = jnp.float32(10.0)
        long0 = jnp.float32(0.0)
        flux0 = jnp.float32(0.7)
        h     = jnp.float32(self._H_LATLONG)

        grad_jax = float(jax.grad(
            lambda lat: self._lc_sum(grad_model, lat, long0, flux0)
        )(lat0))
        fd = self._fd(
            lambda lat: self._lc_sum(grad_model, jnp.float32(lat), long0, flux0),
            float(lat0), float(h),
        )
        self._check_fd_agreement("ar_lat", grad_jax, fd, float(h))

    def test_lat_gradient_finite_and_nonzero(self, grad_model):
        lat0 = jnp.float32(10.0)
        grad = float(jax.grad(
            lambda lat: self._lc_sum(grad_model, lat, jnp.float32(0.0), jnp.float32(0.7))
        )(lat0))
        assert np.isfinite(grad)
        assert abs(grad) > 0

    def test_long_gradient_fd_agreement(self, grad_model):
        lat0  = jnp.float32(0.0)
        long0 = jnp.float32(10.0)
        flux0 = jnp.float32(0.7)
        h     = jnp.float32(self._H_LATLONG)

        grad_jax = float(jax.grad(
            lambda lng: self._lc_sum(grad_model, lat0, lng, flux0)
        )(long0))
        fd = self._fd(
            lambda lng: self._lc_sum(grad_model, lat0, jnp.float32(lng), flux0),
            float(long0), float(h),
        )
        self._check_fd_agreement("ar_long", grad_jax, fd, float(h))

    def test_long_gradient_finite_and_nonzero(self, grad_model):
        long0 = jnp.float32(10.0)
        grad = float(jax.grad(
            lambda lng: self._lc_sum(grad_model, jnp.float32(0.0), lng, jnp.float32(0.7))
        )(long0))
        assert np.isfinite(grad)
        assert abs(grad) > 0

    def test_flux_gradient_positive_for_dark_spot(self, grad_model):
        flux0 = jnp.float32(0.7)
        grad = float(jax.grad(
            lambda fa: self._lc_sum(grad_model, jnp.float32(0.0), jnp.float32(0.0), fa)
        )(flux0))
        assert grad > 0

    def test_flux_gradient_fd_agreement(self, grad_model):
        flux0 = jnp.float32(0.7)
        h     = jnp.float32(self._H_FLUX)

        grad_jax = float(jax.grad(
            lambda fa: self._lc_sum(grad_model, jnp.float32(0.0), jnp.float32(0.0), fa)
        )(flux0))
        fd = self._fd(
            lambda fa: self._lc_sum(grad_model, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(fa)),
            float(flux0), float(h),
        )
        assert np.isfinite(fd) and abs(fd) > 0
        np.testing.assert_allclose(grad_jax, fd, rtol=1e-3)


# ===================================================================
# FD-agreement tests for stellar model parameters
# ===================================================================

class TestStellarParamGradientsFD:
    """
    Compare JAX autodiff to central finite differences for stellar model
    parameters: stellar inclination (inc_star), quadratic limb-darkening
    coefficients (u1, u2), and rotation period (P_rot). ``ve`` is checked
    separately (finite-only) -- see note on ``test_ve_gradient_finite``.

    These parameters are baked into the model dict at build time (NumPy),
    so tests call _compute_single_phase directly with the parameter as a
    JAX-traced input, constructing whichever piece of the model that
    parameter feeds into (e.g. ldc_coeffs_quiet for u1/u2, vel_row for ve).
    """

    _SPR    = 50
    _ARSIZE = 20.0    # degrees
    _H_INC  = 0.1     # degrees
    _H_LDC  = 0.01
    _H_PROT = 0.01    # days
    _C      = 299_792.458  # km/s

    @pytest.fixture(scope="class")
    def grad_model(self):
        return build_model(
            wavelength=np.array([550.0]),
            flux_quiet=np.array([1.0]),
            params=dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0),
            phases_rot=np.array([0.0]),
            stellar_grid_size=self._SPR,
            ve=0.0,
        )

    def _ar_cart(self, spr, lat_deg=30.0, long_deg=45.0):
        lat_r  = jnp.deg2rad(jnp.float32(lat_deg))
        long_r = jnp.deg2rad(jnp.float32(long_deg))
        return jnp.array([[
            spr * jnp.sin(long_r) * jnp.cos(lat_r),
            spr * jnp.sin(lat_r),
            spr * jnp.cos(long_r) * jnp.cos(lat_r),
        ]])  # (1, 3)

    def _call_single_phase(self, model, ar_cart_rotated,
                           vel_row=None, ldc_coeffs_quiet=None):
        nwave    = 1
        n_coeffs = 2
        n_mu     = model["mu_profile_pts"].shape[0]
        ldc_q    = ldc_coeffs_quiet if ldc_coeffs_quiet is not None else model["ldc_coeffs"]
        flux_norm, _, _ = _compute_single_phase(
            ar_cart_rotated,
            jnp.array([0.0, 0.0, -1e10]),
            wavelength          = model["wavelength"],
            flux_quiet          = model["flux_quiet"],
            flux_active         = jnp.array([[0.7]], dtype=jnp.float32),
            ldc_coeffs_quiet    = ldc_q,
            ldc_coeffs_active   = jnp.broadcast_to(model["ldc_coeffs"][None, :, :], (1, nwave, n_coeffs)),
            I_profile_quiet     = model["I_profile"],
            I_profile_active    = jnp.broadcast_to(model["I_profile"][None, :, :], (1, nwave, n_mu)),
            mu_profile_pts      = model["mu_profile_pts"],
            x_disc              = model["x_disc"],
            y_disc              = model["y_disc"],
            mu_disc             = model["mu_disc"],
            row_idx             = model["row_idx"],
            vel_row             = vel_row if vel_row is not None else model["vel_row"],
            star_pixel_rad      = model["star_pixel_rad"],
            total_pixels        = model["total_pixels"],
            arsize_rads         = jnp.array([jnp.deg2rad(jnp.float32(self._ARSIZE))]),
            ar_smoothness       = jnp.array([_SM]),
            k                   = jnp.float32(0.0),
            ldc_mode            = model["ldc_mode"],
            plot_map_wavelength = model["plot_map_wavelength"],
            n                   = model["n"],
            flat_indices        = model["flat_indices"],
        )
        return jnp.sum(flux_norm)

    def _fd(self, f, x, h):
        return float((f(x + h) - f(x - h)) / (2.0 * h))

    def _check(self, name, jax_g, fd, h):
        assert abs(fd) > 0.1 * abs(jax_g), (
            f"'{name}' FD degenerate (fd={fd:.3g}, jax={jax_g:.3g}, h={h:.3g})"
        )
        ratio = jax_g / fd
        assert 0.5 <= ratio <= 2.0, (
            f"'{name}' JAX ({jax_g:.4g}) vs FD ({fd:.4g}), ratio={ratio:.3f}"
        )

    def _check_tight(self, name, jax_g, fd, rtol):
        assert np.isfinite(fd) and abs(fd) > 0, f"'{name}' FD zero or non-finite"
        np.testing.assert_allclose(jax_g, fd, rtol=rtol)

    # ---- stellar inclination ------------------------------------------------

    def test_inc_star_gradient_finite_and_nonzero(self, grad_model):
        spr     = grad_model["star_pixel_rad"]
        ar_cart = self._ar_cart(spr)

        def lc(inc_deg):
            rotated = jax.vmap(
                lambda c: rotate_active_region(c, jnp.float32(0.0), inc_deg)
            )(ar_cart)
            return self._call_single_phase(grad_model, rotated)

        g = float(jax.grad(lc)(jnp.float32(75.0)))
        assert np.isfinite(g)
        assert abs(g) > 0

    def test_inc_star_gradient_fd_agreement(self, grad_model):
        spr     = grad_model["star_pixel_rad"]
        ar_cart = self._ar_cart(spr)
        inc0    = jnp.float32(75.0)
        h       = jnp.float32(self._H_INC)

        def lc(inc_deg):
            rotated = jax.vmap(
                lambda c: rotate_active_region(c, jnp.float32(0.0), inc_deg)
            )(ar_cart)
            return self._call_single_phase(grad_model, rotated)

        jax_g = float(jax.grad(lc)(inc0))
        fd    = self._fd(lambda i: float(lc(jnp.float32(i))), float(inc0), float(h))
        self._check("inc_star", jax_g, fd, float(h))

    # ---- limb-darkening coefficient u1 -------------------------------------

    def test_u1_gradient_finite_and_nonzero(self, grad_model):
        spr     = grad_model["star_pixel_rad"]
        rotated = jax.vmap(
            lambda c: rotate_active_region(c, jnp.float32(0.0), jnp.float32(90.0))
        )(self._ar_cart(spr))

        def lc(u1):
            return self._call_single_phase(
                grad_model, rotated,
                ldc_coeffs_quiet=jnp.array([[u1, jnp.float32(0.2)]]),
            )

        g = float(jax.grad(lc)(jnp.float32(0.4)))
        assert np.isfinite(g)
        assert abs(g) > 0

    def test_u1_gradient_fd_agreement(self, grad_model):
        spr     = grad_model["star_pixel_rad"]
        rotated = jax.vmap(
            lambda c: rotate_active_region(c, jnp.float32(0.0), jnp.float32(90.0))
        )(self._ar_cart(spr))
        u1_0 = jnp.float32(0.4)

        def lc(u1):
            return self._call_single_phase(
                grad_model, rotated,
                ldc_coeffs_quiet=jnp.array([[u1, jnp.float32(0.2)]]),
            )

        jax_g = float(jax.grad(lc)(u1_0))
        fd    = self._fd(lambda u: float(lc(jnp.float32(u))), float(u1_0), self._H_LDC)
        self._check_tight("u1", jax_g, fd, rtol=5e-3)

    def test_u2_gradient_fd_agreement(self, grad_model):
        spr     = grad_model["star_pixel_rad"]
        rotated = jax.vmap(
            lambda c: rotate_active_region(c, jnp.float32(0.0), jnp.float32(90.0))
        )(self._ar_cart(spr))
        u2_0 = jnp.float32(0.2)

        def lc(u2):
            return self._call_single_phase(
                grad_model, rotated,
                ldc_coeffs_quiet=jnp.array([[jnp.float32(0.4), u2]]),
            )

        jax_g = float(jax.grad(lc)(u2_0))
        fd    = self._fd(lambda u: float(lc(jnp.float32(u))), float(u2_0), self._H_LDC)
        self._check_tight("u2", jax_g, fd, rtol=5e-2)

    # ---- equatorial velocity ------------------------------------------------

    def test_ve_gradient_finite(self, grad_model):
        """
        d(LC)/d(ve) must be finite. Rotational broadening is now applied by
        Doppler-shifting the spectrum itself (see core.py's module
        docstring), so this gradient is only non-trivial when flux_quiet has
        real spectral structure -- for a flat spectrum (as in grad_model)
        shifting the sampling point changes nothing, and the true gradient
        is exactly zero. This test only checks finiteness (no NaN/inf from
        the interpolation's autodiff path); the FD-agreement case is tested
        with a structured spectrum in test_ve_gradient_fd_agreement instead.
        """
        spr     = grad_model["star_pixel_rad"]
        y_disc  = grad_model["y_disc"]
        coords  = jnp.asarray(np.arange(grad_model["n"]) - grad_model["n"] // 2, dtype=jnp.float32)
        rotated = jax.vmap(
            lambda c: rotate_active_region(c, jnp.float32(0.0), jnp.float32(90.0))
        )(self._ar_cart(spr))

        def lc(ve):
            vel_row = coords / spr * (ve / self._C)
            return self._call_single_phase(grad_model, rotated, vel_row=vel_row)

        g = float(jax.grad(lc)(jnp.float32(2.0)))
        assert np.isfinite(g)

    def test_ve_gradient_fd_agreement(self):
        """
        With a spectrum that has real structure (an absorption line), a
        rotational Doppler shift measurably changes the local contrast, so
        d(LC)/d(ve) is non-zero and can be checked against FD. The intrinsic
        signal is tiny (~1e-9), so a wide wavelength grid and a fairly large
        h are used to keep the FD estimate above float32 noise.
        """
        wl = np.linspace(500.0, 600.0, 200, dtype=np.float64)
        flux_quiet_line = (1.0 - 0.8 * np.exp(-((wl - 555.0) / 1.0) ** 2)).astype(np.float64)
        model = build_model(
            wavelength=wl, flux_quiet=flux_quiet_line,
            params=dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0),
            phases_rot=np.array([0.0]), stellar_grid_size=self._SPR, ve=0.0,
        )
        nwave    = len(wl)
        n_coeffs = 2
        n_mu     = model["mu_profile_pts"].shape[0]
        flux_active_wl = jnp.asarray(np.full((1, nwave), 0.7))
        spr = model["star_pixel_rad"]
        coords = jnp.asarray(np.arange(model["n"]) - model["n"] // 2, dtype=jnp.float32)
        rotated = jax.vmap(
            lambda c: rotate_active_region(c, jnp.float32(0.0), jnp.float32(90.0))
        )(self._ar_cart(spr))

        def lc(ve):
            vel_row = coords / spr * (ve / self._C)
            fval, _, _ = _compute_single_phase(
                rotated, jnp.array([0.0, 0.0, -1e10]),
                wavelength=model["wavelength"], flux_quiet=model["flux_quiet"],
                flux_active=flux_active_wl,
                ldc_coeffs_quiet=model["ldc_coeffs"],
                ldc_coeffs_active=jnp.broadcast_to(model["ldc_coeffs"][None, :, :], (1, nwave, n_coeffs)),
                I_profile_quiet=model["I_profile"],
                I_profile_active=jnp.broadcast_to(model["I_profile"][None, :, :], (1, nwave, n_mu)),
                mu_profile_pts=model["mu_profile_pts"],
                x_disc=model["x_disc"], y_disc=model["y_disc"], mu_disc=model["mu_disc"],
                row_idx=model["row_idx"], vel_row=vel_row,
                star_pixel_rad=spr, total_pixels=model["total_pixels"],
                arsize_rads=jnp.array([jnp.deg2rad(jnp.float32(self._ARSIZE))]),
                ar_smoothness=jnp.array([_SM]),
                k=jnp.float32(0.0), ldc_mode=model["ldc_mode"],
                plot_map_wavelength=model["plot_map_wavelength"], n=model["n"],
                flat_indices=model["flat_indices"],
            )
            return jnp.sum(fval)

        ve0 = jnp.float32(50.0)
        h   = jnp.float32(50.0)
        jax_g = float(jax.grad(lc)(ve0))
        fd    = float((lc(ve0 + h) - lc(ve0 - h)) / (2.0 * h))
        assert np.isfinite(jax_g) and abs(jax_g) > 0
        assert np.isfinite(fd) and abs(fd) > 0
        ratio = jax_g / fd
        assert 0.5 <= ratio <= 2.0, (
            f"'ve' JAX ({jax_g:.4g}) vs FD ({fd:.4g}), ratio={ratio:.3f}"
        )

    # ---- rotation period ----------------------------------------------------

    def test_prot_gradient_finite_and_nonzero(self, grad_model):
        spr     = grad_model["star_pixel_rad"]
        ar_cart = self._ar_cart(spr, lat_deg=0.0, long_deg=0.0)
        t = jnp.float32(2.0)

        def lc(P_rot):
            phase   = (t / P_rot * 360.0) % 360.0
            rotated = jax.vmap(
                lambda c: rotate_active_region(c, phase, jnp.float32(90.0))
            )(ar_cart)
            return self._call_single_phase(grad_model, rotated)

        g = float(jax.grad(lc)(jnp.float32(25.0)))
        assert np.isfinite(g)
        assert abs(g) > 0

    def test_prot_gradient_fd_agreement(self, grad_model):
        spr     = grad_model["star_pixel_rad"]
        ar_cart = self._ar_cart(spr, lat_deg=0.0, long_deg=0.0)
        t  = jnp.float32(2.0)
        P0 = jnp.float32(25.0)

        def lc(P_rot):
            phase   = (t / P_rot * 360.0) % 360.0
            rotated = jax.vmap(
                lambda c: rotate_active_region(c, phase, jnp.float32(90.0))
            )(ar_cart)
            return self._call_single_phase(grad_model, rotated)

        jax_g = float(jax.grad(lc)(P0))
        fd    = self._fd(lambda p: float(lc(jnp.float32(p))), float(P0), self._H_PROT)
        self._check("P_rot", jax_g, fd, self._H_PROT)
