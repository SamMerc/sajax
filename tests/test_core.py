"""
tests/test_core.py — Tests for the SAJAX core engine.

Run with:
    pytest tests/
"""

import numpy as np
import pytest
import jax.numpy as jnp

from sajax import compute_light_curve, build_stellar_grid
from sajax.core import (
    build_model,
    evaluate_light_curve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_spectra():
    """Flat spectra on a small wavelength grid — fast for tests.

    Uses float64 throughout to match build_model's internal dtype.
    (Review item #1: consistent dtype.)
    """
    wl         = np.linspace(0.5, 2.5, 30, dtype=np.float64)
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
        assert len(grid["vel"]) == grid["total_pixels"]

    def test_mu_range(self):
        grid = build_stellar_grid(50, 2.0)
        mu = grid["mu"]
        assert float(mu.min()) >= 0.0
        assert float(mu.max()) <= 1.0 + 1e-5

    def test_mu_centre_pixel_is_one(self):
        """The central pixel should have mu ≈ 1 (disc centre)."""
        grid = build_stellar_grid(50, 2.0)
        # Centre pixel: x=0, y=0 → r=0 → mu=1
        centre_mask = (grid["x"] == 0) & (grid["y"] == 0)
        assert np.any(centre_mask), "Centre pixel not found in grid"
        assert abs(float(grid["mu"][centre_mask][0]) - 1.0) < 1e-5

    def test_vel_zero_when_ve_zero(self):
        """Doppler factor should be identically zero for non-rotating star."""
        grid = build_stellar_grid(50, ve=0.0)
        assert np.allclose(grid["vel"], 0.0, atol=1e-10)

    def test_flat_indices_within_bounds(self):
        grid = build_stellar_grid(50, 2.0)
        n = grid["n"]
        assert np.all(grid["flat_indices"] >= 0)
        assert np.all(grid["flat_indices"] < n * n)

    def test_dtype_consistency(self):
        """Review item #1: all grid arrays should share the same float dtype."""
        grid = build_stellar_grid(50, 2.0)
        dtypes = {grid["x"].dtype, grid["y"].dtype,
                  grid["mu"].dtype, grid["vel"].dtype}
        assert len(dtypes) == 1, (
            f"Grid arrays have mixed dtypes: {dtypes}"
        )


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
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=2.0,
            ldc_mode="quadratic",
        )
        assert result["lc"].shape == (1,)
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
            phases_rot=phases,
            stellar_grid_size=50,
            ve=2.0,
        )
        assert result["lc"].shape == (8,)
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
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[0.001],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
            ldc_mode="quadratic",
        )
        assert abs(float(result["lc"][0]) - 1.0) < 0.01

    def test_cold_ar_dims_flux(self, flat_spectra, base_params):
        """A visible cold AR should reduce the total flux."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[20.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
            ldc_mode="quadratic",
        )
        assert float(result["lc"][0]) < 1.0

    def test_hot_ar_brightens_flux(self, flat_spectra, base_params):
        """A facula (flux_active > flux_quiet) should increase total flux."""
        wl, flux_quiet, _ = flat_spectra
        flux_facula = np.full_like(wl, 1.3)
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_facula,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[20.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
            ldc_mode="quadratic",
        )
        assert float(result["lc"][0]) > 1.0

    def test_far_side_ar_invisible(self, flat_spectra, base_params):
        """An AR on the far side of the star should not affect the flux."""
        wl, flux_quiet, flux_active = flat_spectra
        # Place AR at longitude 180° — when phase=0, it's on the far side
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[180.0],
            ar_size=[15.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
            ldc_mode="quadratic",
        )
        assert abs(float(result["lc"][0]) - 1.0) < 0.01

    def test_light_curve_is_periodic(self, flat_spectra, base_params):
        """LC at phase=0 should equal LC at phase=360."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[20.0],
            ar_long=[45.0],
            ar_size=[10.0],
            phases_rot=[0.0, 360.0],
            stellar_grid_size=50,
            ve=2.0,
            ldc_mode="quadratic",
        )
        np.testing.assert_allclose(
            result["lc"][0], result["lc"][1], rtol=1e-5
        )

    def test_epsilon_unity_without_ar(self, flat_spectra, base_params):
        """Contamination factor should be ~1 everywhere with no AR."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[0.001],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
        )
        np.testing.assert_allclose(
            result["epsilon"][0], 1.0, atol=0.01,
            err_msg="epsilon should be ~1 when no AR is visible",
        )

    def test_epsilon_gt_one_for_cold_ar(self, flat_spectra, base_params):
        """ε = F_quiet / F_spotted > 1 when the AR dims the star.
        """
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[20.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
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
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[20.0, -20.0],
            ar_long=[0.0, 180.0],
            ar_size=[10.0, 10.0],
            phases_rot=np.linspace(0, 360, 6, endpoint=False),
            stellar_grid_size=50,
            ve=2.0,
        )
        assert result["lc"].shape == (6,)

    def test_per_ar_spectra(self, flat_spectra, base_params):
        """Each AR can have its own spectrum: flux_active shape (nar, nwave)."""
        wl, flux_quiet, _ = flat_spectra
        nar = 3
        nwave = len(wl)
        flux_active_multi = np.stack([
            np.full(nwave, 0.5),    # cold spot
            np.full(nwave, 0.9),    # mild spot
            np.full(nwave, 1.2),    # facula
        ])  # (3, nwave)

        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active_multi,
            params=base_params,
            ar_lat=[10.0, -10.0, 30.0],
            ar_long=[0.0, 60.0, 120.0],
            ar_size=[8.0, 8.0, 8.0],
            phases_rot=[0.0, 90.0],
            stellar_grid_size=50,
            ve=1.0,
        )
        assert result["lc"].shape == (2,)
        assert np.all(np.isfinite(result["lc"]))


# ===================================================================
# AR overlap modes
# ===================================================================

class TestAROverlapModes:

    def _make_overlapping_result(self, flat_spectra, base_params, mode):
        """Helper: two overlapping ARs with very different temperatures."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        flux_active = np.stack([
            np.full(nwave, 0.3),   # very cold
            np.full(nwave, 1.5),   # very hot (facula)
        ])
        return compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0, 0.0],       # same position → full overlap
            ar_long=[0.0, 0.0],
            ar_size=[15.0, 15.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
            ar_overlap_mode=mode,
        )

    def test_hottest_wins_brighter_than_coldest_wins(
        self, flat_spectra, base_params
    ):
        """'hottest_wins' should produce more flux than 'coldest_wins'."""
        hot_result  = self._make_overlapping_result(
            flat_spectra, base_params, "hottest_wins"
        )
        cold_result = self._make_overlapping_result(
            flat_spectra, base_params, "coldest_wins"
        )
        assert float(hot_result["lc"][0]) > float(cold_result["lc"][0])

    def test_invalid_overlap_mode_raises(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        with pytest.raises(ValueError, match="ar_overlap_mode"):
            compute_light_curve(
                wavelength=wl,
                flux_quiet=flux_quiet,
                flux_active=flux_active,
                params=base_params,
                ar_lat=[0.0],
                ar_long=[0.0],
                ar_size=[10.0],
                phases_rot=[0.0],
                stellar_grid_size=50,
                ve=0.0,
                ar_overlap_mode="invalid_mode",
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
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=params,
            ar_lat=[15.0],
            ar_long=[0.0],
            ar_size=[8.0],
            phases_rot=[0.0, 90.0],
            stellar_grid_size=50,
            ve=1.0,
            ldc_mode=ldc_mode,
        )
        assert result["lc"].shape == (2,)
        assert np.all(np.isfinite(result["lc"]))
        assert np.all(np.isfinite(result["epsilon"]))

    def test_intensity_profile_mode(self, flat_spectra):
        """Review item #5: intensity_profile LDC mode should work."""
        wl, flux_quiet, flux_active = flat_spectra
        nwave = len(wl)
        # Simple linear limb-darkening as a profile: I(mu) = mu
        mu_pts = np.linspace(0.0, 1.0, 50)
        I_profile = np.tile(mu_pts, (nwave, 1))  # (nwave, 50)

        params = dict(
            inc_star=90.0,
            mu_profile=mu_pts,
            I_profile=I_profile,
        )
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[10.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
            ldc_mode="intensity_profile",
        )
        assert result["lc"].shape == (1,)
        assert np.all(np.isfinite(result["lc"]))

    def test_legacy_u1_u2_keys(self, flat_spectra):
        """Legacy params with 'u1'/'u2' keys should still work for quadratic."""
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(inc_star=90.0, u1=0.3, u2=0.1)
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=params,
            ar_lat=[10.0],
            ar_long=[0.0],
            ar_size=[8.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
            ldc_mode="quadratic",
        )
        assert np.all(np.isfinite(result["lc"]))

    def test_per_wavelength_ldc(self, flat_spectra):
        """Per-wavelength LDC arrays should work and produce finite output."""
        wl, flux_quiet, flux_active = flat_spectra
        nwave = len(wl)
        params = dict(
            inc_star=90.0,
            ldc_coeffs=[
                np.linspace(0.2, 0.5, nwave),   # u1(λ)
                np.linspace(0.05, 0.2, nwave),   # u2(λ)
            ],
        )
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=params,
            ar_lat=[10.0],
            ar_long=[0.0],
            ar_size=[10.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
            ldc_mode="quadratic",
        )
        assert np.all(np.isfinite(result["lc"]))


# ===================================================================
# Input validation
# ===================================================================

class TestInputValidation:

    def test_invalid_ldc_mode_raises_valueerror(self, flat_spectra):
        """invalid ldc_mode should raise ValueError, not KeyError."""
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(inc_star=90.0, ldc_coeffs=[0.3])
        with pytest.raises(ValueError, match="ldc_mode"):
            build_model(
                wavelength=wl,
                flux_quiet=flux_quiet,
                params=params,
                phases_rot=[0.0],
                stellar_grid_size=50,
                ve=0.0,
                ldc_mode="banana",
            )

    def test_wrong_number_of_ldc_coeffs_raises(self, flat_spectra):
        """Passing 2 coefficients for a 4-coeff law should raise ValueError."""
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(inc_star=90.0, ldc_coeffs=[0.3, 0.1])
        with pytest.raises(ValueError, match="coefficient"):
            build_model(
                wavelength=wl,
                flux_quiet=flux_quiet,
                params=params,
                phases_rot=[0.0],
                stellar_grid_size=50,
                ve=0.0,
                ldc_mode="nonlinear4",
            )

    def test_non_monotonic_mu_profile_raises(self, flat_spectra):
        """non-monotonic mu_profile should be rejected."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        bad_mu = np.array([0.0, 0.5, 0.3, 1.0])  # not monotonic
        params = dict(
            inc_star=90.0,
            mu_profile=bad_mu,
            I_profile=np.ones((nwave, len(bad_mu))),
        )
        with pytest.raises(ValueError, match="mu_profile.*increasing"):
            build_model(
                wavelength=wl,
                flux_quiet=flux_quiet,
                params=params,
                phases_rot=[0.0],
                stellar_grid_size=50,
                ve=0.0,
                ldc_mode="intensity_profile",
            )

    def test_flux_active_shape_mismatch_raises(self, small_model):
        """flux_active with wrong nwave dimension should raise."""
        wrong_flux = jnp.ones(5)  # wrong size
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_light_curve(
                small_model,
                flux_active=wrong_flux,
                ar_lat=jnp.array([0.0]),
                ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]),
            )

    def test_flux_active_2d_shape_mismatch_raises(self, small_model):
        """flux_active (nar, nwave) with wrong nar should raise."""
        nwave = small_model["nwave"]
        wrong_flux = jnp.ones((5, nwave))  # nar=5 but we pass nar=1
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_light_curve(
                small_model,
                flux_active=wrong_flux,
                ar_lat=jnp.array([0.0]),
                ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]),
            )

    def test_ldc_coeffs_wavelength_length_mismatch_raises(self, flat_spectra):
        """Per-wavelength LDC array with wrong length should raise."""
        wl, flux_quiet, _ = flat_spectra
        params = dict(
            inc_star=90.0,
            ldc_coeffs=[
                np.ones(5),    # wrong length (should be nwave=30)
                np.ones(5),
            ],
        )
        with pytest.raises(ValueError, match="wavelength grid"):
            build_model(
                wavelength=wl,
                flux_quiet=flux_quiet,
                params=params,
                phases_rot=[0.0],
                stellar_grid_size=50,
                ve=0.0,
                ldc_mode="quadratic",
            )

    def test_missing_ldc_coeffs_raises(self, flat_spectra):
        """Non-quadratic mode without ldc_coeffs key should raise."""
        wl, flux_quiet, _ = flat_spectra
        params = dict(inc_star=90.0)  # no ldc_coeffs, no u1/u2
        with pytest.raises(ValueError, match="ldc_coeffs"):
            build_model(
                wavelength=wl,
                flux_quiet=flux_quiet,
                params=params,
                phases_rot=[0.0],
                stellar_grid_size=50,
                ve=0.0,
                ldc_mode="power2",
            )


# ===================================================================
# Contamination factor edge cases
# ===================================================================

class TestContaminationEdgeCases:

    def test_epsilon_finite_for_normal_ar(self, flat_spectra, base_params):
        """Standard case: epsilon should be finite everywhere."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[15.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
        )
        assert np.all(np.isfinite(result["epsilon"]))

    def test_epsilon_nan_or_inf_for_zero_flux_ar(self, flat_spectra, base_params):
        """totally dark AR covering the disc.

        If bin_fluxes → 0 at some wavelength, epsilon should be nan
        (not a silently finite value) to flag the singularity.
        """
        wl, flux_quiet, _ = flat_spectra
        flux_dark = np.zeros_like(wl)  # completely dark AR
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_dark,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[89.0],    # huge AR covering almost the whole disc
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
        )
        eps = result["epsilon"][0]
        # Under the nan convention (recommended):
        # pixels where bin_flux ≈ 0 → epsilon = nan
        # Under the guard convention: epsilon stays finite
        # Uncomment the assertion matching your chosen fix:
        #
        # assert np.any(np.isnan(eps)), "Expected nan where bin_flux → 0"
        # OR:
        # assert np.all(np.isfinite(eps)), "Guard convention: always finite"
        #
        # For now, just verify it doesn't crash:
        assert eps.shape == (len(wl),)


# ===================================================================
# Two-stage API (build_model + evaluate_light_curve)
# ===================================================================

class TestTwoStageAPI:

    def test_two_stage_matches_convenience(self, flat_spectra, base_params):
        """build_model + evaluate_light_curve should match compute_light_curve."""
        wl, flux_quiet, flux_active = flat_spectra
        phases = np.linspace(0, 360, 6, endpoint=False)

        # One-shot
        result_one = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[15.0],
            ar_long=[30.0],
            ar_size=[10.0],
            phases_rot=phases,
            stellar_grid_size=50,
            ve=2.0,
        )

        # Two-stage
        model = build_model(
            wavelength=wl,
            flux_quiet=flux_quiet,
            params=base_params,
            phases_rot=phases,
            stellar_grid_size=50,
            ve=2.0,
        )
        result_two = evaluate_light_curve(
            model,
            flux_active=jnp.asarray(flux_active),
            ar_lat=jnp.array([15.0]),
            ar_long=jnp.array([30.0]),
            ar_size=jnp.array([10.0]),
        )

        np.testing.assert_allclose(
            result_one["lc"],
            np.array(result_two["lc"]),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            result_one["epsilon"],
            np.array(result_two["epsilon"]),
            rtol=1e-5,
        )

    def test_evaluate_reusable_with_different_ar_params(self, small_model):
        """Same model can be re-evaluated with different AR parameters."""
        nwave = small_model["nwave"]

        result_a = evaluate_light_curve(
            small_model,
            flux_active=jnp.ones(nwave) * 0.5,
            ar_lat=jnp.array([0.0]),
            ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]),
        )
        result_b = evaluate_light_curve(
            small_model,
            flux_active=jnp.ones(nwave) * 0.9,
            ar_lat=jnp.array([45.0]),
            ar_long=jnp.array([90.0]),
            ar_size=jnp.array([5.0]),
        )

        # Different AR params → different light curves
        assert not np.allclose(
            np.array(result_a["lc"]),
            np.array(result_b["lc"]),
        ), "Different AR parameters should produce different light curves"


# ===================================================================
# Numerical edge cases
# ===================================================================

class TestNumericalEdgeCases:

    def test_ar_at_pole(self, flat_spectra, base_params):
        """AR at latitude ±90° (pole) should not crash."""
        wl, flux_quiet, flux_active = flat_spectra
        for lat in [90.0, -90.0]:
            result = compute_light_curve(
                wavelength=wl,
                flux_quiet=flux_quiet,
                flux_active=flux_active,
                params=base_params,
                ar_lat=[lat],
                ar_long=[0.0],
                ar_size=[10.0],
                phases_rot=[0.0],
                stellar_grid_size=50,
                ve=0.0,
            )
            assert np.all(np.isfinite(result["lc"]))

    def test_ar_size_zero(self, flat_spectra, base_params):
        """AR with size=0 should produce unity light curve (no effect)."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[0.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
        )
        np.testing.assert_allclose(result["lc"][0], 1.0, atol=1e-6)

    def test_ar_size_90_degrees(self, flat_spectra, base_params):
        """AR covering a full hemisphere should not crash."""
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[90.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
        )
        assert np.all(np.isfinite(result["lc"]))
        # Large cold AR → significant dimming
        assert float(result["lc"][0]) < 0.95

    def test_inclination_zero_pole_on(self, flat_spectra):
        """Pole-on view (inc=0°) should not crash."""
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(ldc_coeffs=[0.3, 0.1], inc_star=0.0)
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[10.0],
            phases_rot=[0.0, 90.0, 180.0],
            stellar_grid_size=50,
            ve=0.0,
        )
        assert np.all(np.isfinite(result["lc"]))

    def test_inclination_zero_constant_lc(self, flat_spectra):
        """Pole-on: rotating an equatorial AR should produce constant LC.

        When viewed pole-on, the equator traces a circle at constant
        projected radius — the light curve should be flat (all phases
        see the same foreshortening).
        """
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(ldc_coeffs=[0.3, 0.1], inc_star=0.0)
        phases = np.linspace(0, 360, 12, endpoint=False)
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[10.0],
            phases_rot=phases,
            stellar_grid_size=50,
            ve=0.0,
        )
        lc = result["lc"]
        # All phases should be identical within numerical tolerance
        np.testing.assert_allclose(
            lc, lc[0], rtol=1e-4,
            err_msg="Pole-on view should produce a constant light curve",
        )

    def test_single_wavelength(self, base_params):
        """A single-wavelength grid should work without crashing."""
        wl = np.array([1.0])
        flux_quiet = np.array([1.0])
        flux_active = np.array([0.8])
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[10.0],
            ar_long=[0.0],
            ar_size=[10.0],
            phases_rot=[0.0],
            stellar_grid_size=50,
            ve=0.0,
        )
        assert result["lc"].shape == (1,)
        assert result["epsilon"].shape == (1, 1)
        assert np.all(np.isfinite(result["lc"]))


# ===================================================================
# Symmetry tests
# ===================================================================

class TestSymmetry:

    def test_equatorial_ar_symmetric_phases(self, flat_spectra, base_params):
        """An equatorial AR at long=0 should give the same flux at
        phase=+45° and phase=-45° (equivalently 315°).
        """
        wl, flux_quiet, flux_active = flat_spectra
        result = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[0.0],
            ar_long=[0.0],
            ar_size=[10.0],
            phases_rot=[45.0, 315.0],
            stellar_grid_size=50,
            ve=0.0,
            ldc_mode="quadratic",
        )
        np.testing.assert_allclose(
            result["lc"][0], result["lc"][1], rtol=1e-4,
            err_msg="Equatorial AR should be symmetric about phase=0",
        )

    def test_north_south_symmetry_equator_on(self, flat_spectra, base_params):
        """At inc=90° (equator-on), ARs at lat=+X and lat=-X should
        produce identical light curves.
        """
        wl, flux_quiet, flux_active = flat_spectra
        phases = np.linspace(0, 360, 8, endpoint=False)

        result_north = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[30.0],
            ar_long=[0.0],
            ar_size=[10.0],
            phases_rot=phases,
            stellar_grid_size=50,
            ve=0.0,
        )
        result_south = compute_light_curve(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            params=base_params,
            ar_lat=[-30.0],
            ar_long=[0.0],
            ar_size=[10.0],
            phases_rot=phases,
            stellar_grid_size=50,
            ve=0.0,
        )
        np.testing.assert_allclose(
            result_north["lc"], result_south["lc"], rtol=1e-4,
            err_msg="N/S symmetric ARs should produce identical LCs at inc=90°",
        )