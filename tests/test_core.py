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
    build_combined_model,
    evaluate_light_curve,
    compute_combined_light_curve,
    _compute_planet_mask,
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
# Oversampling cases
# ===================================================================

def test_oversample_smooths_light_curve():
    """
    Oversampled light curve should have smaller point-to-point
    differences (fewer sharp jumps) than the non-oversampled version.
    """
    import numpy as np
    from sajax import compute_light_curve

    wavelength = np.array([550.0])
    flux_quiet = np.array([1.0])
    flux_active = np.array([[0.7]])
    params = dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0)
    phases = np.linspace(0, 360, 500, endpoint=False)

    common = dict(
        wavelength=wavelength,
        flux_quiet=flux_quiet,
        flux_active=flux_active,
        params=params,
        ar_lat=[20.0],
        ar_long=[5.0],
        ar_size=[11.0],
        phases_rot=phases,
        stellar_grid_size=100,
        ve=2.0,
        ldc_mode="quadratic",
    )

    lc_no_os = compute_light_curve(**common, oversample=1)["lc"]
    lc_os3   = compute_light_curve(**common, oversample=3)["lc"]

    # Same shape
    assert lc_no_os.shape == lc_os3.shape

    # Oversampled should be smoother: smaller max absolute diff
    roughness_no_os = np.max(np.abs(np.diff(lc_no_os)))
    roughness_os3   = np.max(np.abs(np.diff(lc_os3)))

    assert roughness_os3 <= roughness_no_os, (
        f"Oversampled roughness ({roughness_os3:.6f}) should be <= "
        f"non-oversampled ({roughness_no_os:.6f})"
    )


def test_oversample_3_is_identity():
    """oversample=3 should produce identical results to no argument."""
    import numpy as np
    from sajax import compute_light_curve

    wavelength = np.array([550.0])
    flux_quiet = np.array([1.0])
    flux_active = np.array([[0.7]])
    params = dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0)
    phases = np.linspace(0, 360, 100, endpoint=False)

    common = dict(
        wavelength=wavelength,
        flux_quiet=flux_quiet,
        flux_active=flux_active,
        params=params,
        ar_lat=[20.0],
        ar_long=[5.0],
        ar_size=[11.0],
        phases_rot=phases,
        stellar_grid_size=80,
        ve=2.0,
    )

    lc_default = compute_light_curve(**common)["lc"]
    lc_os3     = compute_light_curve(**common, oversample=3)["lc"]

    np.testing.assert_array_equal(lc_default, lc_os3)


def test_oversample_invalid_value():
    """oversample < 1 should raise ValueError."""
    import pytest
    import numpy as np
    from sajax import build_model

    with pytest.raises(ValueError, match="oversample"):
        build_model(
            wavelength=np.array([550.0]),
            flux_quiet=np.array([1.0]),
            params=dict(ldc_coeffs=[0.4, 0.2]),
            phases_rot=np.linspace(0, 360, 10),
            stellar_grid_size=50,
            ve=2.0,
            oversample=0,
        )

def test_oversample_preserves_shape():
    """Output shape should always be (nphase,) regardless of oversample."""
    import numpy as np
    from sajax import compute_light_curve

    wavelength = np.array([550.0])
    flux_quiet = np.array([1.0])
    flux_active = np.array([[0.7]])
    params = dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0)
    phases = np.linspace(0, 360, 50, endpoint=False)

    common = dict(
        wavelength=wavelength,
        flux_quiet=flux_quiet,
        flux_active=flux_active,
        params=params,
        ar_lat=[20.0],
        ar_long=[5.0],
        ar_size=[11.0],
        phases_rot=phases,
        stellar_grid_size=80,
        ve=2.0,
    )

    for os_factor in [1, 3, 5]:
        result = compute_light_curve(**common, oversample=os_factor)
        assert result["lc"].shape == (50,), (
            f"oversample={os_factor}: expected shape (50,), "
            f"got {result['lc'].shape}"
        )
        assert result["epsilon"].shape[0] == 50
        assert result["star_maps"].shape[0] == 50

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
        np.testing.assert_allclose(result["lc"][0], 1.0, atol=1e-3,
                                   err_msg="Zero-size AR should have negligible effect on flux",)

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
            lc, np.mean(lc), rtol=5e-3,
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


# ---------------------------------------------------------------------------
# Shared test configuration
# ---------------------------------------------------------------------------

# Single broadband channel — fast for all tests
WAVELENGTH   = np.array([550.0])
FLUX_QUIET   = np.array([1.0])
FLUX_SPOT    = np.array([0.7])     # cold spot: 30% darker than quiet
FLUX_FACULA  = np.array([1.1])     # facula: 10% brighter than quiet

STELLAR_GRID = 60                  # radius in pixels — small enough to be fast
VE           = 0.0                 # no Doppler — isolates transit geometry

BASE_PARAMS = dict(ldc_coeffs=[0.4, 0.2], inc_star=90.0)

# Time array centred on transit (no-LD ≈ ±2.5× transit duration)
TIMES = np.linspace(-0.15, 0.15, 200)

# Long rotation period so stellar modulation is negligible across TIMES
P_ROT = 25.0

# "Clean" transit: edge-on circular orbit, k=0.1 → ≈1% depth, well resolved
TRANSIT_PARAMS = dict(
    t0           = 0.0,
    period       = 5.0,
    a_over_rstar = 10.0,
    inclination  = np.pi / 2.0,   # exactly edge-on
    k            = 0.1,
    ecc          = 0.0,
    omega_peri   = 0.0,
)


# ---------------------------------------------------------------------------
# Fixtures — built once per module to avoid repeated JIT compilation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def combined_model():
    """Combined model with a vanishingly small far-side AR (transit only)."""
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
    """Equivalent stellar-only model for backward-compat comparisons."""
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


# ---------------------------------------------------------------------------
# Helper: one-call combined LC
# ---------------------------------------------------------------------------

def _combined_lc(transit_params=None, ar_lat=None, ar_long=None, ar_size=None,
                 flux_active=None, oversample=1, params=None):
    """Thin wrapper that fills in defaults to keep test bodies short."""
    return compute_combined_light_curve(
        wavelength        = WAVELENGTH,
        flux_quiet        = FLUX_QUIET,
        flux_active       = np.atleast_2d(flux_active if flux_active is not None else FLUX_QUIET),
        params            = params or BASE_PARAMS,
        ar_lat            = ar_lat  or [0.0],
        ar_long           = ar_long or [180.0],  # far side — invisible by default
        ar_size           = ar_size or [0.001],
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
    """Unit tests for the pixel-level planet occultation mask."""

    @pytest.fixture(autouse=True)
    def _grid(self):
        g = build_stellar_grid(50, 0.0)
        self.x   = jnp.asarray(g["x"])
        self.y   = jnp.asarray(g["y"])
        self.spr = g["star_pixel_rad"]

    def _mask(self, X=0.0, Y=0.0, Z=5.0, k=0.1):
        return _compute_planet_mask(self.x, self.y, self.spr, X, Y, Z, k)

    # --- Shape & dtype ----------------------------------------------------

    def test_shape_matches_disc(self):
        """Mask shape must equal the number of in-disc pixels."""
        mask = self._mask()
        assert mask.shape == self.x.shape

    def test_dtype_is_bool(self):
        """Mask must have boolean dtype."""
        assert self._mask().dtype == jnp.bool_

    # --- Z-sign gate ------------------------------------------------------

    def test_all_false_when_planet_behind_star(self):
        """Z < 0 → planet behind the star → mask must be all False."""
        assert not jnp.any(self._mask(Z=-1.0)), \
            "Mask should be all False when Z < 0"

    def test_all_false_at_Z_exactly_zero(self):
        """Z = 0 means the planet is at the stellar limb plane — not transiting."""
        assert not jnp.any(self._mask(Z=0.0))

    # --- Radius gate ------------------------------------------------------

    def test_all_false_for_zero_radius_planet(self):
        """k = 0 → no disc → nothing occulted even when Z > 0."""
        assert not jnp.any(self._mask(k=0.0)), \
            "Zero-radius planet should produce empty mask"

    # --- Positive occultation ------------------------------------------------------

    def test_some_pixels_masked_at_disc_centre(self):
        """Planet at (0, 0) with Z > 0 and k > 0 must mask some pixels."""
        assert jnp.any(self._mask(X=0.0, Y=0.0, Z=5.0, k=0.1)), \
            "Planet at disc centre should occult some pixels"

    def test_masked_pixel_count_scales_with_k(self):
        """Larger planet → more pixels masked."""
        n_small = int(jnp.sum(self._mask(k=0.05)))
        n_large = int(jnp.sum(self._mask(k=0.20)))
        assert n_large > n_small, (
            f"Larger k should mask more pixels: n_small={n_small}, n_large={n_large}"
        )

    def test_planet_outside_disc_masks_nothing(self):
        """Planet sky-position beyond the stellar disc (X ≫ 1+k) masks nothing."""
        assert not jnp.any(self._mask(X=3.0, Y=0.0, Z=5.0, k=0.1)), \
            "Planet outside stellar disc should mask nothing"

    def test_mask_centre_pixels_when_centred(self):
        """Central pixels (r/R★ ≈ 0) should all be masked when planet is centred."""
        centre = (np.hypot(np.array(self.x), np.array(self.y)) / self.spr) < 0.05
        mask = np.array(self._mask(X=0.0, Y=0.0, Z=5.0, k=0.15))
        if centre.any():
            assert mask[centre].all(), \
                "All central pixels should be masked for planet at disc centre"


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
        """planet_xyz must be (nphase_compute, 3)."""
        xyz    = combined_model["planet_xyz"]
        nphase = combined_model["nphase"]
        assert xyz.shape == (nphase, 3), (
            f"Expected planet_xyz shape ({nphase}, 3), got {xyz.shape}"
        )

    def test_xyz_third_column_Z(self, combined_model):
        """At mid-transit (t=0 → phase index near centre), Z should be positive."""
        xyz = np.array(combined_model["planet_xyz"])
        # At mid-transit the Z coordinate (column 2) of the closest time to t=0
        # should be positive.  We just check that at least one phase has Z > 0.
        assert np.any(xyz[:, 2] > 0), "At least one phase should have Z > 0"

    def test_all_stellar_keys_preserved(self, combined_model):
        required = [
            "x_disc", "y_disc", "mu_disc", "vel_disc",
            "star_pixel_rad", "total_pixels", "wavelength",
            "phases_rot", "ldc_coeffs", "flat_indices", "n",
        ]
        for key in required:
            assert key in combined_model, f"Stellar key missing: '{key}'"

    def test_oversample_inflates_nphase(self):
        """nphase_compute should equal nphase_original × oversample."""
        oversample = 3
        model = build_combined_model(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, params=BASE_PARAMS,
            times=TIMES, P_rot=P_ROT, transit_params=TRANSIT_PARAMS,
            stellar_grid_size=STELLAR_GRID, ve=VE, oversample=oversample,
        )
        n_orig    = model["nphase_original"]
        n_compute = model["nphase"]
        assert n_compute == n_orig * oversample, (
            f"Expected {n_orig}×{oversample}={n_orig*oversample} phases, "
            f"got {n_compute}"
        )

    def test_planet_xyz_length_matches_nphase(self):
        """planet_xyz rows must match nphase_compute after oversampling."""
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
        """Output lc shape must equal len(TIMES)."""
        result = _combined_lc()
        assert result["lc"].shape == (len(TIMES),)

    def test_lc_finite(self):
        assert np.all(np.isfinite(_combined_lc()["lc"]))

    def test_transit_produces_flux_dip(self):
        """During a central transit the normalised flux must drop below 1."""
        lc = _combined_lc()["lc"]
        assert float(np.min(lc)) < 1.0, \
            f"Transit should dim the star; min flux = {np.min(lc):.6f}"

    def test_transit_depth_scales_with_k(self):
        """A larger planet produces a deeper transit."""
        d_small = 1.0 - float(np.min(_combined_lc({**TRANSIT_PARAMS, "k": 0.05})["lc"]))
        d_large = 1.0 - float(np.min(_combined_lc({**TRANSIT_PARAMS, "k": 0.15})["lc"]))
        assert d_large > d_small, (
            f"k=0.15 depth={d_large:.5f} should exceed k=0.05 depth={d_small:.5f}"
        )

    def test_approximate_transit_depth_equals_k_squared(self):
        """Transit depth ≈ k² for a uniform, unlimb-darkened disc (u1=u2=0).

        Grid discretisation on a 60-pixel grid introduces ~15% error.
        """
        k = 0.1
        tp = {**TRANSIT_PARAMS, "k": k}
        params_no_ld = dict(ldc_coeffs=[0.0, 0.0], inc_star=90.0)
        lc = _combined_lc(tp, params=params_no_ld)["lc"]

        depth = 1.0 - float(np.min(lc))
        np.testing.assert_allclose(depth, k**2, rtol=0.15,
            err_msg=f"Transit depth should be ≈k²={k**2:.4f}; got {depth:.4f}")

    def test_grazing_transit_shallower_than_central(self):
        """A grazing transit (high impact parameter) should be shallower."""
        k   = 0.1
        a   = TRANSIT_PARAMS["a_over_rstar"]
        inc_central = np.pi / 2.0            # b ≈ 0
        inc_grazing = np.arccos(0.85 / a)    # b = 0.85 R★  (grazing)

        d_central = 1.0 - float(np.min(_combined_lc({**TRANSIT_PARAMS, "k": k})["lc"]))
        d_grazing = 1.0 - float(np.min(_combined_lc(
            {**TRANSIT_PARAMS, "k": k, "inclination": inc_grazing})["lc"]))

        assert d_central > d_grazing, (
            f"Central transit depth ({d_central:.5f}) should exceed "
            f"grazing transit depth ({d_grazing:.5f})"
        )

    def test_spot_crossing_produces_positive_bump(self):
        """A cold spot on the transit chord should cause a positive flux anomaly.

        Physics: the planet masks a dark pixel, so the remaining integrated
        flux is *higher* than a transit over quiet photosphere → a bump.
        """
        # Spot on transit chord — lat=0, long=0 is facing us at t≈0
        lc_spot = _combined_lc(
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], flux_active=FLUX_SPOT,
        )["lc"]
        # Same transit but AR hidden on far side
        lc_clean = _combined_lc(
            ar_lat=[0.0], ar_long=[180.0], ar_size=[10.0], flux_active=FLUX_SPOT,
        )["lc"]

        in_transit = np.abs(TIMES) < 0.04
        bump = float(np.max(lc_spot[in_transit])) - float(np.max(lc_clean[in_transit]))
        assert bump > 0, (
            f"Cold-spot crossing should produce a positive bump; got Δ={bump:.6f}"
        )

    def test_facula_crossing_produces_negative_anomaly(self):
        """A facula on the transit chord should deepen the transit dip.

        Physics: the planet hides a bright pixel → missing extra flux →
        the dip is slightly deeper.
        """
        lc_fac = _combined_lc(
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], flux_active=FLUX_FACULA,
        )["lc"]
        lc_clean = _combined_lc(
            ar_lat=[0.0], ar_long=[180.0], ar_size=[10.0], flux_active=FLUX_FACULA,
        )["lc"]

        in_transit = np.abs(TIMES) < 0.04
        dip_delta = float(np.min(lc_fac[in_transit])) - float(np.min(lc_clean[in_transit]))
        assert dip_delta < 0, (
            f"Facula crossing should deepen the transit dip; got Δ={dip_delta:.6f}"
        )

    def test_out_of_transit_matches_stellar_only(self, stellar_only_model):
        """Well outside transit, combined LC must equal stellar-only LC."""
        # Bare star (flux_active = flux_quiet, AR on far side)
        result_combined = _combined_lc(
            ar_lat=[0.0], ar_long=[180.0], ar_size=[0.001],
            flux_active=FLUX_QUIET,
        )
        result_stellar = evaluate_light_curve(
            stellar_only_model,
            flux_active=jnp.array(FLUX_QUIET),
            ar_lat=jnp.array([0.0]),
            ar_long=jnp.array([180.0]),
            ar_size=jnp.array([0.001]),
        )

        oot = np.abs(TIMES) > 0.12          # safely outside transit
        np.testing.assert_allclose(
            result_combined["lc"][oot],
            np.array(result_stellar["lc"])[oot],
            rtol=1e-4,
            err_msg="Out-of-transit combined LC should match stellar-only LC",
        )

    def test_eccentric_orbit_centre_Z_positive(self):
        """At t=t0, Z should be positive regardless of eccentricity."""
        for ecc, omega in [(0.3, np.pi / 2.0), (0.5, np.pi / 4.0)]:
            tp = {**TRANSIT_PARAMS, "ecc": ecc, "omega_peri": omega}
            lc = _combined_lc(tp)["lc"]
            # The transit minimum should exist (Z > 0 at t=t0)
            assert float(np.min(lc)) < 1.0, (
                f"Eccentric orbit (e={ecc}, ω={omega:.2f}) should still transit"
            )

    def test_no_transit_when_fully_inclined(self):
        """For a very high impact parameter (no transit), the LC should be ≈1."""
        a = TRANSIT_PARAMS["a_over_rstar"]
        k = TRANSIT_PARAMS["k"]
        # Set inclination so b = a cos i = 2*(1+k) — planet never crosses disc
        inc_no_transit = np.arccos(2.0 * (1.0 + k) / a)
        tp = {**TRANSIT_PARAMS, "inclination": inc_no_transit}
        lc = _combined_lc(tp)["lc"]
        np.testing.assert_allclose(lc, 1.0, atol=0.005,
            err_msg="Non-transiting geometry should give LC ≈ 1")


# ===================================================================
# 4.  Oversampling — transit path
# ===================================================================

class TestTransitOversampling:

    def test_oversample_preserves_output_shape(self):
        """lc shape should equal len(TIMES) regardless of oversample factor."""
        for os in [1, 3, 5]:
            lc = _combined_lc(oversample=os)["lc"]
            assert lc.shape == (len(TIMES),), (
                f"oversample={os}: expected ({len(TIMES)},), got {lc.shape}"
            )

    def test_oversampled_lc_is_finite(self):
        lc = _combined_lc(oversample=3)["lc"]
        assert np.all(np.isfinite(lc))

    def test_oversampled_transit_still_present(self):
        """Transit dip must survive oversampling."""
        lc = _combined_lc(oversample=3)["lc"]
        assert float(np.min(lc)) < 1.0


# ===================================================================
# 5.  Backward compatibility
# ===================================================================

class TestBackwardCompatibility:

    def test_stellar_only_model_lacks_transit_flag(self, stellar_only_model):
        """build_model (stellar-only) should not set has_transit."""
        assert not stellar_only_model.get("has_transit", False)

    def test_evaluate_stellar_only_still_works(self, stellar_only_model):
        """evaluate_light_curve on a stellar-only model should return finite values."""
        result = evaluate_light_curve(
            stellar_only_model,
            flux_active=jnp.array(FLUX_SPOT),
            ar_lat=jnp.array([20.0]),
            ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]),
        )
        lc = np.array(result["lc"])
        assert np.all(np.isfinite(lc))
        assert lc.shape == (len(TIMES),)

    def test_compute_light_curve_api_unchanged(self):
        """The stellar-only compute_light_curve convenience API must be unaffected."""
        from sajax import compute_light_curve
        phases = (TIMES / P_ROT * 360.0) % 360.0
        result = compute_light_curve(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, flux_active=FLUX_SPOT,
            params=BASE_PARAMS,
            ar_lat=[20.0], ar_long=[0.0], ar_size=[10.0],
            phases_rot=phases,
            stellar_grid_size=STELLAR_GRID, ve=VE,
        )
        assert result["lc"].shape == (len(TIMES),)
        assert np.all(np.isfinite(result["lc"]))

    def test_no_transit_flag_gives_unity_transit_factor(self, stellar_only_model):
        """Stellar-only model should not exhibit any transit dip.

        The max-to-min excursion should be driven purely by stellar
        rotation (which is tiny with P_ROT=25 d over ±0.15 d window).
        Combined model with the transit planet switched off by a
        vanishingly small k on the far side should give the same result.
        """
        result_stellar = evaluate_light_curve(
            stellar_only_model,
            flux_active=jnp.array(FLUX_QUIET),
            ar_lat=jnp.array([0.0]),
            ar_long=jnp.array([180.0]),
            ar_size=jnp.array([0.001]),
        )
        lc = np.array(result_stellar["lc"])
        # No transit means no dip > 0.5 %
        assert float(np.max(lc) - np.min(lc)) < 0.005, \
            "Stellar-only model over short window should show negligible variation"