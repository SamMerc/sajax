"""
tests/test_core.py — Tests for the SAJAX core engine.

Run with:
    pytest tests/
"""

import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import interpax

from sajax import quick_lc, build_stellar_grid
from sajax.core import (
    build_system,
    make_lc,
    flare_template,
    _compute_all_planets_mask,
    _compute_ar_shape,
    _compute_single_phase,
)
from sajax.planet import _compute_planet_mask
from sajax.geometry import rotate_active_region

C_KMS = 299_792.458  # speed of light [km/s]

# Default smoothness used throughout when a test doesn't care about the
# exact shape of the AR boundary -- sharp enough to be visually spot-like.
_SM = 20.0

# build_system/quick_lc now require times+P_rot (a real time axis)
# instead of a directly-given phases_rot array. Most tests below only ever
# cared about a rotational phase snapshot, not real elapsed time, so this
# holds a fixed reference P_rot and converts old phases_rot (degrees)
# values into an equivalent times array, purely to keep those call sites
# concise after the API change -- _t(phases_rot)/_P_ROT reconstructs
# exactly the phases_rot that used to be passed directly.
_P_ROT = 1.0

def _t(phases_rot):
    return np.atleast_1d(np.asarray(phases_rot, dtype=np.float64)) / 360.0 * _P_ROT


def _quiet_baseline(wl, flux_quiet, base_params, times, stellar_grid_size=50, ve=0.0,
                    ld_mode="quadratic"):
    """Disc-integrated flux of the bare quiet star (no AR, no transit) at
    each of ``times``. ``make_lc``/``quick_lc`` no longer normalise their
    output to this internally, so tests that care about the pre-AR/-transit
    baseline (rather than an assumed constant like 1.0) compute it directly
    here -- passing ``flux_active=flux_quiet`` gives contrast exactly 1
    everywhere regardless of the AR geometry, i.e. an exact quiet-star LC.
    """
    result = quick_lc(
        wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_quiet,
        **base_params,
        ar_lat=[0.0], ar_long=[0.0], ar_size=[1.0], ar_smoothness=[_SM],
        times=times, P_rot=_P_ROT, stellar_grid_size=stellar_grid_size,
        ve=ve, ld_mode=ld_mode,
    )
    return np.array(result[0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_spectra():
    """Flat spectra on a small wavelength grid — fast for tests.

    Uses float64 throughout to match build_system's internal dtype.
    """
    wl         = np.linspace(500.0, 600.0, 30, dtype=np.float64)
    flux_quiet = np.ones_like(wl)
    flux_active = np.full_like(wl, 0.7)
    return wl, flux_quiet, flux_active


@pytest.fixture
def base_params():
    return dict(
        ld_coeffs=[0.3, 0.1],   # quadratic law: [u1, u2]
        inc_star=90.0,
    )


@pytest.fixture
def small_model(flat_spectra, base_params):
    """Pre-built model for tests that need the two-stage API."""
    wl, flux_quiet, _ = flat_spectra
    return build_system(
        wavelength=wl,
        flux_quiet=flux_quiet,
        **base_params,
        times=_t(np.linspace(0, 360, 8, endpoint=False)), P_rot=_P_ROT,
        stellar_grid_size=50,
        ve=2.0,
        ld_mode="quadratic",
    )


# ===================================================================
# flare_template — Tovar Mendoza et al. (2022) flare time template
# ===================================================================

class TestFlareTemplate:

    _T = jnp.linspace(-0.5, 1.0, 4001)   # dense grid, 15 FWHM wide

    def test_peaks_near_tpeak_at_roughly_ampl(self):
        """The published fit peaks at ~0.95*ampl within ~0.1 FWHM of tpeak."""
        f = flare_template(self._T, 0.0, 0.1, ampl=2.0)
        i = int(jnp.argmax(f))
        assert abs(float(self._T[i])) < 0.01
        assert float(f[i]) == pytest.approx(2.0, rel=0.1)

    def test_fwhm_matches_requested_width(self):
        f = flare_template(self._T, 0.0, 0.1)
        above = self._T[f >= 0.5 * float(jnp.max(f))]
        assert float(above[-1] - above[0]) == pytest.approx(0.1, rel=0.15)

    def test_fast_rise_slow_decay(self):
        """Classical flare asymmetry: gone 2 FWHM before the peak, still
        visibly decaying 2 FWHM after."""
        assert float(flare_template(jnp.asarray(-0.2), 0.0, 0.1)) == pytest.approx(0.0, abs=1e-6)
        assert float(flare_template(jnp.asarray(0.2), 0.0, 0.1)) > 0.05

    def test_linear_in_ampl(self):
        f1 = flare_template(self._T, 0.0, 0.1, ampl=1.0)
        f3 = flare_template(self._T, 0.0, 0.1, ampl=3.0)
        np.testing.assert_allclose(np.array(f3), 3.0 * np.array(f1), rtol=1e-6)

    def test_gradients_finite(self):
        """The point of a JAX-native template: differentiable w.r.t. its
        parameters, for gradient-based flare fitting."""
        def total(params):
            tpeak, fwhm, ampl = params
            return jnp.sum(flare_template(self._T, tpeak, fwhm, ampl))
        g = jax.grad(total)(jnp.array([0.0, 0.1, 1.0]))
        assert bool(jnp.all(jnp.isfinite(g)))
        assert bool(jnp.any(g != 0.0))

    def test_long_baseline_finite(self):
        """Regression: far before the peak the exp factor used to overflow
        before erfc underflowed, giving inf * 0 = NaN values and NaN
        gradients on long baselines (e.g. a 14-min-FWHM flare in the
        middle of days of data)."""
        t = jnp.linspace(0.0, 2.0, 2001)
        f = flare_template(t, 1.0, 0.01)
        assert bool(jnp.all(jnp.isfinite(f)))
        # exactly zero well before the flare, still nonzero in the tail
        assert float(f[0]) == 0.0
        assert float(flare_template(jnp.asarray(1.05), 1.0, 0.01)) > 0.0

        def total(params):
            tpeak, fwhm, ampl = params
            return jnp.sum(flare_template(t, tpeak, fwhm, ampl))
        g = jax.grad(total)(jnp.array([1.0, 0.01, 1.0]))
        assert bool(jnp.all(jnp.isfinite(g)))
        assert bool(jnp.any(g != 0.0))


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
        assert len(grid["col_idx"]) == grid["total_pixels"]
        assert len(grid["vel_col"]) == grid["n"]

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

    def test_vel_col_zero_when_ve_zero(self):
        """Line-of-sight velocity should be identically zero for non-rotating star."""
        grid = build_stellar_grid(50, ve=0.0)
        assert np.allclose(grid["vel_col"], 0.0, atol=1e-10)

    def test_col_idx_within_bounds(self):
        grid = build_stellar_grid(50, 2.0)
        assert np.all(grid["col_idx"] >= 0)
        assert np.all(grid["col_idx"] < grid["n"])

    def test_col_idx_reconstructs_per_pixel_velocity(self):
        """col_idx + vel_col must exactly reproduce x/spr*(ve/c) per pixel --
        this is the column-based Doppler optimization's core correctness
        property: it must be an exact reformulation, not an approximation."""
        ve = 30.0
        grid = build_stellar_grid(50, ve=ve)
        expected = grid["x"] / grid["star_pixel_rad"] * (ve / C_KMS)
        reconstructed = grid["vel_col"][grid["col_idx"]]
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
        result = quick_lc(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            **base_params,
            ar_lat=[20.0],
            ar_long=[0.0],
            ar_size=[10.0],
            ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT,
            stellar_grid_size=50,
            ve=2.0,
            ld_mode="quadratic",
        )
        assert result[0].shape == (1, len(wl))
        assert result[1].ndim == 3

    def test_multi_phase(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        phases = np.linspace(0, 360, 8, endpoint=False)
        result = quick_lc(
            wavelength=wl,
            flux_quiet=flux_quiet,
            flux_active=flux_active,
            **base_params,
            ar_lat=[20.0],
            ar_long=[0.0],
            ar_size=[10.0],
            ar_smoothness=[_SM],
            times=_t(phases), P_rot=_P_ROT,
            stellar_grid_size=50,
            ve=2.0,
        )
        assert result[0].shape == (8, len(wl))
        assert result[1].shape[0] == 8


# ===================================================================
# Physical sanity checks
# ===================================================================

class TestPhysics:

    def test_no_ar_flux_is_unity(self, flat_spectra, base_params):
        """With a vanishingly small AR the light curve should match the quiet-star baseline."""
        wl, flux_quiet, flux_active = flat_spectra
        baseline = _quiet_baseline(wl, flux_quiet, base_params, _t([0.0]))
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[0.001], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )
        assert abs(float(result[0][0, 0]) - float(baseline[0, 0])) < 0.01

    def test_cold_ar_dims_flux(self, flat_spectra, base_params):
        """A visible cold AR should reduce the total flux below the quiet-star baseline."""
        wl, flux_quiet, flux_active = flat_spectra
        baseline = _quiet_baseline(wl, flux_quiet, base_params, _t([0.0]))
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[20.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )
        assert float(result[0][0, 0]) < float(baseline[0, 0])

    def test_hot_ar_brightens_flux(self, flat_spectra, base_params):
        """A facula (flux_active > flux_quiet) should increase total flux above the quiet-star baseline."""
        wl, flux_quiet, _ = flat_spectra
        flux_facula = np.full_like(wl, 1.3)
        baseline = _quiet_baseline(wl, flux_quiet, base_params, _t([0.0]))
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_facula,
            **base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[20.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )
        assert float(result[0][0, 0]) > float(baseline[0, 0])

    def test_far_side_ar_invisible(self, flat_spectra, base_params):
        """An AR on the far side of the star should not affect the flux."""
        wl, flux_quiet, flux_active = flat_spectra
        baseline = _quiet_baseline(wl, flux_quiet, base_params, _t([0.0]))
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[0.0], ar_long=[180.0], ar_size=[15.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )
        assert abs(float(result[0][0, 0]) - float(baseline[0, 0])) < 0.01

    def test_light_curve_is_periodic(self, flat_spectra, base_params):
        """LC at phase=0 should equal LC at phase=360."""
        wl, flux_quiet, flux_active = flat_spectra
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[20.0], ar_long=[45.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t([0.0, 360.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=2.0, ld_mode="quadratic",
        )
        np.testing.assert_allclose(result[0][0], result[0][1], rtol=1e-5)


# ===================================================================
# Multiple active regions
# ===================================================================

class TestMultiAR:

    def test_multi_ar_shapes(self, flat_spectra, base_params):
        """Two ARs should work and return correct shapes."""
        wl, flux_quiet, flux_active = flat_spectra
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[20.0, -20.0], ar_long=[0.0, 180.0], ar_size=[10.0, 10.0],
            ar_smoothness=[_SM, _SM],
            times=_t(np.linspace(0, 360, 6, endpoint=False)), P_rot=_P_ROT,
            stellar_grid_size=50, ve=2.0,
        )
        assert result[0].shape == (6, len(wl))

    def test_per_ar_spectra(self, flat_spectra, base_params):
        """Each AR can have its own spectrum: flux_active shape (nar, nwave)."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        flux_active_multi = np.stack([
            np.full(nwave, 0.5),    # cold spot
            np.full(nwave, 0.9),    # mild spot
            np.full(nwave, 1.2),    # facula
        ])  # (3, nwave)

        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active_multi,
            **base_params,
            ar_lat=[10.0, -10.0, 30.0], ar_long=[0.0, 60.0, 120.0],
            ar_size=[8.0, 8.0, 8.0], ar_smoothness=[_SM, _SM, _SM],
            times=_t([0.0, 90.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=1.0,
        )
        assert result[0].shape == (2, len(wl))
        assert np.all(np.isfinite(result[0]))


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
        combined = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet,
            flux_active=np.stack([np.full(nwave, 0.3), np.full(nwave, 0.7)]),
            **base_params,
            ar_lat=[0.0, 0.0], ar_long=[0.0, 0.0], ar_size=[15.0, 15.0],
            ar_smoothness=[50.0, 50.0],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        shallower_alone = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=np.full((1, nwave), 0.7),
            **base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[15.0], ar_smoothness=[50.0],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        assert float(combined[0][0, 0]) < float(shallower_alone[0][0, 0]), (
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
        params = dict(ld_coeffs=[0.0, 0.0], inc_star=90.0)
        model = build_system(
            wavelength=wl, flux_quiet=flux_quiet, **params,
            times=_t(np.array([0.0])), P_rot=_P_ROT, stellar_grid_size=60, ve=0.0,
        )
        flux_active = jnp.array([[0.3], [0.7]])
        result = make_lc(
            model, flux_active, jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]),
            jnp.array([15.0, 15.0]), jnp.array([50.0, 50.0]),
        )
        star_map = np.array(result[1][0])
        centre = star_map.shape[0] // 2
        expected = 1.0 - ((1 - 0.3) + (1 - 0.7))
        assert abs(star_map[centre, centre] - expected) < 1e-4

    def test_non_overlapping_ars_independent(self, flat_spectra, base_params):
        """Two well-separated ARs shouldn't measurably interact."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        combined = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet,
            flux_active=np.stack([np.full(nwave, 0.3), np.full(nwave, 1.5)]),
            **base_params,
            ar_lat=[0.0, 0.0], ar_long=[0.0, 180.0], ar_size=[10.0, 10.0],
            ar_smoothness=[_SM, _SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        cold_alone = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=np.full((1, nwave), 0.3),
            **base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        np.testing.assert_allclose(
            combined[0][0], cold_alone[0][0], rtol=1e-3,
            err_msg="far-side AR should not measurably affect the near-side AR",
        )


# ===================================================================
# LDC modes
# ===================================================================

class TestLDCModes:

    @pytest.mark.parametrize("ld_mode,ld_coeffs", [
        ("linear",      [0.3]),
        ("quadratic",   [0.3, 0.1]),
        ("power2",      [0.4, 0.6]),
        ("kipping3",    [0.2, 0.3, 0.1]),
        ("nonlinear4",  [0.1, 0.2, 0.15, 0.05]),
    ])
    def test_analytic_ld_modes(
        self, flat_spectra, base_params, ld_mode, ld_coeffs
    ):
        wl, flux_quiet, flux_active = flat_spectra
        params = {**base_params, "ld_coeffs": ld_coeffs}
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **params,
            ar_lat=[15.0], ar_long=[0.0], ar_size=[8.0], ar_smoothness=[_SM],
            times=_t([0.0, 90.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=1.0, ld_mode=ld_mode,
        )
        assert result[0].shape == (2, len(wl))
        assert np.all(np.isfinite(result[0]))

    def test_intensity_profile_mode(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        nwave = len(wl)
        mu_pts = np.linspace(0.0, 1.0, 50)
        I_profile = np.tile(mu_pts, (nwave, 1))  # (nwave, 50)

        params = dict(inc_star=90.0, mu_profile=mu_pts, I_profile=I_profile)
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="intensity_profile",
        )
        assert result[0].shape == (1, nwave)
        assert np.all(np.isfinite(result[0]))

    def test_quadratic_ld_coeffs_as_plain_list(self, flat_spectra):
        """ld_coeffs=[u1, u2] works directly for the quadratic law -- no
        dict/legacy-key indirection needed now that it's a plain kwarg."""
        wl, flux_quiet, flux_active = flat_spectra
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            inc_star=90.0, ld_coeffs=[0.3, 0.1],
            ar_lat=[10.0], ar_long=[0.0], ar_size=[8.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )
        assert np.all(np.isfinite(result[0]))

    def test_per_wavelength_ldc(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        nwave = len(wl)
        params = dict(
            inc_star=90.0,
            ld_coeffs=[np.linspace(0.2, 0.5, nwave), np.linspace(0.05, 0.2, nwave)],
        )
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **params,
            ar_lat=[10.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )
        assert np.all(np.isfinite(result[0]))

    def test_per_ar_ldc_differs_from_quiet(self, flat_spectra, base_params):
        """An AR with different LDC coefficients than quiet must give a
        different light curve than the default (quiet's own coefficients)."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        model = build_system(
            wavelength=wl, flux_quiet=flux_quiet, **base_params,
            times=_t(np.array([0.0])), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        common = dict(
            flux_active=jnp.asarray(np.full((1, nwave), 0.5)),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([15.0]), ar_smoothness=jnp.array([_SM]),
        )
        default_result = make_lc(model, **common)
        custom_ldc = jnp.asarray(np.tile([0.9, 0.05], (nwave, 1))[None, :, :])
        custom_result = make_lc(model, **common, ld_coeffs_active=custom_ldc)
        assert not np.allclose(default_result[0], custom_result[0])

    def test_default_ar_ldc_matches_quiet(self, flat_spectra, base_params):
        """Omitting ld_coeffs_active must exactly match explicitly passing
        the quiet photosphere's own coefficients, broadcast to all ARs."""
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        model = build_system(
            wavelength=wl, flux_quiet=flux_quiet, **base_params,
            times=_t(np.array([0.0])), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        common = dict(
            flux_active=jnp.asarray(np.full((1, nwave), 0.5)),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([15.0]), ar_smoothness=jnp.array([_SM]),
        )
        r_default = make_lc(model, **common)
        explicit_quiet_ldc = jnp.broadcast_to(model["ld_coeffs"][None, :, :], (1, nwave, 2))
        r_explicit = make_lc(model, **common, ld_coeffs_active=explicit_quiet_ldc)
        np.testing.assert_allclose(r_default[0], r_explicit[0], rtol=1e-6)


# ===================================================================
# Input validation
# ===================================================================

class TestInputValidation:

    def test_invalid_ld_mode_raises_valueerror(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(inc_star=90.0, ld_coeffs=[0.3])
        with pytest.raises(ValueError, match="ld_mode"):
            build_system(
                wavelength=wl, flux_quiet=flux_quiet, **params,
                times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="banana",
            )

    def test_wrong_number_of_ld_coeffs_raises(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(inc_star=90.0, ld_coeffs=[0.3, 0.1])
        with pytest.raises(ValueError, match="coefficient"):
            build_system(
                wavelength=wl, flux_quiet=flux_quiet, **params,
                times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="nonlinear4",
            )

    def test_non_monotonic_mu_profile_raises(self, flat_spectra):
        wl, flux_quiet, _ = flat_spectra
        nwave = len(wl)
        bad_mu = np.array([0.0, 0.5, 0.3, 1.0])
        params = dict(inc_star=90.0, mu_profile=bad_mu, I_profile=np.ones((nwave, len(bad_mu))))
        with pytest.raises(ValueError, match="mu_profile.*increasing"):
            build_system(
                wavelength=wl, flux_quiet=flux_quiet, **params,
                times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="intensity_profile",
            )

    def test_flux_active_shape_mismatch_raises(self, small_model):
        wrong_flux = jnp.ones(5)
        with pytest.raises(ValueError, match="flux_active"):
            make_lc(
                small_model, flux_active=wrong_flux,
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_flux_active_2d_shape_mismatch_raises(self, small_model):
        nwave = small_model["nwave"]
        wrong_flux = jnp.ones((5, nwave))
        with pytest.raises(ValueError, match="flux_active"):
            make_lc(
                small_model, flux_active=wrong_flux,
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_ar_smoothness_shape_mismatch_raises(self, small_model):
        nwave = small_model["nwave"]
        with pytest.raises(ValueError, match="ar_smoothness"):
            make_lc(
                small_model, flux_active=jnp.ones((2, nwave)),
                ar_lat=jnp.array([0.0, 10.0]), ar_long=jnp.array([0.0, 10.0]),
                ar_size=jnp.array([10.0, 10.0]),
                ar_smoothness=jnp.array([1.0, 2.0, 3.0]),  # wrong size
            )

    def test_ar_smoothness_below_one_raises(self, small_model):
        nwave = small_model["nwave"]
        with pytest.raises(ValueError, match="ar_smoothness"):
            make_lc(
                small_model, flux_active=jnp.ones(nwave),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([0.5]),
            )

    def test_ar_size_below_zero_raises(self, small_model):
        nwave = small_model["nwave"]
        with pytest.raises(ValueError, match="ar_size must be >= 0"):
            make_lc(
                small_model, flux_active=jnp.ones(nwave),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([-5.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_flux_active_below_zero_raises(self, small_model):
        nwave = small_model["nwave"]
        with pytest.raises(ValueError, match="flux_active must be >= 0"):
            make_lc(
                small_model, flux_active=jnp.full(nwave, -0.1),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_ar_smoothness_below_one_not_checked_under_jit(self, small_model):
        """The ar_smoothness>=1 check must not break jit/grad-based fitting
        (e.g. numpyro_ext), where ar_smoothness arrives as a Tracer with no
        concrete value available at trace time -- even one that would be
        rejected eagerly, like 0.5 here, must trace through untouched.
        """
        nwave = small_model["nwave"]

        @jax.jit
        def run(smoothness):
            lc, _ = make_lc(
                small_model, flux_active=jnp.ones(nwave),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([smoothness]),
            )
            return lc

        run(0.5)  # must not raise TracerBoolConversionError

    def test_ar_size_below_zero_not_checked_under_jit(self, small_model):
        """Same jit-safety requirement as above, for the ar_size>=0 check."""
        nwave = small_model["nwave"]

        @jax.jit
        def run(size):
            lc, _ = make_lc(
                small_model, flux_active=jnp.ones(nwave),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([size]), ar_smoothness=jnp.array([_SM]),
            )
            return lc

        run(-5.0)  # must not raise TracerBoolConversionError

    def test_flux_active_below_zero_not_checked_under_jit(self, small_model):
        """Same jit-safety requirement as above, for the flux_active>=0 check."""
        nwave = small_model["nwave"]

        @jax.jit
        def run(flux):
            lc, _ = make_lc(
                small_model, flux_active=jnp.full(nwave, flux),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )
            return lc

        run(-0.1)  # must not raise TracerBoolConversionError

    def test_ld_coeffs_active_shape_mismatch_raises(self, small_model):
        nwave = small_model["nwave"]
        with pytest.raises(ValueError, match="ld_coeffs_active"):
            make_lc(
                small_model, flux_active=jnp.ones((1, nwave)),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
                ld_coeffs_active=jnp.ones((1, nwave, 5)),  # wrong n_coeffs (quadratic expects 2)
            )

    def test_ld_coeffs_wavelength_length_mismatch_raises(self, flat_spectra):
        wl, flux_quiet, _ = flat_spectra
        params = dict(inc_star=90.0, ld_coeffs=[np.ones(5), np.ones(5)])
        with pytest.raises(ValueError, match="wavelength grid"):
            build_system(
                wavelength=wl, flux_quiet=flux_quiet, **params,
                times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
            )

    def test_missing_ld_coeffs_raises(self, flat_spectra):
        wl, flux_quiet, _ = flat_spectra
        params = dict(inc_star=90.0)
        with pytest.raises(ValueError, match="ld_coeffs"):
            build_system(
                wavelength=wl, flux_quiet=flux_quiet, **params,
                times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="power2",
            )

    def test_invalid_oversample_raises(self, flat_spectra, base_params):
        wl, flux_quiet, _ = flat_spectra
        with pytest.raises(ValueError, match="oversample"):
            build_system(
                wavelength=wl, flux_quiet=flux_quiet, **base_params,
                times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, oversample=0,
            )


# ===================================================================
# Active-region edge cases
# ===================================================================

class TestContaminationEdgeCases:

    def test_totally_dark_ar_does_not_crash(self, flat_spectra, base_params):
        """A totally dark AR covering almost the whole disc shouldn't crash."""
        wl, flux_quiet, _ = flat_spectra
        flux_dark = np.zeros_like(wl)
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_dark,
            **base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[89.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        assert result[0].shape == (1, len(wl))
        assert np.all(np.isfinite(result[0]))


# ===================================================================
# Two-stage API (build_system + make_lc)
# ===================================================================

class TestTwoStageAPI:

    def test_two_stage_matches_convenience(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        phases = np.linspace(0, 360, 6, endpoint=False)

        result_one = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[15.0], ar_long=[30.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t(phases), P_rot=_P_ROT, stellar_grid_size=50, ve=2.0,
        )

        model = build_system(
            wavelength=wl, flux_quiet=flux_quiet, **base_params,
            times=_t(phases), P_rot=_P_ROT, stellar_grid_size=50, ve=2.0,
        )
        result_two = make_lc(
            model,
            flux_active=jnp.asarray(flux_active),
            ar_lat=jnp.array([15.0]), ar_long=jnp.array([30.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )

        np.testing.assert_allclose(result_one[0], np.array(result_two[0]), rtol=1e-5)

    def test_evaluate_reusable_with_different_ar_params(self, small_model):
        nwave = small_model["nwave"]
        result_a = make_lc(
            small_model, flux_active=jnp.ones(nwave) * 0.5,
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )
        result_b = make_lc(
            small_model, flux_active=jnp.ones(nwave) * 0.9,
            ar_lat=jnp.array([45.0]), ar_long=jnp.array([90.0]),
            ar_size=jnp.array([5.0]), ar_smoothness=jnp.array([_SM]),
        )
        assert not np.allclose(np.array(result_a[0]), np.array(result_b[0]))


# ===================================================================
# Oversampling cases
# ===================================================================

def test_oversample_smooths_light_curve():
    wavelength = np.array([550.0])
    flux_quiet = np.array([1.0])
    flux_active = np.array([[0.7]])
    params = dict(ld_coeffs=[0.4, 0.2], inc_star=90.0)
    phases = np.linspace(0, 360, 500, endpoint=False)

    common = dict(
        wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_active,
        **params,
        ar_lat=[20.0], ar_long=[5.0], ar_size=[11.0], ar_smoothness=[_SM],
        times=_t(phases), P_rot=_P_ROT, stellar_grid_size=100, ve=2.0, ld_mode="quadratic",
    )

    # Single wavelength bin here -- lc is already squeezed to (nphase,).
    lc_no_os = np.asarray(quick_lc(**common, oversample=1)[0])
    lc_os3   = np.asarray(quick_lc(**common, oversample=3)[0])

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
    params = dict(ld_coeffs=[0.4, 0.2], inc_star=90.0)
    phases = np.linspace(0, 360, 100, endpoint=False)

    common = dict(
        wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_active,
        **params,
        ar_lat=[20.0], ar_long=[5.0], ar_size=[11.0], ar_smoothness=[_SM],
        times=_t(phases), P_rot=_P_ROT, stellar_grid_size=80, ve=2.0,
    )

    lc_default = quick_lc(**common)[0]
    lc_os1     = quick_lc(**common, oversample=1)[0]

    np.testing.assert_array_equal(lc_default, lc_os1)


def test_oversample_invalid_value():
    with pytest.raises(ValueError, match="oversample"):
        build_system(
            wavelength=np.array([550.0]), flux_quiet=np.array([1.0]),
            **dict(ld_coeffs=[0.4, 0.2]),
            times=_t(np.linspace(0, 360, 10)), P_rot=_P_ROT,
            stellar_grid_size=50, ve=2.0, oversample=0,
        )


def test_oversample_preserves_shape():
    wavelength = np.array([550.0])
    flux_quiet = np.array([1.0])
    flux_active = np.array([[0.7]])
    params = dict(ld_coeffs=[0.4, 0.2], inc_star=90.0)
    phases = np.linspace(0, 360, 50, endpoint=False)

    common = dict(
        wavelength=wavelength, flux_quiet=flux_quiet, flux_active=flux_active,
        **params,
        ar_lat=[20.0], ar_long=[5.0], ar_size=[11.0], ar_smoothness=[_SM],
        times=_t(phases), P_rot=_P_ROT, stellar_grid_size=80, ve=2.0,
    )

    for os_factor in [1, 3, 5]:
        result = quick_lc(**common, oversample=os_factor)
        assert result[0].shape == (50,)
        assert result[1].shape[0] == 50


# ===================================================================
# Numerical edge cases
# ===================================================================

class TestNumericalEdgeCases:

    def test_ar_at_pole(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        for lat in [90.0, -90.0]:
            result = quick_lc(
                wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
                **base_params,
                ar_lat=[lat], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
                times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
            )
            assert np.all(np.isfinite(result[0]))

    def test_ar_size_zero(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        baseline = _quiet_baseline(wl, flux_quiet, base_params, _t([0.0]))
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[0.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        np.testing.assert_allclose(
            result[0][0], baseline[0], atol=1e-3,
            err_msg="Zero-size AR should have negligible effect on flux",
        )

    def test_ar_size_90_degrees(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        baseline = _quiet_baseline(wl, flux_quiet, base_params, _t([0.0]))
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[90.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        assert np.all(np.isfinite(result[0]))
        assert float(result[0][0, 0]) < 0.95 * float(baseline[0, 0])

    def test_inclination_zero_pole_on(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(ld_coeffs=[0.3, 0.1], inc_star=0.0)
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t([0.0, 90.0, 180.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        assert np.all(np.isfinite(result[0]))

    def test_inclination_zero_constant_lc(self, flat_spectra):
        wl, flux_quiet, flux_active = flat_spectra
        params = dict(ld_coeffs=[0.3, 0.1], inc_star=0.0)
        phases = np.linspace(0, 360, 12, endpoint=False)
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t(phases), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        lc = result[0]
        np.testing.assert_allclose(
            lc, np.mean(lc), rtol=5e-3,
            err_msg="Pole-on view should produce a constant light curve",
        )

    def test_single_wavelength(self, base_params):
        wl = np.array([550.0])
        flux_quiet = np.array([1.0])
        flux_active = np.array([0.8])
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[10.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t([0.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        assert result[0].shape == (1,)
        assert np.all(np.isfinite(result[0]))


# ===================================================================
# Symmetry tests
# ===================================================================

class TestSymmetry:

    def test_equatorial_ar_symmetric_phases(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        result = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t([45.0, 315.0]), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )
        np.testing.assert_allclose(
            result[0][0], result[0][1], rtol=1e-4,
            err_msg="Equatorial AR should be symmetric about phase=0",
        )

    def test_north_south_symmetry_equator_on(self, flat_spectra, base_params):
        wl, flux_quiet, flux_active = flat_spectra
        phases = np.linspace(0, 360, 8, endpoint=False)

        result_north = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[30.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t(phases), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        result_south = quick_lc(
            wavelength=wl, flux_quiet=flux_quiet, flux_active=flux_active,
            **base_params,
            ar_lat=[-30.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=_t(phases), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
        )
        np.testing.assert_allclose(
            result_north[0], result_south[0], rtol=1e-4,
            err_msg="N/S symmetric ARs should produce identical LCs at inc=90°",
        )


# ===================================================================
# Orientation and sign of the rotational Doppler field
#
# The stellar spin axis is the body-frame y-axis (rotate_active_region
# applies rotation_matrix_y; make_lc places latitude on y). For a rigid
# rotator with sky-frame angular velocity omega = w (0, sin i, cos i), the
# line-of-sight component of v = omega x r is
#
#     v_z = -w * sin(i) * x
#
# so the radial velocity varies along the sky x axis, is independent of both
# y and the pixel's depth z, and vanishes pole-on.
#
# The tests below pin down the orientation and sign of that field, which the
# disc-integrated tests above cannot see: the stellar disc is circularly
# symmetric, so the rotationally broadened profile of the quiet star is
# identical whether the velocity is keyed on x or on y. The signature only
# appears in the resolved star map and in wavelength-resolved AR signals.
# ===================================================================

# A rotation fast enough that the limb shift dwarfs the line width, so a pixel
# on the approaching/receding limb moves the line clean out of the channel.
_DOPPLER_VE   = 300.0   # equatorial velocity [km/s] -> limb shift ~5 A at 5005 A
_DOPPLER_LAM0 = 5005.0  # line centre [A]
_DOPPLER_SIG  = 0.3     # line 1-sigma width [A]  << limb shift


def _line_spectrum(wl, depth=0.9, lam0=_DOPPLER_LAM0, sigma=_DOPPLER_SIG):
    """Flat continuum at 1.0 with a single narrow Gaussian absorption line."""
    return 1.0 - depth * np.exp(-0.5 * ((wl - lam0) / sigma) ** 2)


def _quiet_star_map(grid_size, wavelength_target, ve=_DOPPLER_VE, inc_star=90.0):
    """Star map of a bare quiet star (contrast identically 1) at one wavelength.

    ``flux_active = flux_quiet`` makes every active-region contrast exactly 1,
    so the AR arguments are inert and the map is the pure quiet photosphere;
    ``ld_coeffs=[0, 0]`` removes limb darkening so the only structure left in
    the map is the Doppler field itself.
    """
    wl   = np.linspace(4985.0, 5025.0, 801, dtype=np.float64)
    spec = _line_spectrum(wl)

    _, maps = quick_lc(
        wavelength=wl, flux_quiet=spec, flux_active=spec,
        ar_lat=[0.0], ar_long=[0.0], ar_size=[1.0], ar_smoothness=[1.0],
        times=np.array([0.0]), P_rot=1.0,
        stellar_grid_size=grid_size, ve=ve, ld_coeffs=[0.0, 0.0],
        inc_star=inc_star, plot_map_wavelength=wavelength_target,
    )
    # star_maps is (ntimes, n, n) reshaped from meshgrid(..., indexing='xy'):
    # axis 0 is the y (row) coordinate, axis 1 is x (column), both running
    # from -n//2 to +n//2, so index n//2 is the disc centre.
    return np.asarray(maps[0])


class TestDopplerFieldAxis:
    """The field varies along x, not along y."""

    def test_line_core_map_varies_along_x_and_is_flat_along_y(self):
        """Read the map at the rest wavelength of a narrow line.

        Pixels near x = +/-R are Doppler-shifted out of the line and stay
        bright; pixels along the y axis (x = 0) have zero radial velocity and
        must all sit at the same line-core depth as the disc centre.
        """
        G = 20
        m = _quiet_star_map(G, wavelength_target=_DOPPLER_LAM0)
        c = m.shape[0] // 2
        d = int(0.8 * G)

        core = m[c, c]
        # Line depth is 0.9, so shifting out of the core is a ~0.9 swing;
        # 0.05 is far above any interpolation/pixelation noise.
        assert m[c, c + d] > core + 0.05, (
            f"+x limb should be shifted out of the line core: "
            f"map[+0.8R, 0]={m[c, c + d]:.5f} vs centre {core:.5f}"
        )
        assert m[c, c - d] > core + 0.05, (
            f"-x limb should be shifted out of the line core: "
            f"map[-0.8R, 0]={m[c, c - d]:.5f} vs centre {core:.5f}"
        )

        # The x = 0 column is the zero-velocity meridian: no variation at all.
        assert abs(m[c + d, c] - core) < 1e-4, (
            f"x=0 column must have zero radial velocity, but map[0, +0.8R]="
            f"{m[c + d, c]:.5f} differs from centre {core:.5f}"
        )
        assert abs(m[c - d, c] - core) < 1e-4, (
            f"x=0 column must have zero radial velocity, but map[0, -0.8R]="
            f"{m[c - d, c]:.5f} differs from centre {core:.5f}"
        )

    def test_velocity_field_is_independent_of_y(self):
        """Every pixel sharing an x must share a velocity, at all y.

        Stronger than the axis check above: the full in-disc map at the line
        core must be constant down each column, since v_z depends on x alone.
        """
        G = 20
        m = _quiet_star_map(G, wavelength_target=_DOPPLER_LAM0)
        n = m.shape[0]
        c = n // 2

        coords = np.arange(n) - c
        xg, yg = np.meshgrid(coords, coords)
        in_disc = (xg ** 2 + yg ** 2) <= G ** 2

        for col in range(n):
            vals = m[in_disc[:, col], col]
            if vals.size < 2:
                continue
            assert np.ptp(vals) < 1e-4, (
                f"column x={coords[col]} spans {np.ptp(vals):.5f} in flux; "
                f"pixels at the same x must share one velocity"
            )


class TestDopplerFieldSign:
    """+x is the receding (redshifted) limb."""

    def test_features_rotate_toward_positive_x(self):
        """Rotation sense: a disc-centre feature moves toward +x with phase.

        This is the geometric half of the sign convention; the Doppler test
        below is the spectroscopic half. They must agree, or spots would be
        blueshifted while setting.
        """
        spr = 1.0
        centre = jnp.array([0.0, 0.0, spr])           # lat=0, long=0
        rotated = np.asarray(rotate_active_region(centre, 10.0, 90.0))
        assert rotated[0] > 0.0, (
            f"increasing phase must carry a disc-centre feature toward +x, "
            f"got x={rotated[0]:.5f}"
        )
        # ...and it is on its way out of view, not toward the observer.
        assert rotated[2] < spr

    def test_positive_x_is_redshifted(self):
        """Read the map redward of the line by the shift of the x = +0.5R
        column. The absorbed (dark) pixels must be the receding ones at
        x = +0.5R, not the approaching ones at x = -0.5R.
        """
        G = 20
        delta = _DOPPLER_LAM0 * (0.5 * _DOPPLER_VE / C_KMS)   # ~2.5 A
        m = _quiet_star_map(G, wavelength_target=_DOPPLER_LAM0 + delta)
        c = m.shape[0] // 2
        d = int(round(0.5 * G))

        assert m[c, c + d] < m[c, c - d] - 0.05, (
            f"at lambda0+{delta:.2f} A the dark pixels must sit on the "
            f"receding +x side: map[+0.5R]={m[c, c + d]:.5f}, "
            f"map[-0.5R]={m[c, c - d]:.5f}"
        )
        # The redshifted column should be near the line core.
        assert m[c, c + d] < 0.2


class TestEquatorialSpotLineAsymmetry:
    """A spot on the equator is carried across the disc in x, so its spectral
    imprint sweeps from blue to red. Keying the velocity on y instead pins an
    equatorial spot at zero velocity for its whole disc passage, making the
    blue and red channels bit-identical at every phase.

    The quiet photosphere is a featureless continuum here, so every wavelength
    structure in the light curve comes from the spot alone.
    """

    _G       = 12
    _NPHASE  = 21
    _PHI_MAX = 70.0    # degrees either side of meridian crossing (spot on-disc)

    def _blue_red_curves(self):
        wl   = np.linspace(4985.0, 5025.0, 801, dtype=np.float64)
        quiet  = np.ones_like(wl)          # no lines in the quiet photosphere
        active = _line_spectrum(wl)        # the spot carries the line

        # Phases symmetric about the spot's meridian crossing (long=0, phase=0).
        phases = np.linspace(-self._PHI_MAX, self._PHI_MAX, self._NPHASE)
        times  = phases / 360.0            # P_rot = 1 day

        lc, _ = quick_lc(
            wavelength=wl, flux_quiet=quiet, flux_active=active,
            ar_lat=[0.0], ar_long=[0.0], ar_size=[20.0], ar_smoothness=[20.0],
            times=times, P_rot=1.0,
            stellar_grid_size=self._G, ve=_DOPPLER_VE, ld_coeffs=[0.0, 0.0],
            inc_star=90.0,
        )
        lc = np.asarray(lc)                # (nphase, nwave)

        delta   = _DOPPLER_LAM0 * (0.5 * _DOPPLER_VE / C_KMS)
        i_blue  = int(np.argmin(np.abs(wl - (_DOPPLER_LAM0 - delta))))
        i_red   = int(np.argmin(np.abs(wl - (_DOPPLER_LAM0 + delta))))
        return lc[:, i_blue], lc[:, i_red]

    def test_blue_and_red_channels_differ(self):
        """The discriminator: with the velocity keyed on y an equatorial spot
        never acquires any radial velocity, so the two channels are identical.
        """
        blue, red = self._blue_red_curves()
        # The spot covers ~3% of the disc and carries a 0.9-deep line, so each
        # wing dips by ~0.03 at its own half of the passage while the other
        # wing sits at the continuum. 5e-3 is well below that and well above
        # the ~3e-6 far-wing leakage that is all a zero-velocity spot produces.
        assert np.max(np.abs(blue - red)) > 5e-3, (
            "an equatorial spot must imprint different signals on the blue and "
            f"red wings; max|blue-red|={np.max(np.abs(blue - red)):.3e}"
        )

    def test_blue_channel_is_the_time_reverse_of_the_red(self):
        """Sign check at the observable level: about the meridian crossing the
        spot is approaching before and receding after, so with a symmetric line
        profile and a phase grid symmetric about phase 0 the blue-wing curve is
        the red-wing curve run backwards.
        """
        blue, red = self._blue_red_curves()
        # Tolerance is set by float32 spectral interpolation, which breaks the
        # exact x -> -x symmetry of the pixel grid at the ~3e-3 level relative
        # to the dip depth.
        scale = max(np.ptp(blue), np.ptp(red))
        np.testing.assert_allclose(
            blue, red[::-1], atol=5e-3 * scale, rtol=0,
            err_msg="blue wing should mirror the red wing about phase 0",
        )

    def test_blue_leads_red(self):
        """And the ordering is fixed, not merely mirrored: the blue-wing
        absorption peaks while the spot is still approaching (negative phase).
        """
        blue, red = self._blue_red_curves()
        assert np.argmin(blue) < np.argmin(red), (
            f"blue-wing dip must precede the red-wing dip, got "
            f"argmin(blue)={np.argmin(blue)}, argmin(red)={np.argmin(red)}"
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

BASE_PARAMS = dict(ld_coeffs=[0.4, 0.2], inc_star=90.0)

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
    return build_system(
        wavelength        = WAVELENGTH,
        flux_quiet        = FLUX_QUIET,
        **BASE_PARAMS,
        times             = TIMES,
        P_rot             = P_ROT,
        **TRANSIT_PARAMS,
        stellar_grid_size = STELLAR_GRID,
        ve                = VE,
        ld_mode          = "quadratic",
        oversample        = 1,
    )


@pytest.fixture(scope="module")
def stellar_only_model():
    return build_system(
        wavelength        = WAVELENGTH,
        flux_quiet        = FLUX_QUIET,
        **BASE_PARAMS,
        times             = TIMES,
        P_rot             = P_ROT,
        stellar_grid_size = STELLAR_GRID,
        ve                = VE,
        ld_mode          = "quadratic",
        oversample        = 1,
    )


def _combined_lc(transit_overrides=None, ar_lat=None, ar_long=None, ar_size=None,
                 ar_smoothness=None, flux_active=None, oversample=1, params=None):
    """Thin wrapper that fills in defaults to keep test bodies short."""
    tp = {**TRANSIT_PARAMS, **(transit_overrides or {})}
    return quick_lc(
        wavelength        = WAVELENGTH,
        flux_quiet        = FLUX_QUIET,
        flux_active       = np.atleast_2d(flux_active if flux_active is not None else FLUX_QUIET),
        **(params or BASE_PARAMS),
        ar_lat            = ar_lat  or [0.0],
        ar_long           = ar_long or [180.0],  # far side — invisible by default
        ar_size           = ar_size or [0.001],
        ar_smoothness     = ar_smoothness or [_SM],
        times             = TIMES,
        P_rot             = P_ROT,
        **tp,
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
# 1b.  _compute_all_planets_mask — multi-planet combination
# ===================================================================

class TestComputeAllPlanetsMask:
    """
    Unit tests for combining several planets' occultation masks. Planets
    are opaque occulters (unlike active regions, whose contrasts sum -- see
    core.py's module docstring), so the combination rule here is
    multiplicative: 1 - prod(1 - mask_i), not 1 - sum(mask_i).
    """

    @pytest.fixture(autouse=True)
    def _grid(self):
        g = build_stellar_grid(50, 0.0)
        self.x   = jnp.asarray(g["x"])
        self.y   = jnp.asarray(g["y"])
        self.spr = g["star_pixel_rad"]

    def _combined(self, planet_xyz, k, softness=0.0):
        return _compute_all_planets_mask(
            self.x, self.y, self.spr, jnp.asarray(planet_xyz), jnp.asarray(k), softness,
        )

    def test_shape_matches_disc(self):
        mask = self._combined([[0.0, 0.0, 5.0]], [0.1])
        assert mask.shape == self.x.shape

    def test_nplanet_one_matches_single_planet_mask(self):
        combined = self._combined([[0.3, 0.1, 5.0]], [0.12])
        single   = _compute_planet_mask(self.x, self.y, self.spr, 0.3, 0.1, 5.0, 0.12)
        np.testing.assert_allclose(np.array(combined), np.array(single))

    def test_disjoint_planets_union(self):
        """Two well-separated planets: combined mask equals the sum (union,
        since disjoint) of the two individual hard masks -- up to a handful
        of exact-boundary pixels (d2 == k**2 to float32 precision), where
        vmap's batched execution and a standalone scalar call can round the
        strict '<' comparison differently by 1 ULP; this is a well-known
        float non-determinism at a hard threshold, not a correctness bug."""
        combined = self._combined(
            [[-0.5, 0.0, 5.0], [0.5, 0.0, 5.0]], [0.1, 0.1],
        )
        m1 = _compute_planet_mask(self.x, self.y, self.spr, -0.5, 0.0, 5.0, 0.1)
        m2 = _compute_planet_mask(self.x, self.y, self.spr, 0.5, 0.0, 5.0, 0.1)
        n_mismatch = int(jnp.sum(jnp.abs(combined - (m1 + m2)) > 0.5))
        assert n_mismatch <= 8, f"{n_mismatch} pixels disagree, expected only boundary ties"

    def test_overlapping_planets_do_not_exceed_full_occultation(self):
        """Two heavily overlapping high-k planets: a naive additive rule
        (mask1 + mask2) would exceed 1 and drive flux negative -- the
        multiplicative combination must stay within [0, 1]."""
        combined = self._combined(
            [[0.0, 0.0, 5.0], [0.05, 0.0, 5.0]], [0.4, 0.4],
        )
        assert bool(jnp.all(combined <= 1.0))
        assert bool(jnp.all(combined >= 0.0))

    def test_overlap_matches_multiplicative_formula(self):
        planet_xyz = [[0.0, 0.0, 5.0], [0.15, 0.0, 5.0]]
        k = [0.3, 0.25]
        combined = self._combined(planet_xyz, k)
        m1 = _compute_planet_mask(self.x, self.y, self.spr, 0.0, 0.0, 5.0, 0.3)
        m2 = _compute_planet_mask(self.x, self.y, self.spr, 0.15, 0.0, 5.0, 0.25)
        expected = 1.0 - (1.0 - m1) * (1.0 - m2)
        np.testing.assert_allclose(np.array(combined), np.array(expected), atol=1e-6)

    def test_disabled_planet_is_a_no_op(self):
        """A k=0 planet mixed in with a real one: combined mask equals the
        real planet's mask alone (the k>0 gate composes correctly), modulo
        the same handful of exact-boundary float ties described in
        test_disjoint_planets_union."""
        combined = self._combined(
            [[0.0, 0.0, 5.0], [0.9, 0.9, 5.0]], [0.2, 0.0],
        )
        single = _compute_planet_mask(self.x, self.y, self.spr, 0.0, 0.0, 5.0, 0.2)
        n_mismatch = int(jnp.sum(jnp.abs(combined - single) > 0.5))
        assert n_mismatch <= 8, f"{n_mismatch} pixels disagree, expected only boundary ties"


# ===================================================================
# 2.  build_system (transit-attached) — model dict structure
# ===================================================================

class TestBuildCombinedModel:

    def test_has_transit_flag_set(self, combined_model):
        assert combined_model.get("has_transit") is True

    def test_planet_xyz_key_present(self, combined_model):
        assert "planet_xyz" in combined_model

    def test_k_value_stored(self, combined_model):
        assert "k" in combined_model
        # k is always stored as an array of shape (nplanet, nwave) now (even
        # for a scalar input) so it can also hold a per-planet and/or
        # per-wavelength value.
        assert float(combined_model["k"][0, 0]) == pytest.approx(TRANSIT_PARAMS["k"])

    def test_nplanet_key_present_and_correct(self, combined_model):
        assert combined_model["nplanet"] == 1

    def test_planet_xyz_shape(self, combined_model):
        xyz    = combined_model["planet_xyz"]
        nphase = combined_model["nphase"]
        assert xyz.shape == (nphase, 1, 3)

    def test_xyz_third_column_Z(self, combined_model):
        xyz = np.array(combined_model["planet_xyz"])
        assert np.any(xyz[:, :, 2] > 0)

    def test_all_stellar_keys_preserved(self, combined_model):
        required = [
            "x_disc", "y_disc", "mu_disc", "col_idx", "vel_col",
            "star_pixel_rad", "total_pixels", "wavelength",
            "phases_rot", "ld_coeffs", "flat_indices", "n",
        ]
        for key in required:
            assert key in combined_model, f"Stellar key missing: '{key}'"

    def test_oversample_inflates_nphase(self):
        oversample = 3
        model = build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=TIMES, P_rot=P_ROT, **TRANSIT_PARAMS,
            stellar_grid_size=STELLAR_GRID, ve=VE, oversample=oversample,
        )
        n_orig    = model["nphase_original"]
        n_compute = model["nphase"]
        assert n_compute == n_orig * oversample

    def test_planet_xyz_length_matches_nphase(self):
        oversample = 3
        model = build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=TIMES, P_rot=P_ROT, **TRANSIT_PARAMS,
            stellar_grid_size=STELLAR_GRID, ve=VE, oversample=oversample,
        )
        assert model["planet_xyz"].shape[0] == model["nphase"]


# ===================================================================
# 3.  Transit physics
# ===================================================================

class TestTransitPhysics:

    def test_output_shape_matches_times(self):
        result = _combined_lc()
        assert result[0].shape == (len(TIMES),)

    def test_lc_finite(self):
        assert np.all(np.isfinite(_combined_lc()[0]))

    def test_transit_produces_flux_dip(self, stellar_only_model):
        lc = _combined_lc()[0]
        baseline = float(np.min(np.array(make_lc(stellar_only_model)[0])))
        assert float(np.min(lc)) < baseline

    def test_transit_depth_scales_with_k(self):
        d_small = 1.0 - float(np.min(_combined_lc({**TRANSIT_PARAMS, "k": 0.05})[0]))
        d_large = 1.0 - float(np.min(_combined_lc({**TRANSIT_PARAMS, "k": 0.15})[0]))
        assert d_large > d_small

    def test_approximate_transit_depth_equals_k_squared(self):
        k = 0.1
        tp = {**TRANSIT_PARAMS, "k": k}
        params_no_ld = dict(ld_coeffs=[0.0, 0.0], inc_star=90.0)
        lc = _combined_lc(tp, params=params_no_ld)[0]
        depth = 1.0 - float(np.min(lc))
        np.testing.assert_allclose(depth, k**2, rtol=0.15)

    def test_grazing_transit_shallower_than_central(self):
        k   = 0.1
        a   = TRANSIT_PARAMS["a_over_rstar"]
        inc_grazing = np.arccos(0.85 / a)

        d_central = 1.0 - float(np.min(_combined_lc({**TRANSIT_PARAMS, "k": k})[0]))
        d_grazing = 1.0 - float(np.min(_combined_lc(
            {**TRANSIT_PARAMS, "k": k, "inclination": inc_grazing})[0]))
        assert d_central > d_grazing

    def test_spot_crossing_produces_positive_bump(self):
        lc_spot = _combined_lc(
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], flux_active=FLUX_SPOT,
        )[0]
        lc_clean = _combined_lc(
            ar_lat=[0.0], ar_long=[180.0], ar_size=[10.0], flux_active=FLUX_SPOT,
        )[0]

        oot = np.abs(TIMES) > 0.12
        lc_spot_norm = lc_spot / np.median(lc_spot[oot])
        lc_clean_norm = lc_clean / np.median(lc_clean[oot])

        in_transit = np.abs(TIMES) < 0.04
        bump = float(np.min(lc_spot_norm[in_transit])) - float(np.min(lc_clean_norm[in_transit]))
        assert bump > 0

    def test_facula_crossing_produces_negative_anomaly(self):
        lc_fac = _combined_lc(
            ar_lat=[0.0], ar_long=[0.0], ar_size=[10.0], flux_active=FLUX_FACULA,
        )[0]
        lc_clean = _combined_lc(
            ar_lat=[0.0], ar_long=[180.0], ar_size=[10.0], flux_active=FLUX_FACULA,
        )[0]

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
        result_stellar = make_lc(
            stellar_only_model,
            flux_active=jnp.array(FLUX_QUIET),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([180.0]),
            ar_size=jnp.array([0.001]), ar_smoothness=jnp.array([_SM]),
        )
        oot = np.abs(TIMES) > 0.12
        np.testing.assert_allclose(
            result_combined[0][oot], np.array(result_stellar[0])[oot], rtol=1e-4,
        )

    def test_eccentric_orbit_centre_Z_positive(self, stellar_only_model):
        baseline = float(np.min(np.array(make_lc(stellar_only_model)[0])))
        for ecc, omega in [(0.3, np.pi / 2.0), (0.5, np.pi / 4.0)]:
            tp = {**TRANSIT_PARAMS, "ecc": ecc, "omega_peri": omega}
            lc = _combined_lc(tp)[0]
            assert float(np.min(lc)) < baseline

    def test_no_transit_when_fully_inclined(self, stellar_only_model):
        a = TRANSIT_PARAMS["a_over_rstar"]
        k = TRANSIT_PARAMS["k"]
        inc_no_transit = np.arccos(2.0 * (1.0 + k) / a)
        tp = {**TRANSIT_PARAMS, "inclination": inc_no_transit}
        lc = _combined_lc(tp)[0]
        baseline = np.array(make_lc(stellar_only_model)[0])
        np.testing.assert_allclose(lc, baseline, atol=0.005)


class TestObliquity:
    """
    Integration tests for the ``sp_orb`` transit parameter through
    build_system/make_lc/quick_lc. ``sp_orb`` is given in degrees here
    (converted to radians internally, before reaching sajax.planet, which
    takes radians). The stellar spin axis is fixed along sky-Y (see
    geometry.py / core.py's ``vel_col``), so an aligned (sp_orb=0) transit
    chord runs along the projected stellar equator, while a polar
    (sp_orb=90) chord runs along the spin axis instead -- see the
    module-level ``TRANSIT_PARAMS`` for the shared edge-on, b~0 orbit
    these tests build on.
    """

    # A spot sitting at normalized sky-X ~ 0.3 at t=0 (see ar_long below)
    # sits squarely on the aligned (sp_orb=0) chord, which sweeps X
    # through that value, but off the polar (sp_orb=90) chord, which
    # stays pinned near X~0 for this b~0 orbit (see TestObliquity in
    # test_planet.py's test_polar_obliquity_swaps_null_axis_at_central_transit).
    _AR_LONG = float(np.rad2deg(np.arcsin(0.3)))

    def test_default_obliquity_matches_omitted_argument(self):
        """Omitting spin-orbit angle must give the identical light curve to sp_orb=0.0."""
        lc_omitted = _combined_lc()[0]
        lc_explicit = _combined_lc({**TRANSIT_PARAMS, "sp_orb": 0.0})[0]
        np.testing.assert_allclose(lc_omitted, lc_explicit, rtol=1e-6)

    def test_zero_obliquity_matches_pre_obliquity_transit(self, stellar_only_model):
        """sp_orb=0 must reproduce the plain (pre-spin-orbit angle) transit depth."""
        lc = _combined_lc({**TRANSIT_PARAMS, "sp_orb": 0.0})[0]
        baseline = float(np.min(np.array(make_lc(stellar_only_model)[0])))
        depth_no_obliquity = 1.0 - float(np.min(_combined_lc()[0]))
        depth_zero_obliquity = 1.0 - float(np.min(lc))
        np.testing.assert_allclose(depth_zero_obliquity, depth_no_obliquity, rtol=1e-6)
        assert float(np.min(lc)) < baseline

    def test_polar_obliquity_suppresses_spot_crossing_bump(self):
        """
        A spot placed on the aligned transit chord produces a pronounced
        spot-crossing bump at sp_orb=0 (planet occults it), but the
        bump should nearly vanish at sp_orb=90 (polar chord misses
        the spot's latitude band for this b~0 orbit) -- the same physical
        effect discussed for stellar sp_orb: the transit chord, not the
        spot itself, determines whether a crossing happens.
        """
        def bump(sp_orb):
            tp = {**TRANSIT_PARAMS, "sp_orb": sp_orb}
            lc_spot = _combined_lc(
                tp, ar_lat=[0.0], ar_long=[self._AR_LONG], ar_size=[8.0],
                flux_active=FLUX_SPOT,
            )[0]
            lc_clean = _combined_lc(
                tp, ar_lat=[0.0], ar_long=[180.0], ar_size=[8.0],
                flux_active=FLUX_SPOT,
            )[0]
            oot = np.abs(TIMES) > 0.12
            lc_spot_norm  = lc_spot  / np.median(lc_spot[oot])
            lc_clean_norm = lc_clean / np.median(lc_clean[oot])
            in_transit = np.abs(TIMES) < 0.05
            return float(np.max(np.abs(lc_spot_norm[in_transit] - lc_clean_norm[in_transit])))

        bump_aligned = bump(0.0)
        bump_polar   = bump(90.0)

        assert bump_aligned > 5e-4, (
            f"Aligned spot-crossing bump should be clearly resolved, got {bump_aligned:.2e}"
        )
        assert bump_polar < 0.1 * bump_aligned, (
            f"Polar transit should largely miss the spot: bump_polar={bump_polar:.2e} "
            f"vs bump_aligned={bump_aligned:.2e}"
        )

    def test_obliquity_leaves_out_of_transit_flux_unchanged(self):
        """Spin-orbit angle only rotates the planet's trajectory, so it must not
        affect the (planet-independent) out-of-transit stellar flux."""
        lc_aligned = _combined_lc(
            {**TRANSIT_PARAMS, "sp_orb": 0.0},
            ar_lat=[0.0], ar_long=[self._AR_LONG], ar_size=[8.0], flux_active=FLUX_SPOT,
        )[0]
        lc_polar = _combined_lc(
            {**TRANSIT_PARAMS, "sp_orb": 90.0},
            ar_lat=[0.0], ar_long=[self._AR_LONG], ar_size=[8.0], flux_active=FLUX_SPOT,
        )[0]
        oot = np.abs(TIMES) > 0.12
        np.testing.assert_allclose(lc_aligned[oot], lc_polar[oot], rtol=1e-5)


# ===================================================================
# 4.  Oversampling — transit path
# ===================================================================

class TestTransitOversampling:

    def test_oversample_preserves_output_shape(self):
        for os in [1, 3, 5]:
            lc = _combined_lc(oversample=os)[0]
            assert lc.shape == (len(TIMES),)

    def test_oversampled_lc_is_finite(self):
        lc = _combined_lc(oversample=3)[0]
        assert np.all(np.isfinite(lc))

    def test_oversampled_transit_still_present(self, stellar_only_model):
        lc = _combined_lc(oversample=3)[0]
        baseline = float(np.min(np.array(make_lc(stellar_only_model)[0])))
        assert float(np.min(lc)) < baseline


# ===================================================================
# 4b.  Multiple planets (trailing nplanet axis, mirroring TestMultiAR)
#
# Planets combine *multiplicatively* (opaque occulters), the opposite of
# TestAROverlapCompounding's *additive* AR-contrast rule -- named/documented
# explicitly to avoid confusion, since both "multiple X" cases sit close
# together conceptually.
# ===================================================================

class TestMultiPlanet:

    def test_multi_planet_shapes(self):
        """Two planets should work and preserve the usual (ntimes,) output
        shape -- nplanet is fully consumed before the pixel sum."""
        lc, star_maps = _combined_lc(
            transit_overrides=dict(
                t0=[0.0, 0.09], period=[5.0, 30.0],
                a_over_rstar=[10.0, 10.0], inclination=[np.pi / 2.0, np.pi / 2.0],
                k=[0.1, 0.05],
            )
        )
        assert lc.shape == (len(TIMES),)
        assert star_maps.shape == (len(TIMES), STELLAR_GRID * 2 + 1, STELLAR_GRID * 2 + 1)
        assert np.all(np.isfinite(lc))

    def test_nplanet_one_matches_pre_existing_single_planet_api(self):
        """Regression pin: scalar TRANSIT_PARAMS (today's exact single-planet
        API) must give bit-identical output to before this feature existed."""
        lc_scalar = _combined_lc()[0]
        lc_single_list = _combined_lc(transit_overrides=dict(
            t0=[TRANSIT_PARAMS["t0"]], period=[TRANSIT_PARAMS["period"]],
            a_over_rstar=[TRANSIT_PARAMS["a_over_rstar"]],
            inclination=[TRANSIT_PARAMS["inclination"]], k=[TRANSIT_PARAMS["k"]],
        ))[0]
        np.testing.assert_allclose(np.array(lc_scalar), np.array(lc_single_list))

    def test_per_planet_k_gives_per_planet_depth(self):
        """A larger k for one planet gives a deeper dip at its own transit
        window than the other, well-separated planet's shallower one."""
        lc, _ = _combined_lc(
            transit_overrides=dict(
                t0=[0.0, 0.09], period=[5.0, 30.0],
                a_over_rstar=[10.0, 10.0], inclination=[np.pi / 2.0, np.pi / 2.0],
                k=[0.15, 0.05],
            )
        )
        lc = np.array(lc)
        baseline = float(np.max(lc))
        idx_a = int(np.argmin(np.abs(TIMES - 0.0)))
        idx_b = int(np.argmin(np.abs(TIMES - 0.09)))
        depth_a = baseline - lc[idx_a]
        depth_b = baseline - lc[idx_b]
        assert depth_a > depth_b > 0

    def test_two_non_overlapping_transits_independent(self):
        """Two planets with well-separated transit windows (large a_over_rstar
        keeps each transit brief relative to the narrow TIMES baseline)
        produce dips matching each planet's own isolated single-planet
        computation."""
        shared = dict(a_over_rstar=40.0, inclination=np.pi / 2.0, k=0.1)
        lc_a = np.array(_combined_lc(transit_overrides=dict(t0=0.0, period=5.0, **shared))[0])
        lc_b = np.array(_combined_lc(transit_overrides=dict(t0=0.1, period=8.0, **shared))[0])
        lc_both = np.array(_combined_lc(transit_overrides=dict(
            t0=[0.0, 0.1], period=[5.0, 8.0],
            a_over_rstar=[40.0, 40.0], inclination=[np.pi / 2.0, np.pi / 2.0], k=[0.1, 0.1],
        ))[0])
        baseline = float(np.max(lc_a))
        idx_a = int(np.argmin(np.abs(TIMES - 0.0)))
        idx_b = int(np.argmin(np.abs(TIMES - 0.1)))
        np.testing.assert_allclose(lc_both[idx_a], lc_a[idx_a], atol=1e-5)
        np.testing.assert_allclose(lc_both[idx_b], lc_b[idx_b], atol=1e-5)
        assert lc_both[idx_a] < baseline and lc_both[idx_b] < baseline

    def test_overlapping_transits_do_not_over_subtract(self):
        """Two planets transiting at the same time with large k: flux must
        stay non-negative -- a naive additive combination could go negative
        (1 - k1^2 - k2^2 < 0 for large k's), the multiplicative rule can't."""
        lc, _ = _combined_lc(
            transit_overrides=dict(
                t0=[0.0, 0.0], period=[5.0, 5.0],
                a_over_rstar=[10.0, 10.0], inclination=[np.pi / 2.0, np.pi / 2.0],
                k=[0.4, 0.4],
            )
        )
        assert np.all(np.array(lc) >= 0.0)


# ===================================================================
# 5.  API consistency
# ===================================================================

class TestAPIConsistency:

    def test_stellar_only_model_lacks_transit_flag(self, stellar_only_model):
        assert not stellar_only_model.get("has_transit", False)

    def test_evaluate_stellar_only_still_works(self, stellar_only_model):
        result = make_lc(
            stellar_only_model,
            flux_active=jnp.array(FLUX_SPOT),
            ar_lat=jnp.array([20.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )
        lc = np.array(result[0])
        assert np.all(np.isfinite(lc))
        assert lc.shape == (len(TIMES),)

    def test_quick_lc_api_unchanged(self):
        from sajax import quick_lc
        result = quick_lc(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, flux_active=FLUX_SPOT,
            **BASE_PARAMS,
            ar_lat=[20.0], ar_long=[0.0], ar_size=[10.0], ar_smoothness=[_SM],
            times=TIMES, P_rot=P_ROT,
            stellar_grid_size=STELLAR_GRID, ve=VE,
        )
        assert result[0].shape == (len(TIMES),)
        assert np.all(np.isfinite(result[0]))

    def test_no_transit_flag_gives_unity_transit_factor(self, stellar_only_model):
        result_stellar = make_lc(
            stellar_only_model,
            flux_active=jnp.array(FLUX_QUIET),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([180.0]),
            ar_size=jnp.array([0.001]), ar_smoothness=jnp.array([_SM]),
        )
        lc = np.array(result_stellar[0])
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
        return build_system(
            wavelength=np.array([550.0]),
            flux_quiet=np.array([1.0]),
            **dict(ld_coeffs=[0.4, 0.2], inc_star=90.0),
            times=_t(np.array([0.0])), P_rot=_P_ROT,
            stellar_grid_size=50,
            ve=0.0,
        )

    @staticmethod
    def _lc_scalar(model, flux_active, ar_lat, ar_long, ar_size, ar_smoothness):
        return jnp.sum(make_lc(
            model, flux_active=flux_active, ar_lat=ar_lat, ar_long=ar_long,
            ar_size=ar_size, ar_smoothness=ar_smoothness,
        )[0])

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
    At ar_smoothness=40 the boundary transition is narrow enough that a
    finite-difference step of h=0.1 rad overshoots it, so the FD chord no
    longer approximates the local derivative (JAX's infinitesimal-limit
    gradient is correct; the FD estimate is not). h=1e-4 rad stays well
    inside the boundary's locally-linear regime.
    """

    _spr        = 50.0
    _arsize_deg = 20.0
    _smoothness = 40.0

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
        Place the AR centre on the sphere at spz = cos(arsize) * spr,
        spy = sin(arsize) * spr (spx = 0) so that the disc-centre pixel sits
        exactly on the AR boundary. Since _compute_ar_shape computes the
        chord length directly, the AR-centre vector must actually lie on the sphere.
        Move spz along the sphere (spy = sqrt(spr^2 - spz^2)
        keeps it on-sphere) toward the observer: cos_theta at the disc-centre
        pixel rises -> shape increases -> d(shape)/d(spz) > 0.
        """
        cos_a = float(jnp.cos(jnp.float32(self._arsize_rad)))
        spz0  = jnp.float32(cos_a * self._spr)

        def _shape_at_spz(sz):
            sy = jnp.sqrt(self._spr ** 2 - sz ** 2)
            return self._shape_sum(
                jnp.float32(self._arsize_rad),
                spx=0.0, spy=sy, spz=sz, px=0.0, py=0.0,
            )

        grad = float(jax.grad(_shape_at_spz)(spz0))
        assert grad > 0

    def test_dark_spot_flux_decreases_as_spot_grows(self, flat_spectra, base_params):
        """
        For a dark spot (flux_active < 1.0) centred on the disc, enlarging
        the spot must reduce the total normalised broadband flux.
        """
        wl, flux_quiet, flux_active = flat_spectra
        nwave = len(wl)
        assert float(flux_active[0]) < float(flux_quiet[0])

        model = build_system(
            wavelength=wl, flux_quiet=flux_quiet, **base_params,
            times=_t(np.array([0.0])), P_rot=_P_ROT, stellar_grid_size=int(self._spr),
            ve=0.0, ld_mode="quadratic",
        )
        spr = float(model["star_pixel_rad"])
        ar_cart    = jnp.array([[0.0, 0.0, spr]], dtype=jnp.float32)
        planet_xyz = jnp.array([[0.0, 0.0, -1e10]], dtype=jnp.float32)  # (nplanet=1, 3)
        flux_act   = jnp.array(flux_active[np.newaxis, :], dtype=jnp.float32)  # (1, nwave)
        n_coeffs   = 2
        n_mu       = model["mu_profile_pts"].shape[0]

        def flux_fn(arsize):
            fval, _ = _compute_single_phase(
                ar_cart, planet_xyz,
                wavelength          = model["wavelength"],
                flux_quiet          = model["flux_quiet"],
                flux_active         = flux_act,
                ld_coeffs_quiet    = model["ld_coeffs"],
                ld_coeffs_active   = jnp.broadcast_to(model["ld_coeffs"][None, :, :], (1, nwave, n_coeffs)),
                I_profile_quiet     = model["I_profile"],
                I_profile_active    = jnp.broadcast_to(model["I_profile"][None, :, :], (1, nwave, n_mu)),
                mu_profile_pts      = model["mu_profile_pts"],
                x_disc              = model["x_disc"],
                y_disc              = model["y_disc"],
                mu_disc             = model["mu_disc"],
                col_idx             = model["col_idx"],
                vel_col             = model["vel_col"],
                star_pixel_rad      = spr,
                total_pixels        = model["total_pixels"],
                arsize_rads         = jnp.array([arsize]),
                ar_smoothness       = jnp.array([_SM]),
                k                   = jnp.zeros((1, nwave)),  # k is (nplanet, nwave) now
                ld_mode            = model["ld_mode"],
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
# full make_lc pipeline
# ===================================================================

class TestARParamGradientsFD:
    """
    Compare JAX autodiff to central finite differences for active-region
    latitude, longitude, and flux contrast, using the public
    ``make_lc`` API. Step sizes below were empirically
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
        return build_system(
            wavelength   = np.array([550.0]),
            flux_quiet   = np.array([1.0]),
            **dict(ld_coeffs=[0.4, 0.2], inc_star=90.0),
            times=_t(np.array([0.0])), P_rot=_P_ROT,
            stellar_grid_size = self._SPR,
            ve           = 0.0,
        )

    def _lc_sum(self, model, lat, long, flux, arsize=None, smoothness=None):
        arsize = jnp.float32(self._ARSIZE) if arsize is None else arsize
        smoothness = jnp.float32(_SM) if smoothness is None else smoothness
        return jnp.sum(make_lc(
            model,
            flux_active   = jnp.array([flux]),
            ar_lat        = jnp.array([lat]),
            ar_long       = jnp.array([long]),
            ar_size       = jnp.array([arsize]),
            ar_smoothness = jnp.array([smoothness]),
        )[0])

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
    parameter feeds into (e.g. ld_coeffs_quiet for u1/u2, vel_col for ve).
    """

    _SPR    = 50
    _ARSIZE = 20.0    # degrees
    _H_INC  = 0.1     # degrees
    _H_LDC  = 0.01
    _H_PROT = 0.01    # days
    _C      = 299_792.458  # km/s

    @pytest.fixture(scope="class")
    def grad_model(self):
        return build_system(
            wavelength=np.array([550.0]),
            flux_quiet=np.array([1.0]),
            **dict(ld_coeffs=[0.4, 0.2], inc_star=90.0),
            times=_t(np.array([0.0])), P_rot=_P_ROT,
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
                           vel_col=None, ld_coeffs_quiet=None):
        nwave    = 1
        n_coeffs = 2
        n_mu     = model["mu_profile_pts"].shape[0]
        ldc_q    = ld_coeffs_quiet if ld_coeffs_quiet is not None else model["ld_coeffs"]
        flux_norm, _ = _compute_single_phase(
            ar_cart_rotated,
            jnp.array([[0.0, 0.0, -1e10]]),  # (nplanet=1, 3)
            wavelength          = model["wavelength"],
            flux_quiet          = model["flux_quiet"],
            flux_active         = jnp.array([[0.7]], dtype=jnp.float32),
            ld_coeffs_quiet    = ldc_q,
            ld_coeffs_active   = jnp.broadcast_to(model["ld_coeffs"][None, :, :], (1, nwave, n_coeffs)),
            I_profile_quiet     = model["I_profile"],
            I_profile_active    = jnp.broadcast_to(model["I_profile"][None, :, :], (1, nwave, n_mu)),
            mu_profile_pts      = model["mu_profile_pts"],
            x_disc              = model["x_disc"],
            y_disc              = model["y_disc"],
            mu_disc             = model["mu_disc"],
            col_idx             = model["col_idx"],
            vel_col             = vel_col if vel_col is not None else model["vel_col"],
            star_pixel_rad      = model["star_pixel_rad"],
            total_pixels        = model["total_pixels"],
            arsize_rads         = jnp.array([jnp.deg2rad(jnp.float32(self._ARSIZE))]),
            ar_smoothness       = jnp.array([_SM]),
            k                   = jnp.zeros((1, nwave)),  # k is (nplanet, nwave) now
            ld_mode            = model["ld_mode"],
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
                ld_coeffs_quiet=jnp.array([[u1, jnp.float32(0.2)]]),
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
                ld_coeffs_quiet=jnp.array([[u1, jnp.float32(0.2)]]),
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
                ld_coeffs_quiet=jnp.array([[jnp.float32(0.4), u2]]),
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
            vel_col = coords / spr * (ve / self._C)
            return self._call_single_phase(grad_model, rotated, vel_col=vel_col)

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
        model = build_system(
            wavelength=wl, flux_quiet=flux_quiet_line,
            **dict(ld_coeffs=[0.4, 0.2], inc_star=90.0),
            times=_t(np.array([0.0])), P_rot=_P_ROT, stellar_grid_size=self._SPR, ve=0.0,
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
            vel_col = coords / spr * (ve / self._C)
            fval, _ = _compute_single_phase(
                rotated, jnp.array([[0.0, 0.0, -1e10]]),  # (nplanet=1, 3)
                wavelength=model["wavelength"], flux_quiet=model["flux_quiet"],
                flux_active=flux_active_wl,
                ld_coeffs_quiet=model["ld_coeffs"],
                ld_coeffs_active=jnp.broadcast_to(model["ld_coeffs"][None, :, :], (1, nwave, n_coeffs)),
                I_profile_quiet=model["I_profile"],
                I_profile_active=jnp.broadcast_to(model["I_profile"][None, :, :], (1, nwave, n_mu)),
                mu_profile_pts=model["mu_profile_pts"],
                x_disc=model["x_disc"], y_disc=model["y_disc"], mu_disc=model["mu_disc"],
                col_idx=model["col_idx"], vel_col=vel_col,
                star_pixel_rad=spr, total_pixels=model["total_pixels"],
                arsize_rads=jnp.array([jnp.deg2rad(jnp.float32(self._ARSIZE))]),
                ar_smoothness=jnp.array([_SM]),
                k=jnp.zeros((1, nwave)), ld_mode=model["ld_mode"],  # k is (nplanet, nwave) now
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


# ===================================================================
# 6.  Transit-parameter autodiff -- dynamic make_lc path
# ===================================================================

class TestTransitAutodiff:
    """
    Verify gradient flow through the combined (star + transit) pipeline for
    the orbital parameters, via make_lc's individual transit
    keyword arguments (t0, period, a_over_rstar, inclination, ecc,
    omega_peri, k -- added alongside ``transit_softness``).

    The occultation mask in ``_compute_planet_mask`` is a hard threshold by
    default: on the fixed pixel grid, occulted flux is a genuine staircase
    function of every parameter that moves the mask (k, a_over_rstar,
    inclination, t0, period, ecc, omega_peri), so its analytic derivative is
    exactly 0 almost everywhere -- this is not a bug, and is asserted below
    so a future change doesn't silently paper over it. ``transit_softness``
    is the opt-in escape hatch for gradient-based retrieval.
    """

    _SOFTNESS = 1.0 / (10.0 * STELLAR_GRID)  # matches TestPlanetGradients' convention

    @pytest.fixture(scope="class")
    def transit_model(self):
        return build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=TIMES, P_rot=P_ROT, **TRANSIT_PARAMS,
            stellar_grid_size=STELLAR_GRID, ve=VE, oversample=1,
        )

    def _lc_sum(self, model, transit_overrides, transit_softness=0.0):
        """transit_overrides overrides one or more of TRANSIT_PARAMS' keys;
        the full (required) set is always passed to make_lc."""
        tp = {**TRANSIT_PARAMS, **transit_overrides}
        result = make_lc(
            model,
            flux_active=jnp.array(FLUX_SPOT),
            ar_lat=jnp.array([5.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([8.0]), ar_smoothness=jnp.array([_SM]),
            t0=tp["t0"], period=tp["period"], a_over_rstar=tp["a_over_rstar"],
            inclination=tp["inclination"], ecc=tp.get("ecc", 0.0),
            omega_peri=tp.get("omega_peri", 0.0), k=tp["k"],
            transit_softness=transit_softness,
        )
        return jnp.sum(result[0])

    def test_hard_edge_grad_wrt_k_is_zero(self, transit_model):
        """Documents the known limitation: default (hard-edge) grad is 0."""
        g = jax.grad(
            lambda k: self._lc_sum(transit_model, {"k": k})
        )(jnp.float32(0.1))
        assert float(g) == 0.0

    @pytest.mark.parametrize("param, value", [
        ("k", 0.1),
        ("a_over_rstar", 15.0),
        ("inclination", np.pi / 2.0),
        ("period", 5.0),
    ])
    def test_soft_mask_grad_is_finite_and_nonzero(self, transit_model, param, value):
        g = jax.grad(
            lambda v: self._lc_sum(
                transit_model, {param: v}, transit_softness=self._SOFTNESS,
            )
        )(jnp.float32(value))
        assert jnp.isfinite(g)
        assert jnp.abs(g) > 0

    def test_soft_mask_disabled_by_default(self, transit_model):
        """transit_softness defaults to 0.0 -- identical to the hard edge."""
        lc_default = self._lc_sum(transit_model, {"k": 0.1})
        lc_explicit_hard = self._lc_sum(transit_model, {"k": 0.1}, transit_softness=0.0)
        assert float(lc_default) == float(lc_explicit_hard)


# ===================================================================
# 7.  make_lc API symmetry -- AR & transit parameter groups
# ===================================================================

class TestParameterGroupValidation:
    """
    Both the AR parameter group (flux_active/ar_lat/ar_long/ar_size/
    ar_smoothness) and the transit parameter group (t0/period/a_over_rstar/
    inclination/k) are all-or-nothing: give every one of them, or none.
    Giving some but not all is a user error and should raise ValueError
    rather than silently doing something unintended.
    """

    @pytest.fixture(scope="class")
    def transit_model(self):
        return build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=TIMES, P_rot=P_ROT, **TRANSIT_PARAMS,
            stellar_grid_size=STELLAR_GRID, ve=VE, oversample=1,
        )

    # ---- AR group -----------------------------------------------------

    def test_partial_ar_args_raises(self, stellar_only_model):
        with pytest.raises(ValueError, match="active-region"):
            make_lc(
                stellar_only_model,
                flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([0.0]),
                # ar_long, ar_size, ar_smoothness omitted
            )

    def test_no_ar_args_gives_quiet_star(self, stellar_only_model):
        result = make_lc(stellar_only_model)
        lc = np.array(result[0])
        assert np.all(np.isfinite(lc))
        # A quiet star's flux is constant at every phase/wavelength (no AR,
        # no transit -- nothing varies it).
        np.testing.assert_allclose(lc, np.broadcast_to(lc[0], lc.shape), rtol=1e-5)

    # ---- Transit group --------------------------------------------------

    def test_partial_transit_args_raises(self, transit_model):
        with pytest.raises(ValueError, match="transit"):
            make_lc(
                transit_model,
                flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.array([5.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([8.0]), ar_smoothness=jnp.array([_SM]),
                k=0.1,  # t0/period/a_over_rstar/inclination omitted
            )

    def test_transit_args_without_transit_model_raises(self, stellar_only_model):
        with pytest.raises(ValueError, match="no transit attached"):
            make_lc(
                stellar_only_model,
                flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.array([5.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([8.0]), ar_smoothness=jnp.array([_SM]),
                **TRANSIT_PARAMS,
            )

    def test_lone_ecc_without_required_raises(self, transit_model):
        """ecc/omega_peri alone (without the 5 required) is also an error."""
        with pytest.raises(ValueError, match="transit"):
            make_lc(
                transit_model,
                flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.array([5.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([8.0]), ar_smoothness=jnp.array([_SM]),
                ecc=0.1,
            )

    def test_no_transit_args_uses_static_model_transit(self, transit_model, stellar_only_model):
        """Omitting all transit kwargs falls back to the model's static transit."""
        result = make_lc(
            transit_model,
            flux_active=jnp.array(FLUX_QUIET),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([180.0]),
            ar_size=jnp.array([0.001]), ar_smoothness=jnp.array([_SM]),
        )
        lc = np.array(result[0])
        assert np.all(np.isfinite(lc))
        baseline = float(np.min(np.array(make_lc(stellar_only_model)[0])))
        assert float(np.min(lc)) < baseline  # the static transit still occults

    # ---- Multi-planet shape validation ----------------------------------

    def test_transit_param_nplanet_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_system(
                wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
                times=TIMES, P_rot=P_ROT,
                t0=[0.0, 2.0], period=[5.0, 11.0, 20.0],
                a_over_rstar=[10.0, 10.0], inclination=[np.pi / 2.0, np.pi / 2.0],
                k=[0.1, 0.1],
                stellar_grid_size=STELLAR_GRID, ve=VE,
            )

    def test_k_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="k shape mismatch"):
            build_system(
                wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
                times=TIMES, P_rot=P_ROT,
                t0=[0.0, 2.0], period=[5.0, 11.0],
                a_over_rstar=[10.0, 10.0], inclination=[np.pi / 2.0, np.pi / 2.0],
                k=[0.1, 0.1, 0.1],  # length 3, but nplanet=2 and nwave=1
                stellar_grid_size=STELLAR_GRID, ve=VE,
            )

    def test_k_nplanet_one_legacy_chromatic_still_works(self):
        """A bare (nwave,) k array for a single planet (pre-multi-planet
        convention) must still be accepted and treated as chromatic depth."""
        wl = np.array([500.0, 550.0, 600.0])
        flux_q = np.ones_like(wl)
        model = build_system(
            wavelength=wl, flux_quiet=flux_q, **BASE_PARAMS,
            times=TIMES, P_rot=P_ROT,
            t0=0.0, period=5.0, a_over_rstar=10.0, inclination=np.pi / 2.0,
            k=[0.1, 0.12, 0.09],
            stellar_grid_size=STELLAR_GRID, ve=VE,
        )
        assert model["nplanet"] == 1
        assert model["k"].shape == (1, 3)
        np.testing.assert_allclose(np.array(model["k"][0]), [0.1, 0.12, 0.09])


# ===================================================================
# 8.  Dynamic override of the quiet photosphere's own LDC coefficients
# ===================================================================

class TestQuietLdcOverride:
    """
    ``ld_coeffs_quiet`` lets make_lc override the quiet
    photosphere's own limb-darkening coefficients per call (JAX values/
    tracers included), mirroring how ld_coeffs_active already works for
    active regions -- the build-time ``ld_coeffs`` given to build_system
    was otherwise static for the model's whole lifetime.
    """

    @pytest.fixture(scope="class")
    def grad_model(self):
        return build_system(
            wavelength=np.array([550.0]), flux_quiet=np.array([1.0]),
            times=_t(np.array([0.0])), P_rot=_P_ROT, stellar_grid_size=50, ve=0.0,
            ld_coeffs=[0.4, 0.2], inc_star=90.0,
        )

    def test_default_matches_static_value(self, grad_model):
        kwargs = dict(
            flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([20.0]),
            ar_long=jnp.array([0.0]), ar_size=jnp.array([10.0]),
            ar_smoothness=jnp.array([_SM]),
        )
        lc_default = make_lc(grad_model, **kwargs)[0]
        lc_explicit = make_lc(
            grad_model, **kwargs, ld_coeffs_quiet=jnp.array([[0.4, 0.2]]),
        )[0]
        np.testing.assert_allclose(np.array(lc_default), np.array(lc_explicit))

    def test_overriding_changes_the_light_curve(self, grad_model):
        kwargs = dict(
            flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([20.0]),
            ar_long=jnp.array([0.0]), ar_size=jnp.array([10.0]),
            ar_smoothness=jnp.array([_SM]),
        )
        lc_default = make_lc(grad_model, **kwargs)[0]
        lc_overridden = make_lc(
            grad_model, **kwargs, ld_coeffs_quiet=jnp.array([[0.1, 0.05]]),
        )[0]
        assert not np.allclose(np.array(lc_default), np.array(lc_overridden))

    def test_shape_mismatch_raises(self, grad_model):
        with pytest.raises(ValueError, match="ld_coeffs_quiet"):
            make_lc(
                grad_model, flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.array([20.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
                ld_coeffs_quiet=jnp.array([[0.1, 0.05, 0.0]]),  # wrong n_coeffs
            )

    def test_grad_wrt_quiet_u1_is_finite_and_nonzero(self, grad_model):
        def f(u1):
            result = make_lc(
                grad_model, flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.array([20.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
                ld_coeffs_quiet=jnp.array([[u1, 0.2]]),
            )
            return jnp.sum(result[0])

        g = jax.grad(f)(jnp.float32(0.4))
        assert jnp.isfinite(g)
        assert jnp.abs(g) > 0


# ===================================================================
# 9.  Time-varying active regions
#
# Any of flux_active/ar_lat/ar_long/ar_size/ar_smoothness may carry an
# extra leading time axis (length = the model's original, pre-oversampling
# `times`) to let that property evolve over the observation, independently
# of the others. Omitting the extra axis on every one of them (today's
# usage) must take the exact code path that existed before this feature,
# at zero added cost -- most of the tests below exist to pin that down,
# not just to check the new path works.
# ===================================================================

class TestTimeVaryingAR:

    _NTIME = 12

    @pytest.fixture(scope="class")
    def evolving_model(self):
        return build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=np.linspace(0, 8.0, self._NTIME, endpoint=False), P_rot=10.0,
            stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )

    @pytest.fixture(scope="class")
    def evolving_model_oversampled(self):
        return build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=np.linspace(0, 8.0, self._NTIME, endpoint=False), P_rot=10.0,
            stellar_grid_size=50, ve=0.0, ld_mode="quadratic", oversample=3,
        )

    @pytest.fixture(scope="class")
    def evolving_model_oversampled_cubic(self):
        """Same as evolving_model_oversampled, but ar_time_interp='cubic'
        -- a build_system-level setting, like ld_mode, so it needs its own
        model rather than being choosable per make_lc call."""
        return build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=np.linspace(0, 8.0, self._NTIME, endpoint=False), P_rot=10.0,
            stellar_grid_size=50, ve=0.0, ld_mode="quadratic", oversample=3,
            ar_time_interp="cubic",
        )

    @pytest.fixture(scope="class")
    def evolving_model_cubic(self):
        """Same as evolving_model (oversample=1), but ar_time_interp='cubic'."""
        return build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=np.linspace(0, 8.0, self._NTIME, endpoint=False), P_rot=10.0,
            stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
            ar_time_interp="cubic",
        )

    # ---- Correctness: a time-varying array holding a constant value must
    # reproduce the static-shape call exactly (same kernel, different vmap
    # structure) --------------------------------------------------------

    def test_broadcast_time_varying_ar_size_matches_static(self, evolving_model):
        static_kwargs = dict(
            flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([20.0]),
            ar_long=jnp.array([0.0]), ar_smoothness=jnp.array([_SM]),
        )
        lc_static = make_lc(evolving_model, ar_size=jnp.array([10.0]), **static_kwargs)[0]
        ar_size_const = jnp.broadcast_to(jnp.array([10.0]), (self._NTIME, 1))
        lc_evolving = make_lc(evolving_model, ar_size=ar_size_const, **static_kwargs)[0]
        np.testing.assert_array_equal(np.array(lc_static), np.array(lc_evolving))

    def test_broadcast_time_varying_ar_lat_matches_static(self, evolving_model):
        static_kwargs = dict(
            flux_active=jnp.array(FLUX_SPOT), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )
        lc_static = make_lc(evolving_model, ar_lat=jnp.array([20.0]), **static_kwargs)[0]
        ar_lat_const = jnp.broadcast_to(jnp.array([20.0]), (self._NTIME, 1))
        lc_evolving = make_lc(evolving_model, ar_lat=ar_lat_const, **static_kwargs)[0]
        np.testing.assert_array_equal(np.array(lc_static), np.array(lc_evolving))

    def test_broadcast_time_varying_flux_active_matches_static(self, evolving_model):
        static_kwargs = dict(
            ar_lat=jnp.array([20.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )
        lc_static = make_lc(evolving_model, flux_active=jnp.array(FLUX_SPOT), **static_kwargs)[0]
        flux_active_const = jnp.broadcast_to(
            jnp.array(FLUX_SPOT), (self._NTIME, 1, len(WAVELENGTH))
        )
        lc_evolving = make_lc(evolving_model, flux_active=flux_active_const, **static_kwargs)[0]
        np.testing.assert_array_equal(np.array(lc_static), np.array(lc_evolving))

    # ---- Genuine evolution: finite, and actually different from a
    # constant-parameter call ------------------------------------------

    def test_evolving_ar_size_matches_independent_per_time_calls(self, evolving_model):
        """
        Ground-truth cross-check: the evolving path's output at each phase
        must match an independent build_system+make_lc call at that single
        time with the corresponding static ar_size -- i.e. the vmapped
        evolving kernel does the same per-phase physics as the ordinary
        static kernel, just batched over time.

        (A simpler "growing spot -> monotonically deepening dip" version of
        this test doesn't hold here: with P_rot=10 and times spanning 0-8,
        the AR rotates most of the way around the star, so the changing
        foreshortening/visibility confounds a naive monotonicity check.)
        """
        times = np.linspace(0, 8.0, self._NTIME, endpoint=False)
        ar_size_vals = np.linspace(3.0, 15.0, self._NTIME)
        ar_size_t = jnp.asarray(ar_size_vals)[:, None]

        lc_evolving, _ = make_lc(
            evolving_model, flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([0.0]),
            ar_long=jnp.array([0.0]), ar_size=ar_size_t, ar_smoothness=jnp.array([_SM]),
        )
        lc_evolving = np.array(lc_evolving)
        assert np.all(np.isfinite(lc_evolving))

        for i in [0, self._NTIME // 2, self._NTIME - 1]:
            single_model = build_system(
                wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
                times=times[i:i + 1], P_rot=10.0, stellar_grid_size=50, ve=0.0,
                ld_mode="quadratic",
            )
            lc_single, _ = make_lc(
                single_model, flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([0.0]),
                ar_long=jnp.array([0.0]), ar_size=jnp.array([ar_size_vals[i]]),
                ar_smoothness=jnp.array([_SM]),
            )
            np.testing.assert_allclose(
                float(lc_evolving[i]), float(np.array(lc_single)[0]), atol=1e-5,
                err_msg=f"evolving-path row {i} doesn't match an independent static call",
            )

    def test_evolving_position_differs_from_static(self, evolving_model):
        ar_lat_t = jnp.linspace(-30.0, 30.0, self._NTIME)[:, None]
        static_kwargs = dict(
            flux_active=jnp.array(FLUX_SPOT), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )
        lc_evolving = make_lc(evolving_model, ar_lat=ar_lat_t, **static_kwargs)[0]
        lc_static = make_lc(evolving_model, ar_lat=jnp.array([0.0]), **static_kwargs)[0]
        assert np.all(np.isfinite(np.array(lc_evolving)))
        assert not np.allclose(np.array(lc_evolving), np.array(lc_static))

    def test_evolving_flux_active_differs_from_static(self, evolving_model):
        flux_active_t = jnp.linspace(0.9, 0.4, self._NTIME)[:, None, None] \
            * jnp.ones((self._NTIME, 1, len(WAVELENGTH)))
        static_kwargs = dict(
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
        )
        lc_evolving = make_lc(evolving_model, flux_active=flux_active_t, **static_kwargs)[0]
        lc_static = make_lc(evolving_model, flux_active=jnp.array(FLUX_SPOT), **static_kwargs)[0]
        assert np.all(np.isfinite(np.array(lc_evolving)))
        assert not np.allclose(np.array(lc_evolving), np.array(lc_static))

    def test_mixed_static_and_evolving_params(self, evolving_model):
        """Only some of the five AR parameters need to be time-varying;
        the rest are broadcast across time as usual."""
        ar_size_t = jnp.linspace(5.0, 15.0, self._NTIME)[:, None]
        lc, _ = make_lc(
            evolving_model, flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([0.0]),
            ar_long=jnp.array([0.0]), ar_size=ar_size_t, ar_smoothness=jnp.array([_SM]),
        )
        assert np.all(np.isfinite(np.array(lc)))

    def test_two_active_regions_time_varying(self, evolving_model):
        """Time-varying shape is (ntime, nar) -- nar=2 here."""
        ar_size_t = jnp.stack([
            jnp.linspace(3.0, 12.0, self._NTIME),
            jnp.linspace(12.0, 3.0, self._NTIME),
        ], axis=-1)  # (ntime, 2)
        lc, _ = make_lc(
            evolving_model,
            flux_active=jnp.broadcast_to(jnp.array(FLUX_SPOT), (2, len(WAVELENGTH))),
            ar_lat=jnp.array([20.0, -20.0]), ar_long=jnp.array([0.0, 180.0]),
            ar_size=ar_size_t, ar_smoothness=jnp.array([_SM, _SM]),
        )
        assert np.all(np.isfinite(np.array(lc)))

    def test_oversample_with_evolving_param(self, evolving_model_oversampled):
        """Time-varying values are specified per original time, not per
        oversampled sub-exposure, and expanded internally."""
        ar_size_t = jnp.linspace(5.0, 15.0, self._NTIME)[:, None]
        lc, star_maps = make_lc(
            evolving_model_oversampled, flux_active=jnp.array(FLUX_SPOT),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]), ar_size=ar_size_t,
            ar_smoothness=jnp.array([_SM]),
        )
        assert np.array(lc).shape == (self._NTIME,)  # nwave == 1: axis already dropped
        assert star_maps.shape[0] == self._NTIME
        assert np.all(np.isfinite(np.array(lc)))

    # ---- Interpolation (ar_time_interp) ----------------------------------

    def test_oversample_interpolation_matches_manual_reference(self, evolving_model_oversampled):
        """
        Ground-truth cross-check for oversample>1: make_lc's output for an
        (ntime, nar) time-varying ar_size must match manually
        linear-interpolating ar_size onto the model's exact sub-exposure
        times (model["times_oversampled"]), evaluating each sub-exposure
        independently, and averaging per block -- i.e. genuine
        interpolation is what happens internally, not a step-function
        repeat (which would instead match evaluating only at the nearest
        original cadence point, not this interpolated reference).
        """
        ar_size_vals = np.linspace(5.0, 15.0, self._NTIME)
        ar_size_t = jnp.asarray(ar_size_vals)[:, None]

        # evolving_model_oversampled defaults to ar_time_interp="linear"
        # (build_system's own default, like ld_mode).
        lc_interp, _ = make_lc(
            evolving_model_oversampled, flux_active=jnp.array(FLUX_SPOT),
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]), ar_size=ar_size_t,
            ar_smoothness=jnp.array([_SM]),
        )
        lc_interp = np.array(lc_interp)

        times_orig = np.asarray(evolving_model_oversampled["times"])
        times_over = np.asarray(evolving_model_oversampled["times_oversampled"])
        # np.interp clamps outside [times_orig[0], times_orig[-1]]; interpax's
        # extrap=True (needed since sub-exposure times spill slightly past
        # the first/last cadence point) instead extends the boundary slope,
        # so the reference must match that, not plain clamped np.interp.
        ar_size_over = np.interp(times_over, times_orig, ar_size_vals)
        slope_left  = (ar_size_vals[1] - ar_size_vals[0]) / (times_orig[1] - times_orig[0])
        slope_right = (ar_size_vals[-1] - ar_size_vals[-2]) / (times_orig[-1] - times_orig[-2])
        ar_size_over = np.where(
            times_over < times_orig[0],
            ar_size_vals[0] + slope_left * (times_over - times_orig[0]),
            ar_size_over,
        )
        ar_size_over = np.where(
            times_over > times_orig[-1],
            ar_size_vals[-1] + slope_right * (times_over - times_orig[-1]),
            ar_size_over,
        )

        ref_model = build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=times_over, P_rot=10.0, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )
        lc_ref, _ = make_lc(
            ref_model, flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([0.0]),
            ar_long=jnp.array([0.0]), ar_size=jnp.asarray(ar_size_over)[:, None],
            ar_smoothness=jnp.array([_SM]),
        )
        oversample = evolving_model_oversampled["oversample"]
        lc_ref_avg = np.array(lc_ref).reshape(self._NTIME, oversample).mean(axis=1)

        np.testing.assert_allclose(lc_interp, lc_ref_avg, atol=1e-5)

    def test_linear_vs_cubic_differ_for_nonlinear_evolution(
        self, evolving_model_oversampled, evolving_model_oversampled_cubic,
    ):
        """
        linear and cubic interpolation must give different sub-exposure
        values for a genuinely nonlinear (not just non-constant) evolution.
        ar_time_interp is fixed at build_system time (like ld_mode), so
        comparing the two methods means comparing two separately-built
        models, not one model with a per-call override.
        """
        assert evolving_model_oversampled["ar_time_interp"] == "linear"
        assert evolving_model_oversampled_cubic["ar_time_interp"] == "cubic"

        times_orig = np.asarray(evolving_model_oversampled["times"])
        ar_size_t = jnp.asarray(5.0 + 1.5 * (times_orig - times_orig[0]) ** 2)[:, None]
        kwargs = dict(
            flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([0.0]),
            ar_long=jnp.array([0.0]), ar_size=ar_size_t, ar_smoothness=jnp.array([_SM]),
        )
        lc_linear = np.array(make_lc(evolving_model_oversampled, **kwargs)[0])
        lc_cubic  = np.array(make_lc(evolving_model_oversampled_cubic, **kwargs)[0])
        assert np.all(np.isfinite(lc_linear))
        assert np.all(np.isfinite(lc_cubic))
        # An explicit magnitude threshold rather than np.allclose's default
        # tolerance, which was loose enough to call a genuine (if small)
        # difference "close" -- both curves are finite and physically
        # sensible, but must not be the *same* curve.
        assert np.max(np.abs(lc_linear - lc_cubic)) > 1e-6

    def test_cubic_overshoot_is_clipped_to_physical_domain(self, evolving_model_oversampled_cubic):
        """
        ar_time_interp='cubic' is a natural spline and can overshoot the
        input knots between them, even though the knots themselves satisfy
        ar_size>=0/ar_smoothness>=1/flux_active>=0 -- make_lc must re-clip
        the interpolated sub-exposure values back into the physical domain.

        Ground-truth cross-check: replicate the exact interpax interpolation
        make_lc performs internally, clip it by hand to the physical domain,
        and feed the result through a model built directly at the
        sub-exposure times (the static code path, which needs no
        interpolation) -- make_lc's actual output must match this clipped
        reference, not the raw (unclamped) spline.
        """
        times_orig = np.asarray(evolving_model_oversampled_cubic["times"])
        times_over = np.asarray(evolving_model_oversampled_cubic["times_oversampled"])

        # Knots alternate between a comfortably-valid value and exactly the
        # boundary, which makes a natural cubic spline ring past the
        # boundary between them.
        ar_size_vals = np.tile([8.0, 0.0], self._NTIME // 2)
        smooth_vals  = np.tile([_SM, 1.0], self._NTIME // 2)
        flux_vals    = np.tile([0.9, 0.0], self._NTIME // 2)

        def _raw_interp(vals):
            return np.array(interpax.interp1d(
                jnp.asarray(times_over), jnp.asarray(times_orig), jnp.asarray(vals),
                method="cubic2", extrap=True,
            ))

        ar_size_raw = _raw_interp(ar_size_vals)
        smooth_raw  = _raw_interp(smooth_vals)
        flux_raw    = _raw_interp(flux_vals)
        # Self-check that this fixture actually exercises the clip, so the
        # test doesn't silently pass without ever overshooting.
        assert np.any(ar_size_raw < 0)
        assert np.any(smooth_raw < 1)
        assert np.any(flux_raw < 0)

        lc, _ = make_lc(
            evolving_model_oversampled_cubic,
            flux_active=jnp.asarray(flux_vals)[:, None, None],
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.asarray(ar_size_vals)[:, None],
            ar_smoothness=jnp.asarray(smooth_vals)[:, None],
        )
        lc = np.array(lc)
        assert np.all(np.isfinite(lc))

        ref_model = build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=times_over, P_rot=10.0, stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
        )
        lc_ref, _ = make_lc(
            ref_model,
            flux_active=jnp.asarray(np.maximum(flux_raw, 0.0))[:, None, None],
            ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
            ar_size=jnp.asarray(np.maximum(ar_size_raw, 0.0))[:, None],
            ar_smoothness=jnp.asarray(np.maximum(smooth_raw, 1.0))[:, None],
        )
        oversample = evolving_model_oversampled_cubic["oversample"]
        lc_ref_avg = np.array(lc_ref).reshape(self._NTIME, oversample).mean(axis=1)

        np.testing.assert_allclose(lc, lc_ref_avg, atol=1e-5)

    def test_cubic_matches_linear_when_oversample_is_one(self, evolving_model, evolving_model_cubic):
        """With oversample == 1, the sub-exposure grid equals the original
        cadence grid exactly, so both interpolation laws reproduce the
        given knots exactly and must agree bit-for-bit."""
        ar_size_t = jnp.linspace(5.0, 15.0, self._NTIME)[:, None]
        kwargs = dict(
            flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([0.0]),
            ar_long=jnp.array([0.0]), ar_size=ar_size_t, ar_smoothness=jnp.array([_SM]),
        )
        lc_linear = make_lc(evolving_model, **kwargs)[0]
        lc_cubic  = make_lc(evolving_model_cubic, **kwargs)[0]
        np.testing.assert_array_equal(np.array(lc_linear), np.array(lc_cubic))

    def test_cubic_requires_at_least_two_times(self):
        single_time_model = build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=np.array([0.0]), P_rot=10.0,
            stellar_grid_size=50, ve=0.0, ld_mode="quadratic",
            ar_time_interp="cubic",
        )
        with pytest.raises(ValueError, match="ar_time_interp='cubic'"):
            make_lc(
                single_time_model, flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.zeros((1, 1)), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    # ---- Shape validation -----------------------------------------------

    def test_ar_lat_wrong_time_length_raises(self, evolving_model):
        with pytest.raises(ValueError, match="ar_lat"):
            make_lc(
                evolving_model, flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.zeros((self._NTIME - 1, 1)), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_flux_active_wrong_time_shape_raises(self, evolving_model):
        with pytest.raises(ValueError, match="flux_active"):
            make_lc(
                evolving_model,
                flux_active=jnp.zeros((self._NTIME - 1, 1, len(WAVELENGTH))),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_evolving_ar_smoothness_below_one_raises(self, evolving_model):
        bad_smoothness = jnp.broadcast_to(jnp.array([0.5]), (self._NTIME, 1))
        with pytest.raises(ValueError, match="ar_smoothness must be >= 1"):
            make_lc(
                evolving_model, flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=bad_smoothness,
            )

    def test_evolving_ar_size_below_zero_raises(self, evolving_model):
        bad_size = jnp.broadcast_to(jnp.array([-5.0]), (self._NTIME, 1))
        with pytest.raises(ValueError, match="ar_size must be >= 0"):
            make_lc(
                evolving_model, flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=bad_size, ar_smoothness=jnp.array([_SM]),
            )

    def test_evolving_flux_active_below_zero_raises(self, evolving_model):
        bad_flux = jnp.broadcast_to(jnp.array([-0.1]), (self._NTIME, 1, len(WAVELENGTH)))
        with pytest.raises(ValueError, match="flux_active must be >= 0"):
            make_lc(
                evolving_model, flux_active=bad_flux,
                ar_lat=jnp.array([0.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_ar_size_wrong_static_length_raises(self, evolving_model):
        """Triggers the time-varying path via ar_lat, then ar_size's own
        (nar,)-mismatch check inside it."""
        ar_lat_t = jnp.broadcast_to(jnp.array([0.0]), (self._NTIME, 1))
        with pytest.raises(ValueError, match="ar_size"):
            make_lc(
                evolving_model, flux_active=jnp.array(FLUX_SPOT),
                ar_lat=ar_lat_t, ar_long=jnp.array([0.0]),
                ar_size=jnp.array([1.0, 2.0]),  # size 2, but nar == 1
                ar_smoothness=jnp.array([_SM]),
            )

    def test_ar_smoothness_wrong_static_shape_raises(self, evolving_model):
        ar_lat_t = jnp.broadcast_to(jnp.array([0.0]), (self._NTIME, 1))
        with pytest.raises(ValueError, match="ar_smoothness"):
            make_lc(
                evolving_model, flux_active=jnp.array(FLUX_SPOT),
                ar_lat=ar_lat_t, ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]),
                ar_smoothness=jnp.array([1.0, 2.0]),  # size 2, but nar == 1
            )

    def test_flux_active_wrong_static_2d_shape_raises(self, evolving_model):
        ar_lat_t = jnp.broadcast_to(jnp.array([0.0]), (self._NTIME, 1))
        with pytest.raises(ValueError, match="flux_active"):
            make_lc(
                evolving_model,
                flux_active=jnp.zeros((2, len(WAVELENGTH))),  # (nar, nwave) but nar == 1
                ar_lat=ar_lat_t, ar_long=jnp.array([0.0]),
                ar_size=jnp.array([10.0]), ar_smoothness=jnp.array([_SM]),
            )

    def test_warns_when_nar_coincides_with_ntime(self, evolving_model):
        """
        nar is read off ar_lat's own trailing axis before make_lc knows
        whether any parameter is time-varying. A caller who means a single
        AR to evolve over ntime epochs, but forgets the extra leading axis
        on ar_lat/ar_long (passing flat (ntime,) arrays instead of
        (ntime, 1)), silently ends up with ntime separate *static* active
        regions instead -- indistinguishable from a genuine
        ntime-active-region model by shape alone whenever nar == ntime, so
        make_lc must warn in that case rather than stay silent.

        This reproduces the footgun end-to-end: ar_lat/ar_long become
        (accidentally) per-region constants, ar_size is shaped to broadcast
        the same evolving curve onto every one of the ntime fake regions,
        and the call succeeds without any ValueError -- shape validation
        alone cannot catch this, only the warning does.
        """
        ntime = self._NTIME
        nwave = len(WAVELENGTH)
        ar_lat_flat  = jnp.linspace(-10.0, 10.0, ntime)  # meant as 1 AR's time-course...
        ar_long_flat = jnp.zeros(ntime)                   # ...but nar ends up == ntime instead
        ar_size_t = jnp.broadcast_to(
            jnp.linspace(5.0, 15.0, ntime)[:, None], (ntime, ntime)
        )
        with pytest.warns(UserWarning, match="nar.*ntime|ntime.*nar"):
            lc, _ = make_lc(
                evolving_model,
                flux_active=jnp.full((ntime, nwave), FLUX_SPOT[0]),
                ar_lat=ar_lat_flat, ar_long=ar_long_flat,
                ar_size=ar_size_t, ar_smoothness=jnp.full(ntime, _SM),
            )
        assert np.all(np.isfinite(np.array(lc)))

    def test_no_warning_when_nar_differs_from_ntime(self, evolving_model):
        """Normal time-varying usage -- nar (1 active region here) differs
        from ntime (self._NTIME) -- must not trigger the nar==ntime
        footgun warning."""
        ar_size_t = jnp.linspace(5.0, 15.0, self._NTIME)[:, None]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            make_lc(
                evolving_model, flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([0.0]),
                ar_long=jnp.array([0.0]), ar_size=ar_size_t, ar_smoothness=jnp.array([_SM]),
            )
        assert not any("ntime" in str(w.message) for w in record)

    # ---- Autodiff ---------------------------------------------------------

    def test_grad_through_evolving_ar_size_is_finite_and_nonzero(self, evolving_model):
        def f(size0):
            ar_size_t = jnp.linspace(size0, size0 + 5.0, self._NTIME)[:, None]
            lc, _ = make_lc(
                evolving_model, flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([0.0]),
                ar_long=jnp.array([0.0]), ar_size=ar_size_t, ar_smoothness=jnp.array([_SM]),
            )
            return jnp.sum(lc)

        g = jax.grad(f)(jnp.float32(8.0))
        assert jnp.isfinite(g)
        assert jnp.abs(g) > 0


# ===================================================================
# 10.  BJD-scale reference-epoch shift (numerical precision)
#
# build_system now subtracts a reference epoch (model["t_ref"]) from
# times/t0 internally, before either reaches a JAX array, so absolute
# BJD-scale transit epochs don't need jax_enable_x64 to stay precise.
# See sajax/planet.py's build_transit_model docstring.
# ===================================================================

class TestBjdReferenceEpoch:

    _T0_BJD  = 2_460_123.456789
    _PERIOD  = 3.14159265
    _TRANSIT_PARAMS = dict(
        t0=_T0_BJD, period=_PERIOD, a_over_rstar=15.0,
        inclination=np.pi / 2.0, k=0.1,
    )

    @pytest.fixture(scope="class")
    def bjd_model(self):
        times = self._T0_BJD + np.linspace(-0.1, 0.1, 20)
        return build_system(
            wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
            times=times, P_rot=25.0, stellar_grid_size=STELLAR_GRID, ve=VE,
            ld_mode="quadratic", **self._TRANSIT_PARAMS,
        )

    def test_t_ref_is_stored_and_matches_floor_of_times_min(self, bjd_model):
        expected = float(np.floor(bjd_model["times"].min()))
        assert bjd_model["t_ref"] == pytest.approx(expected)

    def test_quiet_star_model_also_has_t_ref(self, stellar_only_model):
        """
        t_ref/times/times_oversampled are computed unconditionally, not
        just when a transit is attached -- a long enough baseline benefits
        from the reduction for any downstream numerical work, not only the
        transit-position computation.
        """
        assert "t_ref" in stellar_only_model
        expected = float(np.floor(stellar_only_model["times"].min()))
        assert stellar_only_model["t_ref"] == pytest.approx(expected)
        # times_oversampled must be shifted relative to the reference
        # epoch: within one day of the (already-small, here near-zero)
        # times span, not a BJD-scale (~2.46e6) magnitude.
        max_shifted = float(jnp.max(jnp.abs(stellar_only_model["times_oversampled"])))
        assert max_shifted < 2.0
        assert max_shifted == pytest.approx(
            float(np.max(np.abs(stellar_only_model["times"] - stellar_only_model["t_ref"]))),
            abs=1e-3,
        )

    def test_still_warns_for_long_baseline_high_cadence_after_reduction(self):
        """
        build_system's automatic reduction handles the common case, but not
        the fundamental float32 precision limit: a genuinely long baseline
        (here ~10 years) combined with a fine cadence (~1 second) anywhere
        in it leaves a large enough *post-reduction* magnitude (~thousands
        of days) that float32 rounding is still a significant fraction of
        that cadence, so the warning must still fire even after reduction.
        """
        span_days = 3650.0  # ~10 years
        times = self._T0_BJD + np.concatenate([
            np.linspace(0.0, span_days, 20),
            [span_days / 2, span_days / 2 + 1.0 / 86400.0],  # a 1-second-spaced pair
        ])
        with pytest.warns(UserWarning, match="float32 rounding"):
            build_system(
                wavelength=WAVELENGTH, flux_quiet=FLUX_QUIET, **BASE_PARAMS,
                times=times, P_rot=25.0, stellar_grid_size=STELLAR_GRID, ve=VE,
                ld_mode="quadratic", **self._TRANSIT_PARAMS,
            )

    def test_static_and_dynamic_paths_agree_at_bjd_scale(self, bjd_model):
        """
        The static (build-time-baked) and dynamic (make_lc-level override,
        possibly traced t0) transit-position paths must agree when given
        the *same* BJD-scale t0 -- both now go through the reference-epoch
        shift (statically inside build_transit_model, dynamically via
        make_lc subtracting model["t_ref"]), and should be numerically
        indistinguishable at typical float32 precision.
        """
        kwargs = dict(
            flux_active=jnp.array(FLUX_SPOT), ar_lat=jnp.array([5.0]),
            ar_long=jnp.array([0.0]), ar_size=jnp.array([8.0]),
            ar_smoothness=jnp.array([_SM]),
        )
        lc_static = make_lc(bjd_model, **kwargs)[0]
        lc_dynamic = make_lc(bjd_model, **kwargs, **self._TRANSIT_PARAMS)[0]
        np.testing.assert_allclose(
            np.array(lc_static), np.array(lc_dynamic), rtol=1e-4,
        )

    def test_grad_wrt_bjd_scale_t0_is_finite_without_x64(self, bjd_model):
        """
        The whole point of the fix: gradient-based retrieval of an
        absolute BJD-scale t0 must give a finite, informative gradient
        without needing jax_enable_x64.
        """
        assert not jax.config.jax_enable_x64

        def f(t0):
            lc, _ = make_lc(
                bjd_model, flux_active=jnp.array(FLUX_SPOT),
                ar_lat=jnp.array([5.0]), ar_long=jnp.array([0.0]),
                ar_size=jnp.array([8.0]), ar_smoothness=jnp.array([_SM]),
                t0=t0, period=self._PERIOD, a_over_rstar=15.0,
                inclination=np.pi / 2.0, k=0.1,
            )
            return jnp.sum(lc)

        g = jax.grad(f)(self._T0_BJD)
        assert jnp.isfinite(g)
