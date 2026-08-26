"""
tests/test_planet.py — Tests for the planet.py orbital module.
"""

import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sajax.planet import (
    _kepler,
    planet_sky_position,
    compute_planet_sky_positions,
    compute_multi_planet_sky_positions,
    build_transit_model,
    stellar_density_to_a_over_rstar,
    a_over_rstar_to_stellar_density,
    _compute_planet_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sky_pos(t, **overrides):
    """Call planet_sky_position with default circular edge-on orbit."""
    defaults = dict(
        t0=0.0, period=10.0, a_over_rstar=15.0,
        inclination=np.pi / 2.0, ecc=0.0, omega_peri=0.0,
    )
    defaults.update(overrides)
    return planet_sky_position(jnp.float32(t), **defaults)


def _r3d(t, **overrides):
    """3-D distance from stellar centre (should equal a for circular orbit)."""
    X, Y, Z = _sky_pos(t, **overrides)
    return float(jnp.sqrt(X**2 + Y**2 + Z**2))


# ===================================================================
# 1.  Kepler solver
# ===================================================================

class TestKepler:
    """Tests for _kepler(M, ecc) → (sin f, cos f)."""

    # --- Identities for e = 0 -------------------------------------------

    def test_circular_sinf_equals_sinM(self):
        """For e=0, f = M, so sin f = sin M."""
        M = jnp.linspace(0.0, 2 * jnp.pi, 120)
        sinf, cosf = _kepler(M, 0.0)
        np.testing.assert_allclose(np.array(sinf), np.sin(np.array(M)), atol=1e-5,
            err_msg="e=0: sin f should equal sin M")

    def test_circular_cosf_equals_cosM(self):
        """For e=0, f = M, so cos f = cos M."""
        M = jnp.linspace(0.0, 2 * jnp.pi, 120)
        sinf, cosf = _kepler(M, 0.0)
        np.testing.assert_allclose(np.array(cosf), np.cos(np.array(M)), atol=1e-5,
            err_msg="e=0: cos f should equal cos M")

    # --- Pythagorean identity -------------------------------------------

    @pytest.mark.parametrize("ecc", [0.0, 0.1, 0.3, 0.5, 0.7, 0.85])
    def test_unit_norm(self, ecc):
        """sin²f + cos²f = 1 for all M and all valid eccentricities."""
        M = jnp.linspace(0.0, 2 * jnp.pi, 200)
        sinf, cosf = _kepler(M, ecc)
        norm2 = np.array(sinf**2 + cosf**2)
        np.testing.assert_allclose(norm2, 1.0, atol=1e-5,
            err_msg=f"Pythagorean identity violated at ecc={ecc}")

    # --- Fixed points -------------------------------------------------------

    @pytest.mark.parametrize("ecc", [0.0, 0.2, 0.5, 0.8])
    def test_periapsis_at_M_zero(self, ecc):
        """At M=0 (periapsis), f=0 so sin f=0, cos f=1."""
        sinf, cosf = _kepler(jnp.float32(0.0), ecc)
        assert abs(float(sinf))       < 1e-5, f"e={ecc}: sin f should be 0 at M=0"
        assert abs(float(cosf) - 1.0) < 1e-5, f"e={ecc}: cos f should be 1 at M=0"

    @pytest.mark.parametrize("ecc", [0.0, 0.2, 0.5, 0.7])
    def test_apoapsis_at_M_pi(self, ecc):
        """At M=π (apoapsis), f=π so sin f≈0, cos f≈-1."""
        sinf, cosf = _kepler(jnp.float32(np.pi), ecc)
        assert abs(float(sinf))       < 1e-4, f"e={ecc}: sin f should be 0 at M=π"
        assert abs(float(cosf) + 1.0) < 1e-4, f"e={ecc}: cos f should be -1 at M=π"

    # --- Kepler's equation residual -----------------------------------------

    def test_kepler_equation_satisfied(self):
        """Reconstructed M = E - e sin E must match the input M.

        We convert (sin f, cos f) → E via
            tan(E/2) = sqrt((1-e)/(1+e)) · tan(f/2)
        and verify M = E - e sin E holds.
        """
        M   = jnp.linspace(0.02, 2 * jnp.pi - 0.02, 300)
        ecc = 0.55
        sinf, cosf = _kepler(M, ecc)

        # (sin f, cos f) → E
        half_f = jnp.arctan2(sinf, 1.0 + cosf)
        E = 2.0 * jnp.arctan2(
            jnp.sqrt(1.0 - ecc) * jnp.sin(half_f),
            jnp.sqrt(1.0 + ecc) * jnp.cos(half_f),
        )
        M_reconstructed = E - ecc * jnp.sin(E)

        # Wrap to [-π, π] for robust comparison
        wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(
            np.array(wrap(M_reconstructed)), np.array(wrap(M)),
            atol=1e-4, err_msg="Kepler equation not satisfied")

    # --- Stability at high eccentricity -----------------------------------

    def test_high_eccentricity_stable(self):
        """Solver must remain numerically stable near e=0.9."""
        M = jnp.linspace(0.0, 2 * jnp.pi, 300)
        sinf, cosf = _kepler(M, 0.9)
        norm2 = np.array(sinf**2 + cosf**2)
        np.testing.assert_allclose(norm2, 1.0, atol=1e-4,
            err_msg="High-e Kepler solver: unit-norm violated")

    def test_output_finite_for_all_M(self):
        """(sinf, cosf) should be finite for all M ∈ [0, 2π) and valid e."""
        M = jnp.linspace(0.0, 2 * jnp.pi, 200)
        for ecc in [0.0, 0.3, 0.6, 0.9]:
            sinf, cosf = _kepler(M, ecc)
            assert np.all(np.isfinite(np.array(sinf))), f"sinf not finite (e={ecc})"
            assert np.all(np.isfinite(np.array(cosf))), f"cosf not finite (e={ecc})"

    # --- Differentiability -------------------------------------------------

    def test_differentiable_wrt_M(self):
        """jax.grad should work on a scalar function of M (tests autodiff path)."""
        def scalar_f(M_scalar):
            sinf, cosf = _kepler(M_scalar, 0.4)
            return sinf + cosf

        grad_fn = jax.grad(scalar_f)
        g = grad_fn(jnp.float32(1.2))
        assert np.isfinite(float(g)), "Gradient w.r.t. M should be finite"

    def test_differentiable_wrt_ecc(self):
        """jax.grad should work w.r.t. eccentricity."""
        def scalar_f(ecc_scalar):
            sinf, cosf = _kepler(jnp.float32(1.2), ecc_scalar)
            return sinf + cosf

        grad_fn = jax.grad(scalar_f)
        g = grad_fn(jnp.float32(0.3))
        assert np.isfinite(float(g)), "Gradient w.r.t. ecc should be finite"


# ===================================================================
# 2.  planet_sky_position — single epoch
# ===================================================================

class TestPlanetSkyPosition:

    # --- At mid-transit (t = t0) ----------------------------------------

    def test_mid_transit_X_near_zero(self):
        """At t=t0, the planet crosses the sky centre: X ≈ 0."""
        X, Y, Z = _sky_pos(0.0)
        assert abs(float(X)) < 0.1, f"Mid-transit X should ≈ 0, got {float(X):.4f}"

    def test_mid_transit_Z_positive(self):
        """At t=t0, the planet is in front of the star: Z > 0."""
        X, Y, Z = _sky_pos(0.0)
        assert float(Z) > 0, f"Mid-transit Z should be > 0, got {float(Z):.4f}"

    def test_mid_transit_edge_on_Y_near_zero(self):
        """For i=π/2 at t=t0 (inferior conjunction): Y ≈ 0."""
        X, Y, Z = _sky_pos(0.0)
        assert abs(float(Y)) < 0.1, f"Edge-on mid-transit Y should ≈ 0, got {float(Y):.4f}"

    # --- At opposition (t = t0 + P/2) ------------------------------------

    def test_opposition_Z_negative(self):
        """At t = t0 + P/2, the planet is behind the star: Z < 0."""
        X, Y, Z = _sky_pos(5.0)   # period = 10.0 → half-period = 5.0
        assert float(Z) < 0, f"Opposition Z should be < 0, got {float(Z):.4f}"

    # --- Circular orbit geometry -----------------------------------------

    def test_circular_orbit_constant_3d_radius(self):
        """For e=0, the 3-D distance from the stellar centre must be
        constant and equal to a_over_rstar at all orbital phases."""
        a = 15.0
        P = 10.0
        for t in np.linspace(0.0, P, 50, endpoint=False):
            r = _r3d(t)
            assert abs(r - a) < 0.05, (
                f"Circular orbit: r = {r:.4f} ≠ a = {a} at t={t:.2f}"
            )

    # --- Impact parameter -----------------------------------------------

    def test_impact_parameter_inclined_orbit(self):
        """At mid-transit, |Y| ≈ a cos(i) (impact parameter b)."""
        inc = np.deg2rad(80.0)
        a   = 15.0
        expected_b = a * np.cos(inc)
        X, Y, Z = _sky_pos(0.0, inclination=inc)
        assert abs(abs(float(Y)) - expected_b) < 0.5, (
            f"Impact parameter: expected ≈{expected_b:.3f}, got |Y|={abs(float(Y)):.3f}"
        )

    # --- Periodicity -------------------------------------------------------

    def test_position_is_periodic(self):
        """Position at t and t + P must be identical."""
        P = 10.0
        t = 3.7
        X1, Y1, Z1 = _sky_pos(t)
        X2, Y2, Z2 = _sky_pos(t + P)
        np.testing.assert_allclose(
            [float(X1), float(Y1), float(Z1)],
            [float(X2), float(Y2), float(Z2)],
            atol=1e-3, err_msg="Position not periodic with P")

    # --- Sky separation monotone near transit ----------------------------

    def test_sky_separation_decreases_toward_transit(self):
        """Sky separation √(X²+Y²) should decrease as t → t0."""
        sep = lambda t: float(jnp.sqrt(sum(v**2 for v in _sky_pos(t)[:2])))
        assert sep(-2.0) > sep(-0.1), \
            "Sky separation should decrease as planet approaches transit"

    # --- Eccentric orbit periapsis / apoapsis radii ----------------------

    def test_eccentric_periapsis_apoapsis_radii(self):
        """For an eccentric orbit, min/max 3-D radii should equal a(1±e)."""
        ecc = 0.4
        a   = 15.0
        P   = 10.0
        radii = [_r3d(t, ecc=ecc) for t in np.linspace(0, P, 1000, endpoint=False)]
        np.testing.assert_allclose(min(radii), a * (1 - ecc), rtol=0.02,
            err_msg="Periapsis distance should be a(1-e)")
        np.testing.assert_allclose(max(radii), a * (1 + ecc), rtol=0.02,
            err_msg="Apoapsis distance should be a(1+e)")

    # --- Output quality --------------------------------------------------

    def test_output_finite_all_phases(self):
        """(X, Y, Z) should be finite for all orbital phases."""
        P = 10.0
        for t in np.linspace(0.0, P, 100, endpoint=False):
            X, Y, Z = _sky_pos(t)
            assert np.isfinite(float(X)) and np.isfinite(float(Y)) and np.isfinite(float(Z)), \
                f"Non-finite position at t={t:.3f}"

    # --- Differentiability of full sky-position function ----------------

    def test_differentiable_wrt_t0(self):
        """jax.grad w.r.t. t0 should be finite (used in gradient-based fitting)."""
        def f(t0):
            X, Y, Z = planet_sky_position(
                jnp.float32(0.05), t0=t0, period=10.0, a_over_rstar=15.0,
                inclination=np.pi / 2.0, ecc=0.0, omega_peri=0.0,
            )
            return X + Y + Z
        g = jax.grad(f)(jnp.float32(0.0))
        assert np.isfinite(float(g)), "Gradient w.r.t. t0 should be finite"

    def test_differentiable_wrt_inclination(self):
        """jax.grad w.r.t. inclination should be finite."""
        def f(inc):
            X, Y, Z = planet_sky_position(
                jnp.float32(0.5), t0=0.0, period=10.0, a_over_rstar=15.0,
                inclination=inc, ecc=0.0, omega_peri=0.0,
            )
            return Y
        g = jax.grad(f)(jnp.float32(np.pi / 2.0))
        assert np.isfinite(float(g))

    def test_differentiable_wrt_ecc(self):
        """jax.grad w.r.t. eccentricity should be finite away from transit centre."""
        def f(ecc):
            X, Y, Z = planet_sky_position(
                jnp.float32(1.5), t0=0.0, period=10.0, a_over_rstar=15.0,
                inclination=np.pi / 2.0, ecc=ecc, omega_peri=0.0,
            )
            return X + Y + Z
        g = jax.grad(f)(jnp.float32(0.2))
        assert np.isfinite(float(g))


# ============================================================================================
# 2b.  Planetary sky-projected spin-orbit angle -- rotation of (X, Y) about the stellar centre
# ============================================================================================

class TestObliquity:
    """
    Tests for the sky-projected spin-orbit angle rotation
    in ``planet_sky_position``. The stellar spin axis is fixed along sky-Y
    elsewhere in sajax (geometry.py / core.py's ``vel_col``), so sp_orb
    lives entirely in this rotation of the planet's trajectory.
    """

    # --- Backwards compatibility ------------------------------------------

    def test_default_matches_omitted_argument(self):
        """Omitting spin-orbit angle must be identical to sp_orb=0.0."""
        X1, Y1, Z1 = _sky_pos(0.3)
        X2, Y2, Z2 = _sky_pos(0.3, sp_orb=0.0)
        np.testing.assert_allclose([float(X1), float(Y1), float(Z1)],
                                    [float(X2), float(Y2), float(Z2)])

    def test_zero_obliquity_matches_pre_obliquity_geometry(self):
        """sp_orb=0 must reproduce the un-rotated (legacy) X, Y, Z."""
        for t in np.linspace(0.0, 10.0, 15, endpoint=False):
            X0, Y0, Z0 = _sky_pos(t)
            X, Y, Z = _sky_pos(t, sp_orb=0.0)
            np.testing.assert_allclose([float(X), float(Y), float(Z)],
                                        [float(X0), float(Y0), float(Z0)], atol=1e-5)

    # --- Rotation correctness ----------------------------------------------

    @pytest.mark.parametrize("obliquity_deg", [15.0, 30.0, 45.0, 90.0, 135.0, 200.0, 315.0])
    def test_rotation_matches_manual_formula(self, obliquity_deg):
        """(X, Y) must equal the standard 2-D rotation of the λ=0 position."""
        t = 0.7
        X0, Y0, Z0 = _sky_pos(t)
        lam = np.deg2rad(obliquity_deg)
        X_expected = float(X0) * np.cos(lam) - float(Y0) * np.sin(lam)
        Y_expected = float(X0) * np.sin(lam) + float(Y0) * np.cos(lam)
        X, Y, Z = _sky_pos(t, sp_orb=lam)
        np.testing.assert_allclose(float(X), X_expected, atol=1e-4)
        np.testing.assert_allclose(float(Y), Y_expected, atol=1e-4)
        np.testing.assert_allclose(float(Z), float(Z0), atol=1e-5,
            err_msg="spin-orbit angle must not affect Z (line-of-sight position)")

    def test_quarter_turn_swaps_axes(self):
        """sp_orb=π/2 must map (X, Y) -> (-Y, X)."""
        X0, Y0, Z0 = _sky_pos(0.3)
        X, Y, Z = _sky_pos(0.3, sp_orb=np.pi / 2.0)
        np.testing.assert_allclose(float(X), -float(Y0), atol=1e-4)
        np.testing.assert_allclose(float(Y),  float(X0), atol=1e-4)

    def test_full_turn_is_identity(self):
        """sp_orb=2π must reproduce sp_orb=0 (rotation is periodic)."""
        X0, Y0, Z0 = _sky_pos(0.4)
        X, Y, Z = _sky_pos(0.4, sp_orb=2.0 * np.pi)
        np.testing.assert_allclose(float(X), float(X0), atol=1e-3)
        np.testing.assert_allclose(float(Y), float(Y0), atol=1e-3)

    # --- Rotation-invariant quantities --------------------------------------

    @pytest.mark.parametrize("obliquity_deg", [0.0, 37.0, 90.0, 180.0])
    def test_sky_separation_invariant_under_obliquity(self, obliquity_deg):
        """sqrt(X^2+Y^2) is a rotation-invariant quantity, so sp_orb must
        leave the sky-plane separation from the stellar centre unchanged."""
        t = 1.1
        X0, Y0, Z0 = _sky_pos(t)
        X, Y, Z = _sky_pos(t, sp_orb=np.deg2rad(obliquity_deg))
        sep0 = float(jnp.sqrt(X0 ** 2 + Y0 ** 2))
        sep  = float(jnp.sqrt(X ** 2 + Y ** 2))
        np.testing.assert_allclose(sep, sep0, atol=1e-4)

    # --- Physical sanity: polar spin-orbit angle moves the transit off the equator

    def test_polar_obliquity_swaps_null_axis_at_central_transit(self):
        """
        For an edge-on, b≈0 orbit, sp_orb=0 keeps Y≈0 across the whole
        transit (the chord runs along the projected stellar equator, where
        v_los is maximal at the limbs). sp_orb=π/2 instead keeps X≈0
        (the chord runs along the spin axis instead, crossing every
        latitude but staying near the v_los=0 meridian) -- the geometric
        picture underlying the suppressed classic Rossiter-McLaughlin
        amplitude expected for polar transits.
        """
        for t in np.linspace(-0.3, 0.3, 7):
            X0, Y0, Z0 = _sky_pos(t)
            assert abs(float(Y0)) < 0.15, f"aligned: Y should stay near 0, got {float(Y0):.4f}"

            Xp, Yp, Zp = _sky_pos(t, sp_orb=np.pi / 2.0)
            assert abs(float(Xp)) < 0.15, f"polar: X should stay near 0, got {float(Xp):.4f}"

    # --- Differentiability ---------------------------------------------------

    def test_differentiable_wrt_obliquity(self):
        """jax.grad w.r.t. sp_orb should be finite."""
        def f(sp_orb):
            X, Y, Z = planet_sky_position(
                jnp.float32(0.5), t0=0.0, period=10.0, a_over_rstar=15.0,
                inclination=np.pi / 2.0, ecc=0.0, omega_peri=0.0,
                sp_orb=sp_orb,
            )
            return X + Y + Z
        g = jax.grad(f)(jnp.float32(np.pi / 4.0))
        assert np.isfinite(float(g))


# ===================================================================
# 3.  compute_planet_sky_positions — vectorised
# ===================================================================

class TestComputePlanetSkyPositions:

    _kw = dict(t0=0.0, period=5.0, a_over_rstar=15.0,
               inclination=np.pi / 2.0, ecc=0.0, omega_peri=0.0)

    def test_output_shape(self):
        """Output must be (ntime, 3)."""
        times = jnp.linspace(-0.5, 0.5, 80)
        xyz = compute_planet_sky_positions(times, **self._kw)
        assert xyz.shape == (80, 3)

    def test_each_row_matches_scalar_call(self):
        """Every row of the vectorised result must match the scalar call."""
        times = np.linspace(-0.1, 0.1, 10)
        xyz_v = compute_planet_sky_positions(jnp.asarray(times), **self._kw)
        for i, t in enumerate(times):
            X, Y, Z = planet_sky_position(jnp.float32(t), **self._kw)
            np.testing.assert_allclose(
                np.array(xyz_v[i]),
                [float(X), float(Y), float(Z)],
                atol=1e-4, err_msg=f"Mismatch at index {i} (t={t:.4f})")

    def test_output_finite(self):
        """All (X, Y, Z) values must be finite across an eccentric orbit."""
        times = jnp.linspace(0.0, 5.0, 300)
        xyz = compute_planet_sky_positions(
            times, t0=0.0, period=5.0, a_over_rstar=10.0,
            inclination=np.pi / 2.0, ecc=0.4, omega_peri=np.pi / 4.0,
        )
        assert np.all(np.isfinite(np.array(xyz)))

    def test_mid_transit_row_z_positive(self):
        """The row closest to t=t0 should have Z > 0."""
        t0     = 0.0
        times  = np.linspace(-0.5, 0.5, 100)
        xyz    = compute_planet_sky_positions(jnp.asarray(times), t0=t0, **{
            k: v for k, v in self._kw.items() if k != "t0"
        })
        idx_mid = int(np.argmin(np.abs(times - t0)))
        assert float(xyz[idx_mid, 2]) > 0, \
            f"Z at mid-transit should be > 0, got {float(xyz[idx_mid, 2]):.4f}"


# ===================================================================
# 3b.  compute_multi_planet_sky_positions
# ===================================================================

class TestComputeMultiPlanetSkyPositions:

    _kw2 = dict(
        t0=[0.0, 2.0], period=[5.0, 11.0], a_over_rstar=[15.0, 25.0],
        inclination=[np.pi / 2.0, np.pi / 2.0], ecc=[0.0, 0.0],
        omega_peri=[0.0, 0.0],
    )

    def test_output_shape_nplanet_one(self):
        times = jnp.linspace(-0.5, 0.5, 80)
        xyz = compute_multi_planet_sky_positions(
            times, t0=0.0, period=5.0, a_over_rstar=15.0,
            inclination=np.pi / 2.0,
        )
        assert xyz.shape == (80, 1, 3)

    def test_output_shape_multi(self):
        times = jnp.linspace(-0.5, 0.5, 80)
        xyz = compute_multi_planet_sky_positions(times, **self._kw2)
        assert xyz.shape == (80, 2, 3)

    def test_nplanet_one_matches_compute_planet_sky_positions(self):
        """Regression pin: nplanet=1 must exactly equal the single-planet fn."""
        times = jnp.linspace(-0.2, 0.2, 30)
        xyz_multi = compute_multi_planet_sky_positions(
            times, t0=0.0, period=5.0, a_over_rstar=15.0,
            inclination=np.pi / 2.0, ecc=0.1, omega_peri=0.3,
        )
        xyz_single = compute_planet_sky_positions(
            times, 0.0, 5.0, 15.0, np.pi / 2.0, 0.1, 0.3,
        )
        np.testing.assert_allclose(
            np.array(xyz_multi[:, 0, :]), np.array(xyz_single), atol=1e-5,
        )

    def test_each_planet_matches_independent_scalar_calls(self):
        """Each planet's column must match a separate scalar-orbit call."""
        times = jnp.linspace(-0.2, 0.2, 30)
        xyz = compute_multi_planet_sky_positions(times, **self._kw2)
        for i in range(2):
            expected = compute_planet_sky_positions(
                times,
                self._kw2["t0"][i], self._kw2["period"][i],
                self._kw2["a_over_rstar"][i], self._kw2["inclination"][i],
                self._kw2["ecc"][i], self._kw2["omega_peri"][i],
            )
            np.testing.assert_allclose(
                np.array(xyz[:, i, :]), np.array(expected), atol=1e-5,
                err_msg=f"Mismatch for planet {i}",
            )

    def test_scalar_broadcasts_to_all_planets(self):
        """A scalar param (e.g. period shared by all planets) broadcasts."""
        times = jnp.linspace(-0.2, 0.2, 20)
        xyz = compute_multi_planet_sky_positions(
            times, t0=[0.0, 2.0], period=5.0, a_over_rstar=15.0,
            inclination=np.pi / 2.0,
        )
        expected0 = compute_planet_sky_positions(times, 0.0, 5.0, 15.0, np.pi / 2.0, 0.0, 0.0)
        expected1 = compute_planet_sky_positions(times, 2.0, 5.0, 15.0, np.pi / 2.0, 0.0, 0.0)
        np.testing.assert_allclose(np.array(xyz[:, 0, :]), np.array(expected0), atol=1e-5)
        np.testing.assert_allclose(np.array(xyz[:, 1, :]), np.array(expected1), atol=1e-5)

    def test_size_mismatch_raises(self):
        times = jnp.linspace(-0.2, 0.2, 10)
        with pytest.raises(ValueError):
            compute_multi_planet_sky_positions(
                times, t0=[0.0, 2.0, 4.0], period=[5.0, 11.0],
                a_over_rstar=15.0, inclination=np.pi / 2.0,
            )

    def test_grad_through_one_planets_t0(self):
        """jax.grad through a multi-planet call is finite (vmap-through-grad
        works) -- the underlying Kepler-solver math is untouched and already
        covered by TestOrbitalParamGradientsFD, so this is a smoke test."""
        times = jnp.linspace(-0.2, 0.2, 20)

        def f(t0_0):
            t0 = jnp.stack([t0_0, jnp.asarray(2.0)])
            xyz = compute_multi_planet_sky_positions(
                times, t0=t0, period=jnp.asarray([5.0, 11.0]),
                a_over_rstar=jnp.asarray(15.0), inclination=jnp.asarray(np.pi / 2.0),
            )
            return jnp.sum(xyz[:, 0, :] ** 2)

        g = jax.grad(f)(jnp.asarray(0.0))
        assert np.isfinite(float(g))


# ===================================================================
# 4.  build_transit_model
# ===================================================================

class TestBuildTransitModel:

    _kw = dict(t0=0.0, period=5.0, a_over_rstar=15.0,
               inclination=np.pi / 2.0, k=0.1)

    def test_output_keys_present(self):
        times = np.linspace(-0.2, 0.2, 50)
        tm = build_transit_model(times=times, **self._kw)
        assert "planet_xyz" in tm
        assert "k" in tm

    def test_xyz_shape_matches_times(self):
        times = np.linspace(-0.2, 0.2, 50)
        tm = build_transit_model(times=times, **self._kw)
        assert tm["planet_xyz"].shape == (50, 1, 3)

    def test_k_stored_correctly(self):
        times = np.array([0.0])
        tm = build_transit_model(times=times, **{**self._kw, "k": 0.15})
        assert tm["k"] == pytest.approx(0.15)

    def test_mid_transit_Z_positive(self):
        """At t=t0, Z (column 2) should be positive."""
        tm = build_transit_model(times=np.array([0.0]), **self._kw)
        assert float(tm["planet_xyz"][0, 0, 2]) > 0

    def test_default_circular_orbit(self):
        """Default ecc=0, omega_peri=0 should work without explicit kwargs."""
        times = np.linspace(-0.1, 0.1, 20)
        tm = build_transit_model(times=times, **self._kw)
        assert np.all(np.isfinite(np.array(tm["planet_xyz"])))

    def test_eccentric_orbit(self):
        """Eccentric orbit should not crash and should give finite positions."""
        times = np.linspace(-0.2, 0.2, 20)
        tm = build_transit_model(
            times=times, t0=0.0, period=5.0, a_over_rstar=15.0,
            inclination=np.pi / 2.0, k=0.1, ecc=0.3, omega_peri=np.pi / 2.0,
        )
        assert np.all(np.isfinite(np.array(tm["planet_xyz"])))

    def test_nplanet_key_present_and_correct(self):
        tm = build_transit_model(times=np.array([0.0]), **self._kw)
        assert tm["nplanet"] == 1

    def test_multi_planet_shape(self):
        times = np.linspace(-0.2, 0.2, 50)
        tm = build_transit_model(
            times=times, t0=[0.0, 2.5], period=[5.0, 11.0],
            a_over_rstar=[15.0, 25.0], inclination=[np.pi / 2.0, np.pi / 2.0],
            k=[0.1, 0.05],
        )
        assert tm["planet_xyz"].shape == (50, 2, 3)
        assert tm["nplanet"] == 2


# ===================================================================
# 4b.  Time precision (float32 vs. jax_enable_x64)
#      BJD-scale times can induce signifcant rouding errors when processed
#      in float32. In this scenario, users should manually swap to float64
# ===================================================================

@pytest.fixture
def _x64_enabled():
    """Toggle jax_enable_x64 on for the test, then restore the prior value."""
    prior = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


class TestTimePrecision:

    # A BJD-like epoch: far enough from 0 that float32's 24 mantissa bits
    # leave a rounding error of a large fraction of a day (~0.1-0.25 d here).
    _t0_bjd   = 2_460_123.456789
    _period   = 3.14159265
    _pos_kw   = dict(period=_period, a_over_rstar=15.0, inclination=1.55,
                      ecc=0.0, omega_peri=0.0)
    _kw       = dict(period=_period, a_over_rstar=15.0, inclination=1.55, k=0.1)

    def _bjd_times(self, n=50, half_width=0.1):
        return self._t0_bjd + np.linspace(-half_width, half_width, n)

    # --- No hardcoded downcast: dtype should follow the ambient JAX config ---

    def test_default_mode_stays_float32(self):
        """Without jax_enable_x64, times/positions stay float32 (JAX default)."""
        times = self._bjd_times()
        xyz = compute_planet_sky_positions(times, t0=self._t0_bjd, **self._pos_kw)
        assert xyz.dtype == jnp.float32

    def test_x64_enabled_preserves_float64(self, _x64_enabled):
        """With jax_enable_x64, float64 times must not be silently downcast."""
        times = self._bjd_times()
        xyz = compute_planet_sky_positions(times, t0=self._t0_bjd, **self._pos_kw)
        assert xyz.dtype == jnp.float64

    def test_x64_enabled_resolves_close_times(self, _x64_enabled):
        """
        At BJD scale, float32 collapses closely-spaced times onto the same
        value (ULP ~ 0.25 days there). float64 must keep them distinct.
        """
        times = self._bjd_times(n=50, half_width=0.1)
        tm = build_transit_model(times=times, t0=self._t0_bjd, **self._kw)
        xyz = np.array(tm["planet_xyz"])
        assert len(np.unique(xyz[:, 0, 0])) == len(times)

    def test_x64_build_transit_model_matches_reduced_time_reference(self, _x64_enabled):
        """
        float64 positions at BJD-scale times should match positions computed
        after manually subtracting a reference epoch (the numerically-safe
        approach even in plain float32) -- i.e. x64 removes the need for it.
        """
        times = self._bjd_times()
        tm_absolute = build_transit_model(times=times, t0=self._t0_bjd, **self._kw)

        t_ref = np.floor(times.min())
        tm_reduced = build_transit_model(
            times=times - t_ref, t0=self._t0_bjd - t_ref, **self._kw
        )
        np.testing.assert_allclose(
            np.array(tm_absolute["planet_xyz"]),
            np.array(tm_reduced["planet_xyz"]),
            atol=1e-9,
        )

    # --- Manual reference-epoch reduction ---------------------------------
    # build_transit_model itself does not shift times/t0 (see its
    # docstring) -- build_system does that automatically before calling
    # it; a direct/standalone caller must reduce both themselves.

    def test_raw_bjd_times_still_lose_precision_in_build_transit_model(self):
        """
        Passed straight through, raw BJD-scale times/t0 still produce the
        original float32 cancellation error in build_transit_model itself
        -- exactly why build_system reduces them before ever calling it.
        """
        times = self._bjd_times()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the precision warning is expected here
            tm_f32 = build_transit_model(times=times, t0=self._t0_bjd, **self._kw)
        try:
            jax.config.update("jax_enable_x64", True)
            tm_f64 = build_transit_model(times=times, t0=self._t0_bjd, **self._kw)
        finally:
            jax.config.update("jax_enable_x64", False)
        assert not np.allclose(
            np.array(tm_f32["planet_xyz"]), np.array(tm_f64["planet_xyz"]), atol=1e-2,
        )

    def test_manual_reduction_matches_float64_without_x64(self):
        """
        Manually subtracting a reference epoch from times and t0 before
        calling build_transit_model -- what build_system now does for you
        -- recovers full precision without needing x64.
        """
        times = self._bjd_times()
        t_ref = np.floor(times.min())
        tm_reduced_f32 = build_transit_model(
            times=times - t_ref, t0=self._t0_bjd - t_ref, **self._kw
        )
        try:
            jax.config.update("jax_enable_x64", True)
            tm_f64 = build_transit_model(times=times, t0=self._t0_bjd, **self._kw)
        finally:
            jax.config.update("jax_enable_x64", False)
        np.testing.assert_allclose(
            np.array(tm_reduced_f32["planet_xyz"]), np.array(tm_f64["planet_xyz"]),
            atol=1e-4,
        )

    def test_build_transit_model_traceable_under_jit_over_times(self):
        """
        _warn_if_precision_insufficient must not break jit/vmap over
        `times` itself -- e.g. a standalone planet.py user jitting
        build_transit_model directly. Before the isinstance Tracer guard,
        this raised TracerArrayConversionError.
        """
        times = self._bjd_times()

        def f(times_arg, t0_val):
            tm = build_transit_model(times=times_arg, t0=t0_val, **self._kw)
            return jnp.sum(tm["planet_xyz"])

        val = jax.jit(f)(jnp.asarray(times), self._t0_bjd)
        assert jnp.isfinite(val)

    def test_compute_planet_sky_positions_differentiable_wrt_t0_under_jit(self):
        """
        Mirrors make_lc's dynamic transit-override path: times pre-shifted
        (as core.py now does via model["t_ref"]), t0 traced under jit/grad.
        """
        times = self._bjd_times()
        t_ref = np.floor(times.min())
        times_shifted = jnp.asarray(times - t_ref)

        def f(t0_shifted):
            xyz = compute_planet_sky_positions(
                times_shifted, t0_shifted, **self._pos_kw
            )
            return jnp.sum(xyz)

        jitted = jax.jit(f)
        val = jitted(self._t0_bjd - t_ref)
        g = jax.grad(jitted)(self._t0_bjd - t_ref)
        assert jnp.isfinite(val)
        assert jnp.isfinite(g)

    # --- Precision warning ---------------------------------------------------

    def test_warns_for_bjd_scale_times_without_x64(self):
        """
        build_transit_model doesn't shift times/t0 itself, so raw BJD-scale
        values passed directly to it still trigger the warning -- see
        tests/test_core.py's TestBjdReferenceEpoch for confirmation that
        build_system's automatic reduction avoids this in normal usage.
        """
        times = self._bjd_times()
        with pytest.warns(UserWarning, match="float32 rounding"):
            build_transit_model(times=times, t0=self._t0_bjd, **self._kw)

    def test_no_warning_for_small_times_without_x64(self):
        """Ordinary short-baseline, near-zero times should not trigger it."""
        times = np.linspace(-0.1, 0.1, 50)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_transit_model(times=times, t0=0.0, **self._kw)

    def test_no_warning_for_bjd_scale_times_with_x64(self, _x64_enabled):
        """x64 removes the precision concern, so the warning must not fire."""
        times = self._bjd_times()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_transit_model(times=times, t0=self._t0_bjd, **self._kw)

    def test_no_crash_on_duplicate_timestamps(self):
        """
        All-identical times collapse to a single point under np.unique, which
        used to make the cadence computation raise on an empty np.diff.
        """
        times = np.full(5, self._t0_bjd)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tm = build_transit_model(times=times, t0=self._t0_bjd, **self._kw)
        assert tm["planet_xyz"].shape == (5, 1, 3)


# ===================================================================
# 5.  Unit conversion: density ↔ a/R★
# ===================================================================

class TestDensityConversions:

    # --- Known value: Earth / Sun ----------------------------------------

    def test_earth_orbit_a_over_rstar(self):
        """Solar density + 1-year period should give a/R★ ≈ 215."""
        rho_sun = 1.41          # g cm⁻³
        P_earth = 365.25        # days
        a       = stellar_density_to_a_over_rstar(rho_sun, P_earth)
        assert 200 < a < 230, f"Expected a/R★ ≈ 215 (Earth), got {a:.1f}"

    # --- Round-trip tests ------------------------------------------------

    def test_round_trip_density_to_a(self):
        """ρ → a/R★ → ρ should recover the original density."""
        rho_in  = 1.41
        P       = 5.0
        a       = stellar_density_to_a_over_rstar(rho_in, P)
        rho_out = a_over_rstar_to_stellar_density(a, P)
        assert abs(rho_out - rho_in) / rho_in < 1e-8

    def test_round_trip_a_to_density(self):
        """a/R★ → ρ → a/R★ should recover the original a/R★."""
        a_in = 12.5
        P    = 4.0
        rho  = a_over_rstar_to_stellar_density(a_in, P)
        a_out = stellar_density_to_a_over_rstar(rho, P)
        assert abs(a_out - a_in) / a_in < 1e-8

    # --- Monotonicity (Kepler's 3rd law) ----------------------------------

    def test_longer_period_gives_larger_a(self):
        """P ∝ a^(3/2) → longer period → larger a/R★."""
        rho = 1.0
        a1  = stellar_density_to_a_over_rstar(rho, 1.0)
        a10 = stellar_density_to_a_over_rstar(rho, 10.0)
        assert a10 > a1

    def test_kepler_third_law_exponent(self):
        """a/R★ should scale as P^(2/3) for fixed density."""
        rho   = 1.0
        P1, P2 = 2.0, 8.0
        a1 = stellar_density_to_a_over_rstar(rho, P1)
        a2 = stellar_density_to_a_over_rstar(rho, P2)
        # a ∝ P^(2/3) → a2/a1 = (P2/P1)^(2/3)
        ratio_expected = (P2 / P1) ** (2.0 / 3.0)
        np.testing.assert_allclose(a2 / a1, ratio_expected, rtol=1e-6,
            err_msg="a/R★ should scale as P^(2/3) (Kepler's 3rd law)")

    def test_denser_star_larger_a_for_fixed_period(self):
        """For fixed P, denser star gives larger a/R★ (smaller physical R★)."""
        P    = 3.0
        a_lo = stellar_density_to_a_over_rstar(0.5, P)
        a_hi = stellar_density_to_a_over_rstar(5.0, P)
        assert a_hi > a_lo

    # --- Parametric scan -------------------------------------------------

    @pytest.mark.parametrize("rho,P", [
        (0.1, 1.0),
        (1.0, 5.0),
        (5.0, 0.5),
        (10.0, 365.0),
    ])
    def test_output_positive(self, rho, P):
        """a/R★ must always be positive for positive ρ and P."""
        a = stellar_density_to_a_over_rstar(rho, P)
        assert a > 0, f"a/R★ should be positive (rho={rho}, P={P})"


# ===================================================================
# Gradient sign and FD-agreement tests for planet_sky_position
# and _compute_planet_mask
# ===================================================================

class TestPlanetGradients:
    """
    Two complementary gradient verification strategies:

    1. Analytical comparison (planet_sky_position):
       For a circular edge-on orbit at mid-transit the sky coordinates are
       exact trig functions of inclination:
           Y = a·cos(i)   →   dY/di = −a·sin(i)
           Z = a·sin(i)   →   dZ/di =  a·cos(i)
       JAX autodiff is checked against these closed-form expressions.

    2. Physical sign + calibrated FD (_compute_planet_mask):
       The transit-edge sigmoid has transition width
           softness_transit = 1 / (10 · star_pixel_rad)  [R★ units]
       For spr=50 this is 0.002 R★.  A finite-difference step of h=0.1 in
       planet position (or inclination converted to planet shift) is ~50×
       this width, making FD completely unreliable.  The FD tests here use
       h = 1 % of the transition width where the chord approximation is valid.
    """

    _spr = 50.0

    @property
    def _softness_transit(self):
        return 1.0 / (10.0 * self._spr)

    # ---- Analytical gradient tests for planet_sky_position ----------------

    def test_dY_di_matches_analytical_at_mid_transit(self):
        """
        At mid-transit with circular orbit:  Y = a·cos(i)  →  dY/di = −a·sin(i).
        """
        a   = jnp.float32(15.0)
        inc = jnp.float32(np.deg2rad(87.0))

        dY_jax = float(jax.grad(
            lambda i: planet_sky_position(
                jnp.float32(0.0), 0.0, 10.0, a,
                i, jnp.float32(0.0), jnp.float32(0.0),
            )[1]
        )(inc))

        dY_analytical = -float(a) * float(jnp.sin(inc))
        np.testing.assert_allclose(
            dY_jax, dY_analytical, rtol=5e-3,
            err_msg=f"dY/di: JAX={dY_jax:.5g}, analytical={dY_analytical:.5g}",
        )

    def test_dZ_di_matches_analytical_at_mid_transit(self):
        """
        At mid-transit with circular orbit:  Z = a·sin(i)  →  dZ/di = a·cos(i).
        """
        a   = jnp.float32(15.0)
        inc = jnp.float32(np.deg2rad(87.0))

        dZ_jax = float(jax.grad(
            lambda i: planet_sky_position(
                jnp.float32(0.0), 0.0, 10.0, a,
                i, jnp.float32(0.0), jnp.float32(0.0),
            )[2]
        )(inc))

        dZ_analytical = float(a) * float(jnp.cos(inc))
        np.testing.assert_allclose(
            dZ_jax, dZ_analytical, rtol=5e-3,
            err_msg=f"dZ/di: JAX={dZ_jax:.5g}, analytical={dZ_analytical:.5g}",
        )

    def test_dY_da_matches_analytical_at_mid_transit(self):
        """
        At mid-transit with circular orbit:  Y = a·cos(i)  →  dY/da = cos(i).
        Tests differentiability through the orbital radius calculation.
        """
        a   = jnp.float32(15.0)
        inc = jnp.float32(np.deg2rad(87.0))

        dY_jax = float(jax.grad(
            lambda a_val: planet_sky_position(
                jnp.float32(0.0), 0.0, 10.0, a_val,
                inc, jnp.float32(0.0), jnp.float32(0.0),
            )[1]
        )(a))

        dY_analytical = float(jnp.cos(inc))
        np.testing.assert_allclose(
            dY_jax, dY_analytical, rtol=5e-3,
            err_msg=f"dY/da: JAX={dY_jax:.5g}, analytical={dY_analytical:.5g}",
        )

    # ---- Planet mask gradient sign test -----------------------------------

    def test_planet_mask_large_h_gives_wrong_result(self):
        """
        Documents that h=0.1 R★ — ~50x the transit transition width — yields a
        FD estimate that is unreliable, explaining sign/value mismatches in
        external test suites that use h=0.1 in unconstrained parameter space.
        """
        k_val = jnp.float32(0.1)
        h_large = jnp.float32(0.1)

        assert float(h_large) > 5.0 * self._softness_transit, (
            "Precondition: h_large must exceed 5x softness_transit."
        )

        px = jnp.array([float(k_val) * self._spr], dtype=jnp.float32)
        py = jnp.array([0.0], dtype=jnp.float32)

        def f(k):
            return jnp.sum(_compute_planet_mask(
                px, py, self._spr,
                jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0), k,
            ))

        grad_jax = float(jax.grad(f)(k_val))
        fd_large = float((f(k_val + h_large) - f(k_val - h_large)) / (2.0 * h_large))

        if abs(fd_large) < 1e-30:
            return  # FD is zero; trivially unreliable

        ratio = abs(grad_jax / fd_large)
        sign_flip = float(np.sign(grad_jax)) != float(np.sign(fd_large))
        assert ratio < 0.1 or ratio > 10.0 or sign_flip, (
            f"h=0.1 FD unexpectedly agrees with JAX (ratio={ratio:.3f}). "
            "Transition may be wider than expected — check spr."
        )

    def test_default_softness_is_exact_hard_edge(self):
        """softness=0.0 (the default) reproduces the old boolean threshold."""
        xn = jnp.array([0.05, 0.15], dtype=jnp.float32)  # inside / outside k=0.1
        px = xn * self._spr
        py = jnp.zeros_like(px)
        mask = _compute_planet_mask(
            px, py, self._spr,
            jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.1),
        )
        np.testing.assert_array_equal(np.array(mask), [1.0, 0.0])

    def test_soft_mask_grad_wrt_k_matches_fd(self):
        """
        With softness = softness_transit (the opt-in path used for
        gradient-based transit retrieval), jax.grad through
        _compute_planet_mask must be finite, non-zero, and agree with a
        finite-difference estimate taken at h = 1% of the transition width.
        """
        k_val    = jnp.float32(0.1)
        softness = jnp.float32(self._softness_transit)
        h        = 0.01 * self._softness_transit

        px = jnp.array([float(k_val) * self._spr], dtype=jnp.float32)
        py = jnp.array([0.0], dtype=jnp.float32)

        def f(k):
            return jnp.sum(_compute_planet_mask(
                px, py, self._spr,
                jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0), k, softness,
            ))

        grad_jax = float(jax.grad(f)(k_val))
        fd = float((f(k_val + h) - f(k_val - h)) / (2.0 * h))

        assert np.isfinite(grad_jax) and grad_jax != 0.0
        np.testing.assert_allclose(grad_jax, fd, rtol=0.05)


# ===================================================================
# FD-agreement tests for the six Keplerian orbital parameters
# ===================================================================

class TestOrbitalParamGradientsFD:
    """
    Verify that JAX autodiff agrees with calibrated finite differences for
    five Keplerian orbital parameters: a/R★, period, orbital inclination,
    eccentricity, and argument of periastron.

    Strategy
    --------
    a/R★, period, inclination, ecc, omega_peri:
        Tested through planet_sky_position (pure trig / Kepler solver).
        The output scalar X+Y+Z is smooth in all parameters, so a standard
        h = 0.01 in natural units is safe everywhere.

    Orbit geometry: inc=89° so impact parameter b = a·cos(i) ≈ 0.26 R★
    is non-zero, making gradients w.r.t. inclination and a/R★ non-trivial.
    For period and ecc tests the planet is evaluated at t=0.5 d (off
    mid-transit) so the mean-anomaly derivatives are non-zero.
    """

    _A   = 15.0
    _INC = float(np.deg2rad(89.0))   # b ≈ 0.26 R★
    _P   = 5.0
    _ECC = 0.0
    _OMG = 0.0
    _T0  = 0.0

    def _xyz(self, t=0.0, **kw):
        """planet_sky_position with class defaults, keyword overrides."""
        p = dict(t0=self._T0, period=self._P, a_over_rstar=self._A,
                 inclination=self._INC, ecc=self._ECC, omega_peri=self._OMG)
        p.update(kw)
        return planet_sky_position(jnp.float32(t), **p)

    def _scalar(self, xyz):
        X, Y, Z = xyz
        return X + Y + Z

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
        """Assert JAX and FD agree to rtol for smooth (non-sigmoid) parameters."""
        assert np.isfinite(fd) and abs(fd) > 0, f"'{name}' FD zero or non-finite"
        np.testing.assert_allclose(
            jax_g, fd, rtol=rtol,
            err_msg=f"'{name}' JAX ({jax_g:.8g}) vs FD ({fd:.8g})",
        )

    # ---- a_over_rstar -------------------------------------------------------

    def test_a_gradient_finite_and_nonzero(self):
        """d(X+Y+Z)/d(a) must be finite and non-zero (Y = r·sin(ω+f)·cos i ∝ a)."""
        g = float(jax.grad(
            lambda a: self._scalar(self._xyz(a_over_rstar=a))
        )(jnp.float32(self._A)))
        assert np.isfinite(g), f"d/d(a) non-finite: {g}"
        assert abs(g) > 0,     "d/d(a) is zero"

    def test_a_gradient_fd_agreement(self):
        """FD at h=0.01 R★; smooth trig → rtol=1e-3 (observed rel_err < 5e-5)."""
        h = 0.01
        jax_g = float(jax.grad(
            lambda a: self._scalar(self._xyz(a_over_rstar=a))
        )(jnp.float32(self._A)))
        fd = self._fd(
            lambda a: self._scalar(self._xyz(a_over_rstar=float(a))),
            self._A, h,
        )
        self._check_tight("a_over_rstar", jax_g, fd, rtol=1e-3)

    # ---- period -------------------------------------------------------------

    def test_period_gradient_finite_and_nonzero(self):
        """
        At t=0.5 d (off mid-transit) d(X+Y+Z)/d(period) is non-zero.
        At t=t0 the mean anomaly M=0 and dM/dP = −(2π/P²)·(t−t_peri) = 0,
        so the off-transit evaluation is essential.
        """
        g = float(jax.grad(
            lambda P: self._scalar(self._xyz(t=0.5, period=P))
        )(jnp.float32(self._P)))
        assert np.isfinite(g), f"d/d(period) non-finite: {g}"
        assert abs(g) > 0,     "d/d(period) is zero at t=0.5 d"

    def test_period_gradient_fd_agreement(self):
        """FD at h=0.01 d; smooth trig → rtol=1e-3 (observed rel_err < 4e-4)."""
        h = 0.01
        jax_g = float(jax.grad(
            lambda P: self._scalar(self._xyz(t=0.5, period=P))
        )(jnp.float32(self._P)))
        fd = self._fd(
            lambda P: self._scalar(self._xyz(t=0.5, period=float(P))),
            self._P, h,
        )
        self._check_tight("period", jax_g, fd, rtol=1e-3)

    # ---- orbital inclination ------------------------------------------------

    def test_inclination_gradient_finite_and_nonzero(self):
        """
        Y = r·sin(ω+f)·cos(i), so d(Y)/d(i) = −r·sin(ω+f)·sin(i) ≠ 0
        at i=89°.  The non-zero impact parameter makes this non-trivial.
        """
        g = float(jax.grad(
            lambda i: self._scalar(self._xyz(inclination=i))
        )(jnp.float32(self._INC)))
        assert np.isfinite(g), f"d/d(inc) non-finite: {g}"
        assert abs(g) > 0,     "d/d(inc) is zero at inc=89°"

    def test_inclination_gradient_fd_agreement(self):
        """FD at h=1e-4 rad; smooth trig → rtol=1e-3 (observed rel_err < 3e-4)."""
        h = 1e-4
        jax_g = float(jax.grad(
            lambda i: self._scalar(self._xyz(inclination=i))
        )(jnp.float32(self._INC)))
        fd = self._fd(
            lambda i: self._scalar(self._xyz(inclination=float(i))),
            self._INC, h,
        )
        self._check_tight("inclination", jax_g, fd, rtol=1e-3)

    # ---- eccentricity -------------------------------------------------------

    def test_ecc_gradient_finite_and_nonzero(self):
        """
        d(X+Y+Z)/d(ecc) at ecc=0.1, t=0.5 d (off mid-transit).
        The orbital radius r = a(1−e²)/(1+e·cos f) depends on ecc through
        both the Kepler solver and the radius formula.
        """
        g = float(jax.grad(
            lambda e: self._scalar(self._xyz(t=0.5, ecc=e))
        )(jnp.float32(0.1)))
        assert np.isfinite(g), f"d/d(ecc) non-finite: {g}"
        assert abs(g) > 0,     "d/d(ecc) is zero"

    def test_ecc_gradient_fd_agreement(self):
        """FD at h=0.01; smooth Kepler solver → rtol=1e-3 (observed rel_err < 2e-4)."""
        h = 0.01
        jax_g = float(jax.grad(
            lambda e: self._scalar(self._xyz(t=0.5, ecc=e))
        )(jnp.float32(0.1)))
        fd = self._fd(
            lambda e: self._scalar(self._xyz(t=0.5, ecc=float(e))),
            0.1, h,
        )
        self._check_tight("ecc", jax_g, fd, rtol=1e-3)

    # ---- argument of periastron --------------------------------------------

    def test_omega_gradient_finite_and_nonzero(self):
        """
        d(X+Y+Z)/d(omega_peri) at ω=π/4, ecc=0.1, t=0.5 d.
        For a circular orbit (ecc=0) ω cancels in the expression ω+f, so
        ecc=0.1 is required to make the projected position depend on ω.
        """
        omega0 = jnp.float32(np.pi / 4.0)
        g = float(jax.grad(
            lambda w: self._scalar(self._xyz(t=0.5, ecc=0.1, omega_peri=w))
        )(omega0))
        assert np.isfinite(g), f"d/d(omega_peri) non-finite: {g}"
        assert abs(g) > 0,     "d/d(omega_peri) is zero at ecc=0.1"

    def test_omega_gradient_fd_agreement(self):
        """FD at h=0.01 rad; smooth trig → rtol=1e-3 (observed rel_err < 6e-5)."""
        omega0 = float(np.pi / 4.0)
        h = 0.01
        jax_g = float(jax.grad(
            lambda w: self._scalar(self._xyz(t=0.5, ecc=0.1, omega_peri=w))
        )(jnp.float32(omega0)))
        fd = self._fd(
            lambda w: self._scalar(self._xyz(t=0.5, ecc=0.1, omega_peri=float(w))),
            omega0, h,
        )
        self._check_tight("omega_peri", jax_g, fd, rtol=1e-3)