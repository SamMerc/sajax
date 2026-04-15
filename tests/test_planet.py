"""
tests/test_planet.py — Tests for the planet.py orbital module.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sajax.planet import (
    _kepler,
    planet_sky_position,
    compute_planet_sky_positions,
    build_transit_model,
    stellar_density_to_a_over_rstar,
    a_over_rstar_to_stellar_density,
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
        assert tm["planet_xyz"].shape == (50, 3)

    def test_k_stored_correctly(self):
        times = np.array([0.0])
        tm = build_transit_model(times=times, **{**self._kw, "k": 0.15})
        assert tm["k"] == pytest.approx(0.15)

    def test_mid_transit_Z_positive(self):
        """At t=t0, Z (column 2) should be positive."""
        tm = build_transit_model(times=np.array([0.0]), **self._kw)
        assert float(tm["planet_xyz"][0, 2]) > 0

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