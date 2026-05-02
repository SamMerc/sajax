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
    _compute_planet_mask,
)
from sajax import build_stellar_grid


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

    def test_planet_mask_grad_k_positive_at_boundary(self):
        """
        A pixel at the transit edge (distance from planet = k) should have a
        positive gradient w.r.t. k: enlarging the planet captures that pixel.
        d(mask_sum)/dk > 0.
        """
        k_val = jnp.float32(0.1)
        # Pixel at the boundary: xn = k (in R★), so pixel coordinate = k * spr
        px = jnp.array([float(k_val) * self._spr], dtype=jnp.float32)
        py = jnp.array([0.0], dtype=jnp.float32)

        def f(k):
            return jnp.sum(_compute_planet_mask(
                px, py, self._spr,
                jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0), k,
            ))

        grad = float(jax.grad(f)(k_val))
        assert grad > 0, (
            f"d(mask)/dk = {grad:.4g} at transit boundary; "
            "expected > 0 (larger planet covers boundary pixel)"
        )

    def test_planet_mask_grad_k_fd_agreement_with_small_h(self):
        """
        JAX autodiff agrees with central FD at h = 1 % of the transit-edge
        transition width (softness_transit = 1 / (10 · spr) ≈ 0.002 R★).

        Using h=0.1 would be ~50× the transition width — the same regime that
        produces sign-flipped FD estimates in external test suites.
        """
        k_val = jnp.float32(0.1)
        h = jnp.float32(0.01 * self._softness_transit)

        px = jnp.array([float(k_val) * self._spr], dtype=jnp.float32)
        py = jnp.array([0.0], dtype=jnp.float32)

        def f(k):
            return jnp.sum(_compute_planet_mask(
                px, py, self._spr,
                jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0), k,
            ))

        grad_jax = float(jax.grad(f)(k_val))
        fd = float((f(k_val + h) - f(k_val - h)) / (2.0 * h))

        assert abs(fd) > 0.1 * abs(grad_jax), (
            f"FD is numerically degenerate (fd={fd:.3g}, jax={grad_jax:.3g}). "
            f"h={float(h):.2e} may be too small for float32 at spr={self._spr}."
        )
        ratio = grad_jax / fd
        assert 0.5 <= ratio <= 2.0, (
            f"JAX ({grad_jax:.4g}) disagrees with FD ({fd:.4g}), ratio={ratio:.3f}. "
            f"softness_transit={self._softness_transit:.4g}, h={float(h):.4g}."
        )

    def test_planet_mask_large_h_gives_wrong_result(self):
        """
        Documents that h=0.1 R★ — ~50× the transit transition width — yields a
        FD estimate that is unreliable, explaining sign/value mismatches in
        external test suites that use h=0.1 in unconstrained parameter space.
        """
        k_val = jnp.float32(0.1)
        h_large = jnp.float32(0.1)

        assert float(h_large) > 5.0 * self._softness_transit, (
            "Precondition: h_large must exceed 5× softness_transit."
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


# ===================================================================
# FD-agreement tests for the six Keplerian orbital parameters
# ===================================================================

class TestOrbitalParamGradientsFD:
    """
    Verify that JAX autodiff agrees with calibrated finite differences for
    all six Keplerian orbital parameters: a/R★, period, k, orbital
    inclination, eccentricity, and argument of periastron.

    Strategy
    --------
    a/R★, period, inclination, ecc, omega_peri:
        Tested through planet_sky_position (pure trig / Kepler solver —
        no sigmoid).  The output scalar X+Y+Z is smooth in all parameters,
        so a standard h = 0.01 in natural units is safe everywhere.

    k (planet radius ratio):
        Tested through _compute_planet_mask, where the transit-edge sigmoid
        has softness = 1/(10·spr).  Uses the same calibrated h as
        TestPlanetGradients.

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
    _K   = 0.1
    _SPR = 50.0

    @property
    def _softness(self):
        return 1.0 / (10.0 * self._SPR)

    @pytest.fixture(autouse=True)
    def _grid(self):
        g = build_stellar_grid(int(self._SPR), 0.0)
        self._x   = jnp.asarray(g["x"])
        self._y   = jnp.asarray(g["y"])
        self._spr = float(g["star_pixel_rad"])

    def _xyz(self, t=0.0, **kw):
        """planet_sky_position with class defaults, keyword overrides."""
        p = dict(t0=self._T0, period=self._P, a_over_rstar=self._A,
                 inclination=self._INC, ecc=self._ECC, omega_peri=self._OMG)
        p.update(kw)
        return planet_sky_position(jnp.float32(t), **p)

    def _scalar(self, xyz):
        X, Y, Z = xyz
        return X + Y + Z

    def _mask_sum(self, X, Y, Z, k=None):
        k = jnp.float32(self._K) if k is None else k
        return jnp.sum(_compute_planet_mask(
            self._x, self._y, self._spr, X, Y, Z, k,
        ))

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

    # ---- a_over_rstar -------------------------------------------------------

    def test_a_gradient_finite_and_nonzero(self):
        """d(X+Y+Z)/d(a) must be finite and non-zero (Y = r·sin(ω+f)·cos i ∝ a)."""
        g = float(jax.grad(
            lambda a: self._scalar(self._xyz(a_over_rstar=a))
        )(jnp.float32(self._A)))
        assert np.isfinite(g), f"d/d(a) non-finite: {g}"
        assert abs(g) > 0,     "d/d(a) is zero"

    def test_a_gradient_fd_agreement(self):
        """FD at h=0.01 R★; planet_sky_position is smooth and linear in a."""
        h = 0.01
        jax_g = float(jax.grad(
            lambda a: self._scalar(self._xyz(a_over_rstar=a))
        )(jnp.float32(self._A)))
        fd = self._fd(
            lambda a: self._scalar(self._xyz(a_over_rstar=float(a))),
            self._A, h,
        )
        self._check("a_over_rstar", jax_g, fd, h)

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
        """FD at h=0.01 d; planet_sky_position is smooth in period."""
        h = 0.01
        jax_g = float(jax.grad(
            lambda P: self._scalar(self._xyz(t=0.5, period=P))
        )(jnp.float32(self._P)))
        fd = self._fd(
            lambda P: self._scalar(self._xyz(t=0.5, period=float(P))),
            self._P, h,
        )
        self._check("period", jax_g, fd, h)

    # ---- k (planet radius ratio) --------------------------------------------

    def test_k_gradient_positive_at_transit_boundary(self):
        """
        At mid-transit a pixel at distance k from the planet centre is on
        the boundary; enlarging the planet must increase its mask value:
        d(mask_sum)/dk > 0.
        """
        X, Y, Z = self._xyz()
        k0 = jnp.float32(self._K)
        g  = float(jax.grad(lambda k: self._mask_sum(X, Y, Z, k=k))(k0))
        assert g > 0, f"d(mask)/dk = {g:.4g}; expected > 0"

    def test_k_gradient_fd_agreement(self):
        """
        FD at h = 0.01·softness_transit = 1/(10·spr)·0.01 ≈ 2×10⁻⁴ R★,
        keeping the step well inside the sigmoid linear regime.
        """
        X, Y, Z = self._xyz()
        k0 = jnp.float32(self._K)
        h  = jnp.float32(0.01 * self._softness)

        jax_g = float(jax.grad(lambda k: self._mask_sum(X, Y, Z, k=k))(k0))
        fd    = self._fd(
            lambda k: float(self._mask_sum(X, Y, Z, k=jnp.float32(k))),
            float(k0), float(h),
        )
        self._check("k", jax_g, fd, float(h))

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
        """FD at h=1×10⁻⁴ rad; planet_sky_position is smooth in inclination."""
        h = 1e-4
        jax_g = float(jax.grad(
            lambda i: self._scalar(self._xyz(inclination=i))
        )(jnp.float32(self._INC)))
        fd = self._fd(
            lambda i: self._scalar(self._xyz(inclination=float(i))),
            self._INC, h,
        )
        self._check("inclination", jax_g, fd, h)

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
        """FD at h=0.01; planet_sky_position is smooth in eccentricity."""
        h = 0.01
        jax_g = float(jax.grad(
            lambda e: self._scalar(self._xyz(t=0.5, ecc=e))
        )(jnp.float32(0.1)))
        fd = self._fd(
            lambda e: self._scalar(self._xyz(t=0.5, ecc=float(e))),
            0.1, h,
        )
        self._check("ecc", jax_g, fd, h)

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
        """FD at h=0.01 rad; planet_sky_position is smooth in omega_peri."""
        omega0 = float(np.pi / 4.0)
        h = 0.01
        jax_g = float(jax.grad(
            lambda w: self._scalar(self._xyz(t=0.5, ecc=0.1, omega_peri=w))
        )(jnp.float32(omega0)))
        fd = self._fd(
            lambda w: self._scalar(self._xyz(t=0.5, ecc=0.1, omega_peri=float(w))),
            omega0, h,
        )
        self._check("omega_peri", jax_g, fd, h)