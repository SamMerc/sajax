"""
tests/test_doppler_axis.py — Orientation of the rotational Doppler field.

The stellar spin axis is the body-frame y-axis (``geometry.rotate_active_region``
applies ``rotation_matrix_y``; ``make_lc`` places latitude on y). For a rigid
rotator with sky-frame angular velocity ``omega = w (0, sin i, cos i)``, the
line-of-sight component of ``v = omega x r`` is::

    v_z = -w * sin(i) * x

so the radial velocity varies along the sky **x** axis, is independent of both
y and the pixel's depth z, and vanishes pole-on.

These tests pin down the *orientation and sign* of that field, which the
disc-integrated tests elsewhere cannot see: the stellar disc is circularly
symmetric, so the rotationally broadened profile of the quiet star is identical
whether the velocity is keyed on x or on y. The signature only appears in the
resolved star map and in wavelength-resolved active-region signals.

Run with:
    pytest tests/test_doppler_axis.py
"""

import numpy as np
import jax.numpy as jnp

from sajax import quick_lc
from sajax.geometry import rotate_active_region

C_KMS = 299_792.458  # speed of light [km/s]

# A rotation fast enough that the limb shift dwarfs the line width, so a pixel
# on the approaching/receding limb moves the line clean out of the channel.
_VE   = 300.0   # equatorial velocity [km/s]  -> limb shift ~5 A at 5005 A
_LAM0 = 5005.0  # line centre [A]
_SIG  = 0.3     # line 1-sigma width [A]  << limb shift


def _line_spectrum(wl, depth=0.9, lam0=_LAM0, sigma=_SIG):
    """Flat continuum at 1.0 with a single narrow Gaussian absorption line."""
    return 1.0 - depth * np.exp(-0.5 * ((wl - lam0) / sigma) ** 2)


def _quiet_star_map(grid_size, wavelength_target, ve=_VE, inc_star=90.0):
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
        ar_lat=np.array([0.0]), ar_long=np.array([0.0]),
        ar_size=np.array([1.0]), ar_smoothness=np.array([1.0]),
        times=np.array([0.0]), P_rot=1.0,
        stellar_grid_size=grid_size, ve=ve, ld_coeffs=[0.0, 0.0],
        inc_star=inc_star, plot_map_wavelength=wavelength_target,
    )
    # star_maps is (ntimes, n, n) reshaped from meshgrid(..., indexing='xy'):
    # axis 0 is the y (row) coordinate, axis 1 is x (column), both running
    # from -n//2 to +n//2, so index n//2 is the disc centre.
    return np.asarray(maps[0])


# ===================================================================
# Test 1 -- the Doppler field varies along x, not along y
# ===================================================================

class TestDopplerFieldAxis:

    def test_line_core_map_varies_along_x_and_is_flat_along_y(self):
        """Read the map at the rest wavelength of a narrow line.

        Pixels near x = +/-R are Doppler-shifted out of the line and stay
        bright; pixels along the y axis (x = 0) have zero radial velocity and
        must all sit at the same line-core depth as the disc centre.
        """
        G = 20
        m = _quiet_star_map(G, wavelength_target=_LAM0)
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
        m = _quiet_star_map(G, wavelength_target=_LAM0)
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


# ===================================================================
# Test 2 -- sign: +x is the receding (redshifted) limb
# ===================================================================

class TestDopplerFieldSign:

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
        delta = _LAM0 * (0.5 * _VE / C_KMS)   # ~2.5 A
        m = _quiet_star_map(G, wavelength_target=_LAM0 + delta)
        c = m.shape[0] // 2
        d = int(round(0.5 * G))

        assert m[c, c + d] < m[c, c - d] - 0.05, (
            f"at lambda0+{delta:.2f} A the dark pixels must sit on the "
            f"receding +x side: map[+0.5R]={m[c, c + d]:.5f}, "
            f"map[-0.5R]={m[c, c - d]:.5f}"
        )
        # The redshifted column should be near the line core.
        assert m[c, c + d] < 0.2


# ===================================================================
# Test 3 -- observable level: an equatorial spot's line asymmetry
# ===================================================================

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
            ar_lat=np.array([0.0]), ar_long=np.array([0.0]),
            ar_size=np.array([20.0]), ar_smoothness=np.array([20.0]),
            times=times, P_rot=1.0,
            stellar_grid_size=self._G, ve=_VE, ld_coeffs=[0.0, 0.0],
            inc_star=90.0,
        )
        lc = np.asarray(lc)                # (nphase, nwave)

        delta   = _LAM0 * (0.5 * _VE / C_KMS)
        i_blue  = int(np.argmin(np.abs(wl - (_LAM0 - delta))))
        i_red   = int(np.argmin(np.abs(wl - (_LAM0 + delta))))
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
