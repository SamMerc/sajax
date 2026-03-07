"""
geometry.py — JAX rotation matrices and coordinate transforms.

Replaces the ``astropy.coordinates.matrix_utilities.rotation_matrix``
dependency from the original SAGE code with pure JAX, making all
geometry operations differentiable and JIT-compilable.

Geometry convention (identical to original SAGE):
    - Observer is at z → +∞.  The plane of sky is X-Y.
    - The stellar rotation axis is the y-axis.
    - inc_star = 90°  →  equator-on  (observer sees the equator).
    - inc_star =  0°  →  pole-on     (observer looks at the north pole).
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Primitive rotation matrices
# ---------------------------------------------------------------------------

def rotation_matrix_y(angle_rad: float) -> jnp.ndarray:
    """
    3x3 active rotation matrix around the y-axis.

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.

    Returns
    -------
    jnp.ndarray, shape (3, 3)
    """
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    return jnp.array([
        [ c,  0.,  s],
        [ 0., 1.,  0.],
        [-s,  0.,  c],
    ])


def rotation_matrix_x(angle_rad: float) -> jnp.ndarray:
    """
    3x3 active rotation matrix around the x-axis.

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.

    Returns
    -------
    jnp.ndarray, shape (3, 3)
    """
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    return jnp.array([
        [1.,  0.,  0.],
        [0.,  c,  -s],
        [0.,  s,   c],
    ])


# ---------------------------------------------------------------------------
# Combined stellar-rotation + inclination transform
# ---------------------------------------------------------------------------

def rotate_active_region(
    cart: jnp.ndarray,
    phase_deg: float,
    inc_deg: float,
) -> jnp.ndarray:
    """
    Apply stellar rotation (y-axis) then stellar inclination (x-axis)
    to a Cartesian coordinate vector of an active region.

    This replaces the two-step ``stellar_rotation`` + ``stellar_inc``
    functions from the original SAGE code.

    Parameters
    ----------
    cart : jnp.ndarray, shape (3,)
        [x, y, z] pixel-coordinate position of the active region on the
        stellar sphere.
    phase_deg : float
        Rotational phase in degrees.
    inc_deg : float
        Stellar inclination in degrees (90 = equator-on, 0 = pole-on).

    Returns
    -------
    jnp.ndarray, shape (3,)
        Rotated [x, y, z] coordinates.
    """
    phase_rad = jnp.deg2rad(phase_deg)
    # Original SAGE applies (90 - inc_star) as the x-axis tilt
    tilt_rad = jnp.deg2rad(90.0 - inc_deg)

    # .T because the original used rotation_matrix(...).T — passive convention
    R_rot = rotation_matrix_y(phase_rad).T
    R_inc = rotation_matrix_x(tilt_rad).T

    return R_inc @ R_rot @ cart