"""Orientation targets use the same short arc and angular error as the cell."""

import numpy as np
from rotation_targets import interpolate_rotations, rotation_errors

# Binary64 rotation matrices round-trip through Eigen at this absolute scale.
MATRIX_ROUNDOFF = 1e-12


def _z_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_interpolation_tracks_short_arc_across_pi_and_matches_endpoints():
    start = _z_rotation(np.deg2rad(179.0))
    end = _z_rotation(np.deg2rad(-179.0))
    matrices = interpolate_rotations(start, end, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(matrices[0], start, atol=MATRIX_ROUNDOFF, rtol=0)
    np.testing.assert_allclose(matrices[1], _z_rotation(np.pi), atol=MATRIX_ROUNDOFF, rtol=0)
    np.testing.assert_allclose(matrices[2], end, atol=MATRIX_ROUNDOFF, rtol=0)


def test_errors_measure_geodesic_angle_for_each_pair():
    expected = np.stack([np.eye(3), _z_rotation(np.pi / 4)])
    actual = np.stack([_z_rotation(np.pi / 6), _z_rotation(-np.pi / 4)])
    np.testing.assert_allclose(
        rotation_errors(expected, actual), [np.pi / 6, np.pi / 2], atol=MATRIX_ROUNDOFF, rtol=0
    )


def test_interpolation_handles_rotations_about_different_axes():
    start = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    end = _z_rotation(np.pi / 2)
    expected_midpoint = np.array([[2, -2, 1], [2, 1, -2], [1, 2, 2]]) / 3.0
    midpoint = interpolate_rotations(start, end, np.array([0.5]))[0]
    np.testing.assert_allclose(midpoint, expected_midpoint, atol=MATRIX_ROUNDOFF, rtol=0)
