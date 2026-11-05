"""Eigen-backed interpolation and errors for sampled process orientations."""

import numpy as np

from tesseract_qt_py.core.math_utils import (
    rotation_geodesic_angle,
    rotation_matrix_to_quat_wxyz,
    slerp_rotation_wxyz,
)


class RotationTargetsError(RuntimeError):
    """Sampled process orientations do not form matching ordered arrays."""


def interpolate_rotations(start, end, progress):
    """Return shortest-arc rotation matrices at dimensionless progress samples."""
    start_quaternion = rotation_matrix_to_quat_wxyz(start)
    end_quaternion = rotation_matrix_to_quat_wxyz(end)
    values = np.asarray(progress, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
    ):
        raise RotationTargetsError("orientation progress must be finite and between zero and one")
    return np.asarray(
        [slerp_rotation_wxyz(start_quaternion, end_quaternion, float(t)) for t in values]
    ).reshape((-1, 3, 3))


def rotation_errors(expected, actual):
    """Return per-sample shortest rotation errors in radians."""
    reference = np.asarray(expected, dtype=float)
    observed = np.asarray(actual, dtype=float)
    if reference.ndim != 3 or reference.shape[1:] != (3, 3) or observed.shape != reference.shape:
        raise RotationTargetsError("expected and actual rotation arrays must have matching shape")
    return np.asarray(
        [float(rotation_geodesic_angle(a, b)) for a, b in zip(reference, observed, strict=True)]
    )
