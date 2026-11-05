"""Sampled Cartesian line and orientation evidence in the workpiece frame."""

import numpy as np
from rotation_targets import interpolate_rotations, rotation_errors


class ProcessGeometryError(RuntimeError):
    """The sampled process poses cannot define a nonzero Cartesian line."""


def assess_process_geometry(first, last, actual):
    """Return physical departures and ordering evidence without a process tolerance."""
    first = np.asarray(first, dtype=float)
    last = np.asarray(last, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if (
        first.shape != (4, 4)
        or last.shape != (4, 4)
        or actual.ndim != 3
        or actual.shape[1:] != (4, 4)
        or len(actual) < 2
        or not np.isfinite(first).all()
        or not np.isfinite(last).all()
        or not np.isfinite(actual).all()
    ):
        raise ProcessGeometryError("finite endpoint poses and sampled 4x4 transforms required")
    axis = last[:3, 3] - first[:3, 3]
    squared = float(axis @ axis)
    if squared == 0:
        raise ProcessGeometryError("zero-length Cartesian process line is unsupported")
    progress = (actual[:, :3, 3] - first[:3, 3]) @ axis / squared
    clipped = np.clip(progress, 0, 1)
    reference = first[:3, 3] + clipped[:, None] * axis
    distance = np.linalg.norm(actual[:, :3, 3] - reference, axis=1)
    rotations = interpolate_rotations(first[:3, :3], last[:3, :3], clipped)
    angle = rotation_errors(rotations, actual[:, :3, :3])
    backward = float(np.max(np.maximum.accumulate(progress) - progress))
    return {
        "max_position_departure_m": float(distance.max()),
        "max_orientation_departure_rad": float(angle.max()),
        "progress_min": float(progress.min()),
        "progress_max": float(progress.max()),
        "max_backward_progress": backward,
        "max_backward_distance_m": backward * float(np.sqrt(squared)),
    }
