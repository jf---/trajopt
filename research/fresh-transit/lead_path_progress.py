"""Ordered local-tangent coordinates for a sampled generated lead target."""

import numpy as np


class LeadPathProgressError(RuntimeError):
    """The generated path cannot define ordered local tangent coordinates."""


def sampled_path_progress(points_m, velocities_m_s):
    """Return unit tangents and sampled polyline arc lengths in the workpiece frame."""
    points = np.asarray(points_m, dtype=float)
    velocities = np.asarray(velocities_m_s, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 2
        or velocities.shape != points.shape
        or not np.isfinite(points).all()
        or not np.isfinite(velocities).all()
    ):
        raise LeadPathProgressError("finite workpiece path points and velocities are required")
    speed = np.linalg.norm(velocities, axis=1)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(speed <= 0) or np.any(steps <= 0):
        raise LeadPathProgressError("zero target velocity or repeated path samples are unsupported")
    return velocities / speed[:, None], np.r_[0.0, np.cumsum(steps)]
