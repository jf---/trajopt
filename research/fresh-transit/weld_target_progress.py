"""Measure sampled actual motion against an ordered 3D process target."""

from typing import Any

import numpy as np


class WeldTargetProgressError(RuntimeError):
    """Actual, target or modeled sampling times are incompatible."""


def measure_target_progress(actual_m: Any, target_m: Any, times_s: Any) -> dict[str, float | int]:
    """Compare sampled points in one workpiece frame and modeled seconds.

    Nearest-target arc length is discretized at the supplied target samples;
    signed local-tangent displacement uses the same sampled tangent. Neither is
    a continuous-order proof or an imposed manufacturing tolerance.
    """
    actual, target, times = (
        np.asarray(value, dtype=float) for value in (actual_m, target_m, times_s)
    )
    if (
        actual.ndim != 2
        or actual.shape[1:] != (3,)
        or target.shape != actual.shape
        or times.shape != (len(actual),)
        or len(actual) < 2
        or not np.isfinite(actual).all()
        or not np.isfinite(target).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise WeldTargetProgressError(
            "finite matching 3D samples and increasing modeled times required"
        )
    nearest_list: list[int] = []
    for chunk in np.array_split(actual, min(16, len(actual))):
        nearest_list.extend(
            np.argmin(
                np.sum((chunk[:, None, :] - target[None, :, :]) ** 2, axis=2), axis=1
            ).tolist()
        )
    nearest = np.asarray(nearest_list, dtype=int)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(target, axis=0), axis=1))]
    differences = np.diff(distance[nearest])
    tangent = np.gradient(target, times, axis=0)[nearest[:-1]]
    norm = np.linalg.norm(tangent, axis=1)
    signed = np.divide(
        np.sum(np.diff(actual, axis=0) * tangent, axis=1),
        norm,
        out=np.zeros(len(actual) - 1),
        where=norm > 0,
    )
    return {
        "nearest_target_max_backward_step_m": float(max(0.0, -differences.min())),
        "nearest_target_total_backward_m": float(np.maximum(-differences, 0).sum()),
        "nearest_target_backward_steps": int(np.sum(differences < 0)),
        "actual_step_along_local_target_tangent_min_m": float(signed.min()),
        "actual_local_tangent_backward_total_m": float(np.maximum(-signed, 0).sum()),
        "max_distance_to_sampled_3d_target_m": float(
            np.max(np.linalg.norm(actual - target[nearest], axis=1))
        ),
        "sampling": len(actual),
    }
