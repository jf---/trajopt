"""Modeled derivative peaks and exact cubic-segment integrated jerk."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from cubic_velocity_extrema import cubic_velocity_extrema


class WeldBlockDynamicsError(RuntimeError):
    """The cubic block or its velocity scales are invalid."""


def block_dynamics(curves: Sequence[Any], speed_scales: Any) -> dict[str, float | list[float]]:
    """Compute per-axis peaks and jerk integrals using exact cubic pieces.

    Clock is modeled seconds. Track uses metres and each arm axis radians;
    speed scales use their corresponding units per second. This does not
    measure controller timing or establish sourced acceleration/jerk limits.
    """
    scales = np.asarray(speed_scales, dtype=float)
    if not curves or scales.ndim != 1 or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise WeldBlockDynamicsError("curves and positive finite axis speed scales required")
    peak_velocity = np.zeros(len(scales))
    peak_acceleration = np.zeros(len(scales))
    peak_jerk = np.zeros(len(scales))
    integrated = np.zeros(len(scales))
    duration = 0.0
    for curve in curves:
        times = np.asarray(curve.x, dtype=float)
        if curve.c.ndim != 3 or curve.c.shape[2] != len(scales) or np.any(np.diff(times) <= 0):
            raise WeldBlockDynamicsError("each cubic stage must match the axis scales")
        extrema = cubic_velocity_extrema(curve)
        peak_velocity = np.maximum(
            peak_velocity, [max(abs(v["minimum"]), abs(v["maximum"])) for v in extrema]
        )
        peak_acceleration = np.maximum(peak_acceleration, np.max(np.abs(curve(times, 2)), axis=0))
        widths = np.diff(times)
        jerk = np.asarray(curve((times[:-1] + times[1:]) / 2, 3))
        peak_jerk = np.maximum(peak_jerk, np.max(np.abs(jerk), axis=0))
        integrated += np.sum(widths[:, None] * jerk**2, axis=0)
        duration += float(times[-1] - times[0])
    return {
        "duration_s": duration,
        "peak_velocity": peak_velocity.tolist(),
        "peak_acceleration": peak_acceleration.tolist(),
        "peak_jerk": peak_jerk.tolist(),
        "integrated_jerk_squared": integrated.tolist(),
        "integrated_speed_normalized_jerk_squared": (integrated / scales**2).tolist(),
    }
