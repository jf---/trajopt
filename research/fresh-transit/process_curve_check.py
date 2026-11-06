"""Independent finite-sample process-curve and numerical position checks."""

from typing import Any

import numpy as np
from cubic_position_extrema import cubic_position_extrema
from fresh_validation import CurveLimitError, inspect_sampled_curve
from process_curve_representation import build_process_curve
from process_geometry import assess_process_geometry
from rotation_targets import interpolate_rotations, rotation_errors

# Match the independent sampling density used for native transit checks.
PROCESS_SAMPLES = 1025
# Allow floating-point rounding when a sampled speed is exactly at its bound.
SPEED_BOUND_ULPS = 32


def inspect_process_curve(
    label: str,
    states: Any,
    duration: float,
    checker: Any,
    limits: Any,
    speeds: Any,
    pose_for: Any,
    feasibility_band: float,
) -> dict[str, Any]:
    """Measure generated cubic in the workpiece frame; no process tolerance implied.

    The clock and endpoint poses are modeled from a previously verified fresh
    candidate. Axis order is track m, six arm rad, positioner rad, tailstock m.
    Collision is sampled, while cubic position extrema are numerical roots.
    """
    q = np.asarray(states, dtype=float)
    bounds = np.asarray(limits, dtype=float)
    speed = np.asarray(speeds, dtype=float)
    if speed.shape != (7,) or not np.isfinite(speed).all() or np.any(speed <= 0):
        raise ProcessCurveCheckError("seven finite positive sourced axis speeds required")
    _, curve = build_process_curve(label, q, duration)
    times = np.linspace(0, duration, PROCESS_SAMPLES)
    modeled = np.column_stack((curve(times), np.tile(q[0, 7:], (len(times), 1))))
    checks = inspect_sampled_curve(checker, modeled, bounds, feasibility_band)  # type: ignore[no-untyped-call]
    extrema = cubic_position_extrema(curve)
    excursions = [
        max(bounds[i, 0] - row["minimum"], row["maximum"] - bounds[i, 1], 0.0)
        for i, row in enumerate(extrema)
    ]
    if any(value > feasibility_band for value in excursions):
        raise CurveLimitError(f"{label} actual cubic extrema exceed configured band: {excursions}")
    poses = np.asarray([pose_for(state) for state in modeled])
    first, last = np.asarray(pose_for(q[0])), np.asarray(pose_for(q[-1]))
    geometry = assess_process_geometry(first, last, poses)  # type: ignore[no-untyped-call]
    progress = times / duration
    target_xyz = first[:3, 3] + progress[:, None] * (last[:3, 3] - first[:3, 3])
    target_rotations = interpolate_rotations(first[:3, :3], last[:3, :3], progress)  # type: ignore[no-untyped-call]
    geometry.update(
        {
            "max_commanded_position_error_m": float(
                np.linalg.norm(poses[:, :3, 3] - target_xyz, axis=1).max()
            ),
            "max_commanded_orientation_error_rad": float(
                rotation_errors(target_rotations, poses[:, :3, :3]).max()  # type: ignore[no-untyped-call]
            ),
            "commanded_feed_m_s": float(np.linalg.norm(last[:3, 3] - first[:3, 3]) / duration),
        }
    )
    ratios = np.max(np.abs(curve(times, 1)), axis=0) / speed
    feed = np.linalg.norm(np.gradient(poses[:, :3, 3], times, axis=0), axis=1)
    geometry["sampled_feed_min_m_s"] = float(feed.min())
    geometry["sampled_feed_max_m_s"] = float(feed.max())
    return {
        "stage": label,
        "sample_count": PROCESS_SAMPLES,
        "fitted": geometry,
        "sampled_speed_ratio_by_axis": ratios.tolist(),
        "sampled_collision_and_position": checks,
        "sampled_sourced_speeds_pass": bool(
            np.all(ratios <= 1 + SPEED_BOUND_ULPS * np.finfo(float).eps)
        ),
        "position_extrema": extrema,
        "extrema_raw_limit_excursions": excursions,
        "start_velocity": curve(0.0, 1).tolist(),
        "end_velocity": curve(duration, 1).tolist(),
        "start_acceleration": curve(0.0, 2).tolist(),
        "end_acceleration": curve(duration, 2).tolist(),
    }


class ProcessCurveCheckError(RuntimeError):
    """The declared modeled process curve cannot be independently checked."""
