"""Exploratory, local Cartesian weld blends through source waypoints."""

from collections.abc import Callable
from math import isfinite
from typing import Any

import numpy as np
from scipy.interpolate import BPoly  # type: ignore[import-untyped]

from tesseract_qt_py.core.math_utils import (  # type: ignore[import-untyped]
    quat_wxyz_to_rotation_matrix,
    rotation_matrix_to_quat_wxyz,
)


class WeldBlendError(RuntimeError):
    """The source path cannot form the explicit local blend hypothesis."""


def build_local_weld_blend(
    points_m: Any, rotations: Any, durations_s: Any
) -> tuple[Callable[..., Any], Callable[..., Any], dict[str, Any]]:
    """Return modeled per-stage position/orientation curves through waypoints.

    Points are workpiece-frame metres, rotations are workpiece-frame 3x3
    matrices and durations are modeled seconds. The first/last stages are
    generated tangent leads. Source-bead corner blends use half the shorter
    adjacent stage on each side; only the generated lead is changed at a
    lead/bead join. No manufacturing tolerance, actual controller timing or
    admissibility is asserted by constructing this exploratory geometry.
    """
    points = np.asarray(points_m, dtype=float)
    matrices = np.asarray(rotations, dtype=float)
    durations = np.asarray(durations_s, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 4
        or matrices.shape != (len(points), 3, 3)
        or durations.shape != (len(points) - 1,)
        or not np.isfinite(points).all()
        or not np.isfinite(matrices).all()
        or not np.isfinite(durations).all()
        or np.any(durations <= 0)
    ):
        raise WeldBlendError(
            "finite workpiece waypoints, rotations and positive durations required"
        )
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths == 0):
        raise WeldBlendError("zero-length process segment")
    directions = delta / lengths[:, None]
    source_velocity = delta / durations[:, None]
    n = len(durations)
    shared = [None] * (n + 1)
    shared[0], shared[-1] = source_velocity[0], source_velocity[-1]
    shared[1], shared[-2] = source_velocity[1], source_velocity[-2]
    for i in range(2, n - 1):
        tangent = directions[i - 1] + directions[i]
        length = float(np.linalg.norm(tangent))
        if length == 0:
            raise WeldBlendError("opposing bead directions have no specified blend tangent")
        shared[i] = (
            tangent / length * min(lengths[i - 1] / durations[i - 1], lengths[i] / durations[i])
        )
    quaternions: list[Any] = []
    for matrix in matrices:
        q = np.asarray(rotation_matrix_to_quat_wxyz(matrix), dtype=float)
        if quaternions and float(q @ quaternions[-1]) < 0:
            q = -q
        quaternions.append(q)

    def source_quaternion(stage: int, t: float) -> tuple[Any, Any, Any]:
        q0, q1 = quaternions[stage], quaternions[stage + 1]
        dot = float(np.clip(q0 @ q1, -1.0, 1.0))
        tangent = q1 - dot * q0
        sine = float(np.linalg.norm(tangent))
        if sine == 0:
            return q0, np.zeros(4), np.zeros(4)
        axis = tangent / sine
        theta = float(np.arctan2(sine, dot))
        omega = theta / durations[stage]
        u = omega * t
        value = q0 * np.cos(u) + axis * np.sin(u)
        derivative = omega * (-q0 * np.sin(u) + axis * np.cos(u))
        return value, derivative, -(omega**2) * value

    orientation_velocity = [None] * (n + 1)
    orientation_acceleration = [None] * (n + 1)
    for i in range(1, n):
        q = quaternions[i]
        _, left_v, left_a = source_quaternion(i - 1, durations[i - 1])
        _, right_v, right_a = source_quaternion(i, 0.0)
        if i == 1:
            velocity, acceleration = right_v, right_a
        elif i == n - 1:
            velocity, acceleration = left_v, left_a
        else:
            velocity = (left_v + right_v) / 2
            velocity = velocity - q * (q @ velocity)
            mean = (left_a + right_a) / 2
            acceleration = mean - q * (q @ mean + velocity @ velocity)
        orientation_velocity[i], orientation_acceleration[i] = velocity, acceleration

    position_windows: list[list[tuple[float, float, Any]]] = [[] for _ in range(n)]
    orientation_windows: list[list[tuple[float, float, Any]]] = [[] for _ in range(n)]
    for i in range(1, n):
        width = min(durations[i - 1], durations[i]) / 2
        use_left = i != n - 1
        use_right = i != 1
        if use_left:
            start = durations[i - 1] - width
            p0 = points[i - 1] + source_velocity[i - 1] * start
            pcurve = BPoly.from_derivatives(
                [start, durations[i - 1]],
                [[p0, source_velocity[i - 1], np.zeros(3)], [points[i], shared[i], np.zeros(3)]],
            )
            q0, v0, a0 = source_quaternion(i - 1, start)
            qcurve = BPoly.from_derivatives(
                [start, durations[i - 1]],
                [
                    [q0, v0, a0],
                    [quaternions[i], orientation_velocity[i], orientation_acceleration[i]],
                ],
            )
            position_windows[i - 1].append((start, durations[i - 1], pcurve))
            orientation_windows[i - 1].append((start, durations[i - 1], qcurve))
        if use_right:
            p1 = points[i] + source_velocity[i] * width
            pcurve = BPoly.from_derivatives(
                [0.0, width],
                [[points[i], shared[i], np.zeros(3)], [p1, source_velocity[i], np.zeros(3)]],
            )
            q1, v1, a1 = source_quaternion(i, width)
            qcurve = BPoly.from_derivatives(
                [0.0, width],
                [
                    [quaternions[i], orientation_velocity[i], orientation_acceleration[i]],
                    [q1, v1, a1],
                ],
            )
            position_windows[i].append((0.0, width, pcurve))
            orientation_windows[i].append((0.0, width, qcurve))

    def validate(stage: int, t: float) -> None:
        if (
            not isinstance(stage, int)
            or not 0 <= stage < n
            or not isfinite(t)
            or not 0 <= t <= durations[stage]
        ):
            raise WeldBlendError("sample outside modeled weld stage")

    def position(stage: int, t: float, derivative: int = 0) -> Any:
        validate(stage, t)
        if derivative not in (0, 1, 2, 3):
            raise WeldBlendError("unsupported position derivative")
        for start, end, curve in position_windows[stage]:
            if start <= t <= end:
                return np.asarray(curve(t, derivative), dtype=float)
        if derivative == 0:
            return points[stage] + source_velocity[stage] * t
        return source_velocity[stage].copy() if derivative == 1 else np.zeros(3)

    def orientation(stage: int, t: float) -> Any:
        validate(stage, t)
        for start, end, curve in orientation_windows[stage]:
            if start <= t <= end:
                q = np.asarray(curve(t), dtype=float)
                length = float(np.linalg.norm(q))
                if not isfinite(length) or length == 0:
                    raise WeldBlendError("degenerate interpolated orientation")
                return np.asarray(quat_wxyz_to_rotation_matrix(q / length), dtype=float)
        return np.asarray(quat_wxyz_to_rotation_matrix(source_quaternion(stage, t)[0]), dtype=float)

    info = {
        "source_bead_internal_join_count": n - 3,
        "position_windows_s": [
            [[float(a), float(b)] for a, b, _ in windows] for windows in position_windows
        ],
        "scope": "local exploratory blends; source waypoint anchors; no approved tolerance",
    }
    return position, orientation, info
