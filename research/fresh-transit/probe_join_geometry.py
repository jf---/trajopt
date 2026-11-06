"""Classify exact Cartesian process joins without constraining robot joints."""

from math import isfinite, pi
from typing import Any, Literal, NamedTuple, NewType

import numpy as np

from tesseract_qt_py.core.math_utils import rotation_delta_rotvec  # type: ignore[import-untyped]

# Dimensionless binary64 rounding allowance, not a physical blend tolerance.
MACHINE_ROUNDOFF_ULPS = 256
WorkpieceDisplacementMeters = NewType("WorkpieceDisplacementMeters", tuple[float, float, float])
WorkpieceRotationRadians = NewType("WorkpieceRotationRadians", tuple[float, float, float])


class ProbeJoinGeometryError(RuntimeError):
    """The requested original Cartesian interpolation cannot be classified."""


class WorkpieceTwist(NamedTuple):
    """Translation in metres and orientation rotation-vector radians per unit progress."""

    translation: WorkpieceDisplacementMeters
    rotation: WorkpieceRotationRadians


class ProbeJoinEvidence(NamedTuple):
    """Exact-geometry tangent evidence; no implied joint or controller state."""

    status: Literal[
        "compatible_with_positive_progress",
        "zero_tcp_velocity_for_exact_corner",
        "stationary_leg_requires_event_semantics",
    ]
    incoming: WorkpieceTwist
    outgoing: WorkpieceTwist
    right_rate_per_left_rate: float | None
    dimensionless_alignment_residual: float | None
    machine_band: float
    translation_below_roundoff: bool
    rotation_below_roundoff: bool


def _pose(value: Any) -> Any:
    matrix = np.asarray(value, dtype=float)
    band = MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or np.max(np.abs(matrix[3] - np.array([0.0, 0.0, 0.0, 1.0]))) > band
        or np.max(np.abs(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3))) > band
        or abs(float(np.linalg.det(matrix[:3, :3])) - 1.0) > band
    ):
        raise ProbeJoinGeometryError("finite rigid workpiece-frame pose required")
    return matrix


def _twist(first: Any, last: Any) -> WorkpieceTwist:
    displacement = np.asarray(last[:3, 3] - first[:3, 3], dtype=float)
    rotation = np.asarray(rotation_delta_rotvec(first[:3, :3], last[:3, :3]), dtype=float)
    band = MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps
    # At pi the shortest rotation arc is not unique; require an explicit path.
    if not np.isfinite(rotation).all() or abs(float(np.linalg.norm(rotation)) - pi) <= band:
        raise ProbeJoinGeometryError("ambiguous half-turn orientation interpolation")
    return WorkpieceTwist(
        WorkpieceDisplacementMeters(
            (float(displacement[0]), float(displacement[1]), float(displacement[2]))
        ),
        WorkpieceRotationRadians((float(rotation[0]), float(rotation[1]), float(rotation[2]))),
    )


def inspect_probe_join(first: Any, joint: Any, last: Any) -> ProbeJoinEvidence:
    """Check positive scalar progress for exact straight/shortest-arc legs.

    Compare workpiece-frame translation and orientation separately normalized
    by their own maximum magnitudes, preserving their different units. A zero
    TCP twist does not impose zero arm or track velocity: Jacobian nullspace
    motion remains available. Machine-scale coincidence is a numerical result,
    never an authorized process tolerance or controller blend assumption.
    Components below binary64 pose roundoff are excluded from the directional
    comparison but preserved in the raw reported twists and explicitly flagged.
    """
    first_pose, joint_pose, last_pose = _pose(first), _pose(joint), _pose(last)
    incoming, outgoing = _twist(first_pose, joint_pose), _twist(joint_pose, last_pose)
    a_pos, b_pos = np.asarray(incoming.translation), np.asarray(outgoing.translation)
    a_rot, b_rot = np.asarray(incoming.rotation), np.asarray(outgoing.rotation)
    pos_scale = max(float(np.linalg.norm(a_pos)), float(np.linalg.norm(b_pos)))
    rot_scale = max(float(np.linalg.norm(a_rot)), float(np.linalg.norm(b_rot)))
    pose_scale = max(
        1.0, *(float(np.linalg.norm(p[:3, 3])) for p in (first_pose, joint_pose, last_pose))
    )
    pos_roundoff = MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps * pose_scale
    rot_roundoff = MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps
    pos_trimmed = bool(pos_scale <= pos_roundoff)
    rot_trimmed = bool(rot_scale <= rot_roundoff)
    if pos_trimmed:
        a_pos, b_pos, pos_scale = np.zeros(3), np.zeros(3), 0.0
    if rot_trimmed:
        a_rot, b_rot, rot_scale = np.zeros(3), np.zeros(3), 0.0
    a = np.r_[a_pos / pos_scale if pos_scale else a_pos, a_rot / rot_scale if rot_scale else a_rot]
    b = np.r_[b_pos / pos_scale if pos_scale else b_pos, b_rot / rot_scale if rot_scale else b_rot]
    band = (
        MACHINE_ROUNDOFF_ULPS
        * np.finfo(float).eps
        * max(1.0, float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    )
    if not np.any(a) or not np.any(b):
        return ProbeJoinEvidence(
            "stationary_leg_requires_event_semantics",
            incoming,
            outgoing,
            None,
            None,
            band,
            pos_trimmed,
            rot_trimmed,
        )
    ratio = float(a @ b / (b @ b))
    residual = float(np.linalg.norm(a - ratio * b))
    if not isfinite(ratio) or not isfinite(residual):
        raise ProbeJoinGeometryError("nonfinite normalized Cartesian tangent")
    status: Literal["compatible_with_positive_progress", "zero_tcp_velocity_for_exact_corner"] = (
        "compatible_with_positive_progress"
        if ratio > 0 and residual <= band
        else "zero_tcp_velocity_for_exact_corner"
    )
    return ProbeJoinEvidence(
        status,
        incoming,
        outgoing,
        ratio if ratio > 0 else None,
        residual,
        band,
        pos_trimmed,
        rot_trimmed,
    )
