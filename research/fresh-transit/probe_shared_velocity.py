"""Unit-scaled shared Cartesian boundary velocity with generated nullspace freedom."""

from typing import Any, Literal, NamedTuple, NewType

import numpy as np

# Dimensionless binary64 allowance for solving and checking scaled linear systems.
MACHINE_ROUNDOFF_ULPS = 256
WorkpieceTwistPerSecond = NewType("WorkpieceTwistPerSecond", tuple[float, ...])
WorkpieceTwistScalePerSecond = NewType("WorkpieceTwistScalePerSecond", tuple[float, ...])
GeneratedJointVelocity7 = NewType("GeneratedJointVelocity7", tuple[float, ...])
SourcedJointSpeed7 = NewType("SourcedJointSpeed7", tuple[float, ...])
GeneratedJointPosition7 = NewType("GeneratedJointPosition7", tuple[float, ...])
JointPositionBounds7 = NewType("JointPositionBounds7", tuple[tuple[float, float], ...])
GeneratedJointAcceleration7 = NewType("GeneratedJointAcceleration7", tuple[float, ...])
WorkpieceAccelerationPerSecond2 = NewType("WorkpieceAccelerationPerSecond2", tuple[float, ...])


class ProbeSharedVelocityError(RuntimeError):
    """The fixed-fixture shared-velocity problem is invalid or singular."""


class SharedVelocityEvidence(NamedTuple):
    """Modeled seven-axis solution, not a measured controller boundary."""

    status: Literal["modeled_candidate", "candidate_exceeds_sourced_speed"]
    velocity: GeneratedJointVelocity7
    task_residual: WorkpieceTwistPerSecond
    sourced_speed_ratio: tuple[float, ...]
    scaled_jacobian_rank: int


class ConstrainedSharedVelocityEvidence(NamedTuple):
    """A local modeled velocity satisfying TCP, speed and active-limit signs."""

    velocity: GeneratedJointVelocity7
    task_residual: WorkpieceTwistPerSecond
    sourced_speed_ratio: tuple[float, ...]
    active_lower_axes: tuple[int, ...]
    active_upper_axes: tuple[int, ...]
    nullspace_step: float


class ConstrainedSharedAccelerationEvidence(NamedTuple):
    """Modeled shared acceleration; relative weights are not physical limits."""

    acceleration: GeneratedJointAcceleration7
    task_residual: WorkpieceAccelerationPerSecond2
    nullspace_step: float


def solve_shared_velocity(
    jacobian: Any,
    desired_twist: WorkpieceTwistPerSecond,
    generated_reference: GeneratedJointVelocity7,
    sourced_speed: SourcedJointSpeed7,
    twist_scale: WorkpieceTwistScalePerSecond,
) -> SharedVelocityEvidence:
    """Project an explicit generated reference onto a workpiece-frame TCP twist.

    The Jacobian is 6 by 7 for fixed fixture, with track metres/s followed by
    six arm radians/s; TCP twist is translation metres/s followed by angular
    radians/s in the workpiece frame. Positive caller-supplied scales normalize
    those two units; sourced axis speeds normalize the optimization metric.
    The reference comes from generated motion, never recorded intermediate joints
    or an assumed controller state. This local candidate does not solve qdd,
    actual process geometry, collision or the whole-tour boundary problem.
    A speed-exceeding candidate does not prove that all nullspace choices fail.
    """
    if generated_reference is None:
        raise ProbeSharedVelocityError("explicit generated reference required")
    j = np.asarray(jacobian, dtype=float)
    target = np.asarray(desired_twist, dtype=float)
    reference = np.asarray(generated_reference, dtype=float)
    speed = np.asarray(sourced_speed, dtype=float)
    scales = np.asarray(twist_scale, dtype=float)
    if (
        j.shape != (6, 7)
        or target.shape != (6,)
        or reference.shape != (7,)
        or speed.shape != (7,)
        or scales.shape != (6,)
        or not all(np.isfinite(item).all() for item in (j, target, reference, speed, scales))
        or np.any(speed <= 0)
        or np.any(scales <= 0)
    ):
        raise ProbeSharedVelocityError(
            "finite Jacobian, twist, reference and positive unit scales required"
        )
    weighted = (j * speed[None, :]) / scales[:, None]
    rhs = (target - j @ reference) / scales
    delta, _, rank, _ = np.linalg.lstsq(weighted, rhs, rcond=None)
    if rank != 6:
        raise ProbeSharedVelocityError("shared Cartesian Jacobian has deficient task rank")
    velocity = reference + speed * delta
    residual = j @ velocity - target
    scaled_residual = residual / scales
    band = (
        MACHINE_ROUNDOFF_ULPS
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.linalg.norm(target / scales)),
            float(np.linalg.norm((j @ velocity) / scales)),
        )
    )
    if not np.isfinite(velocity).all() or np.linalg.norm(scaled_residual) > band:
        raise ProbeSharedVelocityError(
            "shared Cartesian twist solve did not meet its numerical equation"
        )
    ratios = np.abs(velocity) / speed
    status: Literal["modeled_candidate", "candidate_exceeds_sourced_speed"] = (
        "modeled_candidate"
        if np.all(ratios <= 1 + MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps)
        else "candidate_exceeds_sourced_speed"
    )
    return SharedVelocityEvidence(
        status,
        GeneratedJointVelocity7(tuple(map(float, velocity))),
        WorkpieceTwistPerSecond(tuple(map(float, residual))),
        tuple(map(float, ratios)),
        int(rank),
    )


def solve_constrained_shared_velocity(
    jacobian: Any,
    desired_twist: WorkpieceTwistPerSecond,
    generated_reference: GeneratedJointVelocity7,
    sourced_speed: SourcedJointSpeed7,
    twist_scale: WorkpieceTwistScalePerSecond,
    generated_position: GeneratedJointPosition7,
    position_bounds: JointPositionBounds7,
) -> ConstrainedSharedVelocityEvidence:
    """Project onto the 1D nullspace under speeds at an interior shared join.

    The seven-axis source is generated; track position is metres, arm joints
    radians. A machine-scale bound comparison detects an active position limit,
    not a physical margin. An axis at a limit has zero velocity because the
    actual curve must remain in bounds on both sides of the join. Other axes
    retain any compatible nullspace motion. The TCP equation is preserved.
    This local cone does not establish full-curve limits or collision clearance.
    """
    base = solve_shared_velocity(
        jacobian, desired_twist, generated_reference, sourced_speed, twist_scale
    )
    j = np.asarray(jacobian, dtype=float)
    target = np.asarray(desired_twist, dtype=float)
    speed = np.asarray(sourced_speed, dtype=float)
    scales = np.asarray(twist_scale, dtype=float)
    position = np.asarray(generated_position, dtype=float)
    bounds = np.asarray(position_bounds, dtype=float)
    if (
        position.shape != (7,)
        or bounds.shape != (7, 2)
        or not np.isfinite(position).all()
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ProbeSharedVelocityError("finite seven-axis generated position and bounds required")
    scale = np.maximum(1.0, np.maximum(np.abs(position), np.max(np.abs(bounds), axis=1)))
    band = MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps * scale
    if np.any(position < bounds[:, 0] - band) or np.any(position > bounds[:, 1] + band):
        raise ProbeSharedVelocityError("generated position lies outside axis bounds")
    lower = position <= bounds[:, 0] + band
    upper = position >= bounds[:, 1] - band
    weighted = j * speed[None, :] / scales[:, None]
    _, _, vh = np.linalg.svd(weighted, full_matrices=True)
    direction = speed * vh[-1]
    value = np.asarray(base.velocity)
    lo, hi = -float("inf"), float("inf")
    for axis in range(7):
        active = lower[axis] or upper[axis]
        amin = 0.0 if active else -speed[axis]
        amax = 0.0 if active else speed[axis]
        if direction[axis] == 0:
            if value[axis] < amin or value[axis] > amax:
                raise ProbeSharedVelocityError("no feasible TCP velocity at active axis bounds")
            continue
        a = (amin - value[axis]) / direction[axis]
        b = (amax - value[axis]) / direction[axis]
        lo = max(lo, min(a, b))
        hi = min(hi, max(a, b))
    if lo > hi:
        raise ProbeSharedVelocityError("no feasible TCP velocity at active axis bounds")
    step = min(max(0.0, lo), hi)
    velocity = value + step * direction
    residual = j @ velocity - target
    numerical_band = (
        MACHINE_ROUNDOFF_ULPS
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.linalg.norm(target / scales)),
            float(np.linalg.norm((j @ velocity) / scales)),
        )
    )
    if np.linalg.norm(residual / scales) > numerical_band:
        raise ProbeSharedVelocityError("constrained TCP equation did not meet numerical precision")
    ratios = np.abs(velocity) / speed
    if np.any(ratios > 1 + MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps):
        raise ProbeSharedVelocityError("constrained velocity exceeds sourced speed")
    return ConstrainedSharedVelocityEvidence(
        GeneratedJointVelocity7(tuple(map(float, velocity))),
        WorkpieceTwistPerSecond(tuple(map(float, residual))),
        tuple(map(float, ratios)),
        tuple(map(int, np.flatnonzero(lower))),
        tuple(map(int, np.flatnonzero(upper))),
        float(step),
    )


def solve_constrained_shared_acceleration(
    jacobian: Any,
    desired_acceleration: WorkpieceAccelerationPerSecond2,
    relative_joint_weights: GeneratedJointAcceleration7,
    task_scales: WorkpieceAccelerationPerSecond2,
    active_lower_axes: tuple[int, ...],
    active_upper_axes: tuple[int, ...],
) -> ConstrainedSharedAccelerationEvidence:
    """Preserve the differentiated TCP equation and active-limit curvature.

    For a zero-velocity interior join at a lower bound, acceleration must point
    inward; at an upper bound it must point inward in the opposite direction.
    Positive weights only set the relative metric. No physical acceleration
    ceiling or controller boundary is assumed.
    """
    j = np.asarray(jacobian, dtype=float)
    target = np.asarray(desired_acceleration, dtype=float)
    weights = np.asarray(relative_joint_weights, dtype=float)
    scales = np.asarray(task_scales, dtype=float)
    if (
        j.shape != (6, 7)
        or target.shape != (6,)
        or weights.shape != (7,)
        or scales.shape != (6,)
        or not all(np.isfinite(x).all() for x in (j, target, weights, scales))
        or np.any(weights <= 0)
        or np.any(scales <= 0)
        or any(i < 0 or i >= 7 for i in (*active_lower_axes, *active_upper_axes))
    ):
        raise ProbeSharedVelocityError(
            "finite acceleration equation and positive unit weights required"
        )
    weighted = j * weights[None, :] / scales[:, None]
    delta, _, rank, _ = np.linalg.lstsq(weighted, target / scales, rcond=None)
    if rank != 6:
        raise ProbeSharedVelocityError("shared acceleration Jacobian has deficient task rank")
    _, _, vh = np.linalg.svd(weighted, full_matrices=True)
    value = weights * delta
    direction = weights * vh[-1]
    lo, hi = -float("inf"), float("inf")
    for axis in sorted(set(active_lower_axes) | set(active_upper_axes)):
        amin = 0.0 if axis in active_lower_axes else -float("inf")
        amax = 0.0 if axis in active_upper_axes else float("inf")
        if direction[axis] == 0:
            if value[axis] < amin or value[axis] > amax:
                raise ProbeSharedVelocityError(
                    "no feasible shared acceleration at active axis bounds"
                )
            continue
        a = (amin - value[axis]) / direction[axis]
        b = (amax - value[axis]) / direction[axis]
        lo = max(lo, min(a, b))
        hi = min(hi, max(a, b))
    if lo > hi:
        raise ProbeSharedVelocityError("no feasible shared acceleration at active axis bounds")
    step = min(max(0.0, lo), hi)
    acceleration = value + step * direction
    residual = j @ acceleration - target
    band = (
        MACHINE_ROUNDOFF_ULPS
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.linalg.norm(target / scales)),
            float(np.linalg.norm((j @ acceleration) / scales)),
        )
    )
    if np.linalg.norm(residual / scales) > band:
        raise ProbeSharedVelocityError(
            "shared acceleration equation did not meet numerical precision"
        )
    return ConstrainedSharedAccelerationEvidence(
        GeneratedJointAcceleration7(tuple(map(float, acceleration))),
        WorkpieceAccelerationPerSecond2(tuple(map(float, residual))),
        float(step),
    )
