"""Unit-scaled shared Cartesian boundary velocity with generated nullspace freedom."""

from typing import Any, Literal, NamedTuple, NewType

import numpy as np

# Dimensionless binary64 allowance for solving and checking scaled linear systems.
MACHINE_ROUNDOFF_ULPS = 256
WorkpieceTwistPerSecond = NewType("WorkpieceTwistPerSecond", tuple[float, ...])
WorkpieceTwistScalePerSecond = NewType("WorkpieceTwistScalePerSecond", tuple[float, ...])
GeneratedJointVelocity7 = NewType("GeneratedJointVelocity7", tuple[float, ...])
SourcedJointSpeed7 = NewType("SourcedJointSpeed7", tuple[float, ...])


class ProbeSharedVelocityError(RuntimeError):
    """The fixed-fixture shared-velocity problem is invalid or singular."""


class SharedVelocityEvidence(NamedTuple):
    """Modeled seven-axis solution, not a measured controller boundary."""

    status: Literal["modeled_candidate", "candidate_exceeds_sourced_speed"]
    velocity: GeneratedJointVelocity7
    task_residual: WorkpieceTwistPerSecond
    sourced_speed_ratio: tuple[float, ...]
    scaled_jacobian_rank: int


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
