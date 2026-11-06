"""Combine generated robot-track and ordered fixture-axis curves."""

from typing import Any

import numpy as np
from actual_stage_contract import (
    Acceleration9,
    ActualStage,
    KinematicState,
    Position9,
    Seconds,
    Velocity9,
)


class FixtureCompositeError(RuntimeError):
    """The modeled full-axis fixture stage is invalid."""


def fixture_composite_stage(
    label: str, model: Any, nodes: Any, fixture_axes: tuple[Any, Any], start_s: float, end_s: float
) -> ActualStage:
    """Adapt one nine-axis stage with explicit modeled q, qd and qdd.

    Track and tailstock use metres, arm and positioner use radians. Times are
    modeled seconds; the supplied curves do not describe observed controller data.
    """
    positions = np.asarray(nodes, dtype=float)
    if (
        not label
        or positions.ndim != 2
        or positions.shape[1] != 9
        or not np.isfinite(positions).all()
        or len(fixture_axes) != 2
        or not np.isfinite(start_s)
        or not np.isfinite(end_s)
        or start_s >= end_s
    ):
        raise FixtureCompositeError("valid nine-axis fixture curve and positive clock required")
    duration = float(end_s - start_s)

    def evaluate(time_s: Seconds) -> KinematicState:
        local = float(time_s)
        if not np.isfinite(local) or not 0 <= local <= duration:
            raise FixtureCompositeError("sample outside modeled fixture stage")
        at = end_s if local == duration else start_s + local
        q, vel, acc = model.evaluate(np.asarray([at]), positions, 2)
        for axis, fixture in zip((7, 8), fixture_axes, strict=True):
            values = fixture.evaluate_many([at])[0]
            q[0, axis], vel[0, axis], acc[0, axis] = values[:3]
        return KinematicState(
            Position9(tuple(map(float, q[0]))),
            Velocity9(tuple(map(float, vel[0]))),
            Acceleration9(tuple(map(float, acc[0]))),
        )

    return ActualStage(label, Seconds(duration), evaluate)
