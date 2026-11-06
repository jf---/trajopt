"""Adapt verified joint-linear source stages without inventing corner derivatives."""

from math import isfinite
from typing import Any

import numpy as np
from actual_stage_contract import (
    Acceleration9,
    ActualStage,
    KinematicState,
    Position9,
    Seconds,
    StationaryEvent,
    Velocity9,
)


class LinearStageError(RuntimeError):
    """A verified stage or its modeled interpolation is invalid."""


def actual_linear_stage(label: str, states: Any, duration: float) -> ActualStage | StationaryEvent:
    """Preserve the nine-axis path and expose one-sided modeled endpoints.

    Time is uniformly distributed over the given joint-linear intervals, as in
    the modeled clock. At a corner with different adjacent slopes the actual
    velocity and acceleration are undefined. The endpoint derivatives are
    one-sided derivatives of this interpolation, not controller measurements.
    A stationary zero-time stage becomes a named event without implying rest.
    """
    q = np.asarray(states, dtype=float)
    if q.ndim != 2 or q.shape[1:] != (9,) or len(q) < 2 or not np.isfinite(q).all():
        raise LinearStageError("stage needs at least two finite nine-axis states")
    if not isfinite(duration) or duration < 0:
        raise LinearStageError("stage duration must be finite and nonnegative")
    if duration == 0:
        if not np.all(q == q[0]):
            raise LinearStageError("zero-time event changes nine-axis state")
        return StationaryEvent(label, Position9(tuple(map(float, q[0]))))
    q = q.copy()
    count = len(q) - 1
    dt = float(duration) / count
    slopes = np.diff(q, axis=0) / dt
    knots = np.linspace(0.0, duration, count + 1)

    def evaluate(time: Seconds) -> KinematicState:
        if not isfinite(time) or not 0 <= time <= duration:
            raise LinearStageError("time is outside the stage")
        matches = np.flatnonzero(knots == time)
        if len(matches):
            index = int(matches[0])
            position = q[index]
            if 0 < index < count and not np.array_equal(slopes[index - 1], slopes[index]):
                return KinematicState(Position9(tuple(map(float, position))), None, None)
            velocity = slopes[min(index, count - 1)]
        else:
            index = min(int(float(time) / dt), count - 1)
            fraction = (float(time) - float(knots[index])) / dt
            position = q[index] + fraction * (q[index + 1] - q[index])
            velocity = slopes[index]
        return KinematicState(
            Position9(tuple(map(float, position))),
            Velocity9(tuple(map(float, velocity))),
            Acceleration9((0.0,) * 9),
        )

    return ActualStage(label, Seconds(float(duration)), evaluate)
