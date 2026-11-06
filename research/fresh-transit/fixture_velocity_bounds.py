"""Actual-velocity constraints for a modeled fixture chain."""

from typing import Any

import numpy as np


class FixtureVelocityError(RuntimeError):
    """Velocity-bound inputs are invalid."""


def fixture_velocity_bounds(
    stage_states: Any, stage_times: Any, sample_times: Any, robot_speeds: Any
) -> list[list[tuple[float, float]]]:
    """Bound seven sourced robot speeds and preserve each fixture-axis winding.

    States use track m, six arm rad, positioner rad and tailstock m. Clocks use
    modeled seconds; speed bounds use axis units/s. No unsourced fixture speed
    ceiling or robot-wide monotonicity is introduced.
    """
    states = np.asarray(stage_states, dtype=float)
    times = np.asarray(stage_times, dtype=float)
    samples = np.asarray(sample_times, dtype=float)
    speeds = np.asarray(robot_speeds, dtype=float)
    if (
        states.ndim != 2
        or states.shape[1] != 9
        or len(states) < 2
        or times.shape != (len(states),)
        or samples.ndim != 1
        or len(samples) == 0
        or speeds.shape != (7,)
        or not np.isfinite(states).all()
        or not np.isfinite(times).all()
        or not np.isfinite(samples).all()
        or not np.isfinite(speeds).all()
        or np.any(np.diff(times) <= 0.0)
        or np.any(np.diff(samples) <= 0.0)
        or np.any(speeds <= 0.0)
        or samples[0] < times[0]
        or samples[-1] > times[-1]
    ):
        raise FixtureVelocityError("valid nine-axis targets, speeds and ordered clocks required")
    result = []
    for sample in samples:
        active = [i for i in range(len(times) - 1) if times[i] <= sample <= times[i + 1]]
        limits = [(-float(speed), float(speed)) for speed in speeds]
        for axis in (7, 8):
            lower, upper = -float("inf"), float("inf")
            for i in active:
                delta = states[i + 1, axis] - states[i, axis]
                if delta > 0.0:
                    lower = max(lower, 0.0)
                elif delta < 0.0:
                    upper = min(upper, 0.0)
                else:
                    lower, upper = 0.0, 0.0
            limits.append((lower, upper))
        result.append(limits)
    return result
