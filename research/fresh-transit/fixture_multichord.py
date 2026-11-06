"""Native nine-axis interpolation through ordered generated fixture targets."""

from typing import Any

import _quintic_spline as native  # type: ignore[import-not-found]
import numpy as np

# The native quintic interpolator requires six time knots; midpoints are taken
# from existing sampled fixture chords, not added scene collision samples.
MINIMUM_KNOTS = 6
# Binary64 comparison for neighboring representations of the same state.
JOIN_ROUNDOFF_ULPS = 128


class FixtureMultichordError(RuntimeError):
    """The generated fixture path or modeled boundaries are invalid."""


def fixture_knot_times(durations_s: Any) -> tuple[Any, Any]:
    """Return ordered stage boundaries and native knots on a modeled clock."""
    duration = np.asarray(durations_s, dtype=float)
    if (
        duration.ndim != 1
        or not len(duration)
        or not np.isfinite(duration).all()
        or np.any(duration <= 0)
    ):
        raise FixtureMultichordError("positive finite fixture durations required")
    stage_times = np.r_[0.0, np.cumsum(duration)]
    times = list(stage_times)
    times.extend((stage_times[:-1] + duration / 2).tolist())
    if len(times) < MINIMUM_KNOTS:
        times.extend(np.linspace(0.0, float(stage_times[-1]), MINIMUM_KNOTS).tolist())
    return stage_times, np.unique(times)


def build_fixture_multichord(
    stages: list[dict[str, Any]], durations_s: Any, v0: Any, a0: Any, v1: Any, a1: Any
) -> tuple[Any, Any, list[float]]:
    """Interpolate ordered nine-axis targets on a supplied modeled clock.

    Track and tailstock use metres, the six arm axes and positioner use radians;
    velocity and acceleration use these units per second and second squared.
    Endpoint derivatives are supplied explicitly and need not imply rest.
    The interior path can depart source chords and requires independent checks.
    """
    duration = np.asarray(durations_s, dtype=float)
    boundaries = [np.asarray(x, dtype=float) for x in (v0, a0, v1, a1)]
    if (
        not stages
        or duration.shape != (len(stages),)
        or not np.isfinite(duration).all()
        or np.any(duration <= 0.0)
        or any(x.shape != (9,) or not np.isfinite(x).all() for x in boundaries)
    ):
        raise FixtureMultichordError("finite nine-axis boundaries and positive clocks required")
    arrays = []
    for row in stages:
        state = np.asarray(row["states"], dtype=float)
        if (
            not row["stage"].startswith("fixture:")
            or state.ndim != 2
            or state.shape[1] != 9
            or len(state) < 2
            or not np.isfinite(state).all()
        ):
            raise FixtureMultichordError("finite ordered nine-axis fixture samples required")
        if np.array_equal(state[0], state[-1]):
            raise FixtureMultichordError("stationary fixture event requires its own explicit clock")
        arrays.append(state)
    for previous, current in zip(arrays[:-1], arrays[1:], strict=True):
        band = (
            JOIN_ROUNDOFF_ULPS
            * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(previous[-1]))), float(np.max(np.abs(current[0]))))
        )
        if np.any(np.abs(previous[-1] - current[0]) > band):
            raise FixtureMultichordError("fixture stages have different boundary states")
    stage_times, times = fixture_knot_times(duration)
    nodes = []
    for time in times:
        index = min(int(np.searchsorted(stage_times, time, side="right") - 1), len(arrays) - 1)
        local = (float(time) - stage_times[index]) / duration[index]
        samples = arrays[index]
        xp = np.linspace(0.0, 1.0, len(samples))
        nodes.append([float(np.interp(local, xp, samples[:, axis])) for axis in range(9)])
    model = native.QuinticSplineModel.build(times.tolist(), *boundaries)
    return model, np.asarray(nodes), stage_times.tolist()
