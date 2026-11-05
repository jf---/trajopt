"""Explicit modeled easing along one unchanged nine-axis fixture chord."""

import numpy as np

# Double-precision scale for testing whether sampled states share one chord.
CHORD_ROUNDOFF_ULPS = 64
# Bisection resolves a binary64 time to representable precision.
BISECTION_ITERATIONS = 64


class FixtureClockError(RuntimeError):
    """The fixture path, ordered waypoints, or modeled clock is invalid."""


def _progress(u):
    return u**3 * (10 + u * (-15 + 6 * u))


def _inverse_progress(alpha):
    if alpha == 0:
        return 0.0
    if alpha == 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(BISECTION_ITERATIONS):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if _progress(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fixture_chain_clock(stages, duration_s):
    """Ease one ordered straight fixture path, stopping only at its two ends.

    Nine-axis states use track m, six arm rad, positioner rad, tailstock m.
    The caller provides a modeled duration, not measured controller timing.
    The function rejects noncollinear or reversing geometry and leaves source
    states unchanged; intermediate stage waypoint times are derived, not stops.
    """
    if not stages or not np.isfinite(duration_s) or duration_s <= 0:
        raise FixtureClockError("positive finite duration and fixture stages required")
    arrays = []
    for stage in stages:
        q = np.asarray(stage["states"], dtype=float)
        if (
            not str(stage["stage"]).startswith("fixture:")
            or q.ndim != 2
            or q.shape[1] != 9
            or len(q) < 2
            or not np.isfinite(q).all()
        ):
            raise FixtureClockError("finite nine-axis fixture stages required")
        arrays.append(q)
    first, last = arrays[0][0], arrays[-1][-1]
    delta = last - first
    axis = int(np.argmax(np.abs(delta)))
    if delta[axis] == 0:
        raise FixtureClockError("fixture chain has no motion")
    scale = np.maximum(1.0, np.maximum(np.abs(first), np.abs(last)))
    band = CHORD_ROUNDOFF_ULPS * np.finfo(float).eps * scale
    previous = 0.0
    waypoint_progress = [0.0]
    for i, q in enumerate(arrays):
        if i and np.any(np.abs(q[0] - arrays[i - 1][-1]) > band):
            raise FixtureClockError("fixture stages do not join")
        if np.array_equal(q[0], q[-1]):
            raise FixtureClockError(
                "stationary fixture stage requires its own event or dwell clock"
            )
        alpha = (q[:, axis] - first[axis]) / delta[axis]
        progress_band = CHORD_ROUNDOFF_ULPS * np.finfo(float).eps
        if (
            np.any(alpha < -progress_band)
            or np.any(alpha > 1 + progress_band)
            or alpha[0] < previous - progress_band
            or np.any(np.diff(alpha) < -progress_band)
        ):
            raise FixtureClockError("fixture chord progress is not monotone")
        expected = first + alpha[:, None] * delta
        if np.any(np.abs(q - expected) > band):
            raise FixtureClockError("fixture samples do not share one chord")
        previous = float(alpha[-1])
        waypoint_progress.append(float(np.clip(previous, 0, 1)))
    return {
        "start": first.tolist(),
        "end": last.tolist(),
        "duration_s": float(duration_s),
        "stage_labels": [s["stage"] for s in stages],
        "stage_times_s": [float(duration_s) * _inverse_progress(x) for x in waypoint_progress],
        "scope": "fixed 9-axis chord, quintic progress, modeled clock; no sourced fixture limit or controller timing",
    }


def evaluate_fixture_clock(clock, time_s):
    """Evaluate modeled nine-axis position, velocity and acceleration at time."""
    duration = float(clock["duration_s"])
    if not np.isfinite(time_s) or not 0 <= time_s <= duration:
        raise FixtureClockError("time lies outside modeled clock")
    u = float(time_s) / duration
    a = np.asarray(clock["start"])
    delta = np.asarray(clock["end"]) - a
    position = a + delta * _progress(u)
    velocity = delta * (30 * u**2 * (1 - u) ** 2 / duration)
    acceleration = delta * (60 * u * (1 - u) * (1 - 2 * u) / duration**2)
    return position, velocity, acceleration
