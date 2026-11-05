"""Contracts for uniform transit timing and explicit fixture event times."""

import math

import numpy as np

# Binary64 allowance for matching modeled clocks; not a physical tolerance.
CLOCK_ROUNDOFF_ULPS = 128


class DurationPolicyError(RuntimeError):
    """A timing policy or a saved transit violates the shared-clock contract."""


def scale_transit_clock(stages, clock, factor, fixture_times):
    """Scale every transit uniformly while preserving process and fixture duration."""
    if isinstance(factor, bool) or not math.isfinite(factor) or factor <= 0:
        raise DurationPolicyError("transit duration factor must be finite and positive")
    timing = clock.get("stages", ())
    if len(stages) != len(timing) or not stages:
        raise DurationPolicyError("stage and clock lengths differ")
    labels = [stage.get("stage") for stage in stages]
    if any(not isinstance(label, str) or not label for label in labels):
        raise DurationPolicyError("stage labels must be nonempty strings")
    if len(set(labels)) != len(labels) or any(
        a != b.get("stage") for a, b in zip(labels, timing, strict=True)
    ):
        raise DurationPolicyError("stage labels must be unique and match the clock")
    original = np.asarray([item.get("duration_s") for item in timing], dtype=float)
    if not np.isfinite(original).all() or np.any(original <= 0):
        raise DurationPolicyError(
            "stage durations must be finite and positive; zero-time events are unsupported"
        )
    fixture_indices = [i for i, name in enumerate(labels) if name.startswith("fixture:")]
    events = np.asarray(fixture_times, dtype=float)
    if (
        events.shape != (len(fixture_indices) + 1,)
        or not np.isfinite(events).all()
        or np.any(np.diff(events) <= 0)
    ):
        raise DurationPolicyError("fixture event times must be finite, ordered and complete")
    band = CLOCK_ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, abs(float(events[-1])))
    if (
        abs(float(events[0])) > band
        or abs(float(events[-1] - original[fixture_indices].sum())) > band
    ):
        raise DurationPolicyError("fixture total duration must remain unchanged")
    result = []
    for i, item in enumerate(timing):
        duration = float(original[i]) * (factor if labels[i].startswith("transit:") else 1.0)
        if i in fixture_indices:
            k = fixture_indices.index(i)
            duration = float(events[k + 1] - events[k])
        result.append({**item, "duration_s": duration})
    return {**clock, "stages": result}


def validate_transit_results(stages, policy_clock, results):
    """Require one successful, correctly timed and fixture-preserving curve per transit."""
    labels = [s["stage"] for s in stages if s["stage"].startswith("transit:")]
    if len(results) != len(labels) or any(x.get("status") != "sampled_validated" for x in results):
        raise DurationPolicyError("one sampled-valid result is required for every transit")
    if any(
        not isinstance(x.get("stage"), str) or not isinstance(x.get("curve"), dict) for x in results
    ):
        raise DurationPolicyError("transit results require stage labels and curve data")
    curves = {x["stage"]: x["curve"] for x in results}
    if len(curves) != len(labels) or set(curves) != set(labels):
        raise DurationPolicyError("transit results are duplicated or incomplete")
    for stage, timing in zip(stages, policy_clock["stages"], strict=True):
        name = stage["stage"]
        if not name.startswith("transit:"):
            continue
        curve = curves[name]
        duration = float(curve["duration_s"])
        band = CLOCK_ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, abs(duration))
        knots = np.asarray(curve["knots_s"], dtype=float)
        fixture = np.asarray(stage["states"], dtype=float)[:, 7:]
        if (
            curve.get("stage") != name
            or not math.isfinite(duration)
            or duration <= 0
            or abs(duration - float(timing["duration_s"])) > band
            or knots.ndim != 1
            or len(knots) < 2
            or not np.isfinite(knots).all()
            or abs(knots[0]) > band
            or abs(knots[-1] - duration) > band
            or np.any(np.diff(knots) <= 0)
            or not np.all(fixture == fixture[0])
            or not np.array_equal(np.asarray(curve["fixture"]), fixture[0])
        ):
            raise DurationPolicyError(f"{name}: invalid modeled clock or fixture")
    return curves


def validate_fixture_policy(stages, policy):
    """Verify that the eased chord preserves all ordered nine-axis fixture events."""
    fixture = [stage for stage in stages if stage["stage"].startswith("fixture:")]
    if not fixture:
        raise DurationPolicyError("fixture policy requires at least one fixture stage")
    try:
        start = np.asarray(policy["start"], dtype=float)
        end = np.asarray(policy["end"], dtype=float)
        times = np.asarray(policy["stage_times_s"], dtype=float)
        duration = float(policy["duration_s"])
        events = np.asarray(
            [fixture[0]["states"][0]] + [stage["states"][-1] for stage in fixture], dtype=float
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise DurationPolicyError("fixture policy or source events are malformed") from exc
    if (
        start.shape != (9,)
        or end.shape != (9,)
        or events.shape != (len(fixture) + 1, 9)
        or times.shape != (len(fixture) + 1,)
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or not np.isfinite(events).all()
        or not np.isfinite(times).all()
        or not math.isfinite(duration)
        or duration <= 0
        or np.any(np.diff(times) <= 0)
    ):
        raise DurationPolicyError(
            "fixture policy requires finite ordered events and nine-axis states"
        )
    band = (
        CLOCK_ROUNDOFF_ULPS
        * np.finfo(float).eps
        * max(1.0, duration, float(np.max(np.abs(events))))
    )
    if abs(times[0]) > band or abs(times[-1] - duration) > band:
        raise DurationPolicyError("fixture event times do not cover the modeled chord")
    for left, right in zip(fixture[:-1], fixture[1:], strict=True):
        if np.max(np.abs(np.asarray(left["states"][-1]) - np.asarray(right["states"][0]))) > band:
            raise DurationPolicyError("source fixture stages are not connected")
    u = times / duration
    progress = 10 * u**3 - 15 * u**4 + 6 * u**5
    reconstructed = start + (end - start)[None, :] * progress[:, None]
    error = float(np.max(np.abs(reconstructed - events)))
    if error > band:
        raise DurationPolicyError("fixture easing changes a source event state")
    return error
