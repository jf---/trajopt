"""Expose existing modeled spline, process and fixture curves as actual stages."""

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
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
from fixture_chain_clock import evaluate_fixture_clock

_fixture_evaluate: Callable[[Mapping[str, Any], float], tuple[Any, Any, Any]] = (
    evaluate_fixture_clock
)

# Binary64 allowance for deciding whether a progress derivative is negative.
PROGRESS_ROUNDOFF_ULPS = 128


class StageAdapterError(RuntimeError):
    """A source curve, clock or evaluation request is invalid."""


def _positive_duration(duration: float) -> Seconds:
    if not isfinite(duration) or duration <= 0:
        raise StageAdapterError("stage duration must be finite and positive")
    return Seconds(float(duration))


def _time(time: float, duration: Seconds) -> float:
    if not isfinite(time) or not 0 <= time <= duration:
        raise StageAdapterError("time is outside the actual stage")
    return float(time)


def _kinematics(position: Any, velocity: Any, acceleration: Any) -> KinematicState:
    values = []
    for item in (position, velocity, acceleration):
        value = np.asarray(item, dtype=float)
        if value.shape != (9,) or not np.isfinite(value).all():
            raise StageAdapterError("curve must return finite nine-axis position and derivatives")
        values.append(tuple(float(x) for x in value))
    return KinematicState(Position9(values[0]), Velocity9(values[1]), Acceleration9(values[2]))


def _arm_with_fixture(
    position: Any, velocity: Any, acceleration: Any, fixture: Sequence[float]
) -> KinematicState:
    fixed = np.asarray(fixture, dtype=float)
    if fixed.shape != (2,) or not np.isfinite(fixed).all():
        raise StageAdapterError("fixed positioner and tailstock required")
    return _kinematics(
        np.r_[position, fixed], np.r_[velocity, 0.0, 0.0], np.r_[acceleration, 0.0, 0.0]
    )


def transit_stage(
    label: str, duration: float, model: Any, nodes: Any, fixture: Sequence[float]
) -> ActualStage:
    """Adapt the native quintic and its actual fixed-fixture derivatives."""
    seconds = _positive_duration(duration)
    positions = np.asarray(nodes, dtype=float)

    def evaluate(time: Seconds) -> KinematicState:
        at = _time(time, seconds)
        position, velocity, acceleration = model.evaluate(np.asarray([at]), positions, 2)
        return _arm_with_fixture(position[0], velocity[0], acceleration[0], fixture)

    return ActualStage(label, seconds, evaluate)


def process_stage(
    label: str,
    duration: float,
    curve: Callable[..., Any],
    fixture: Sequence[float],
    start_rate: float,
    end_rate: float,
) -> ActualStage:
    """Adapt generated IK progress with its modeled endpoint rates and clock."""
    seconds = _positive_duration(duration)
    if not all(isfinite(x) and x >= 0 for x in (start_rate, end_rate)):
        raise StageAdapterError("finite nonnegative progress rates required")
    a = 10 - 6 * start_rate - 4 * end_rate
    b = -15 + 8 * start_rate + 7 * end_rate
    c = 6 - 3 * start_rate - 3 * end_rate
    progress_polynomial = np.polynomial.Polynomial([0.0, start_rate, 0.0, a, b, c])
    derivative = progress_polynomial.deriv()
    roots = derivative.deriv().roots()
    band = PROGRESS_ROUNDOFF_ULPS * np.finfo(float).eps * max(1.0, start_rate, end_rate)
    candidates = [0.0, 1.0]
    for root in roots:
        numeric_root = complex(root)
        if (
            abs(numeric_root.imag) <= band * max(1.0, abs(numeric_root.real))
            and 0 < numeric_root.real < 1
        ):
            candidates.append(numeric_root.real)
    if min(float(derivative(value)) for value in candidates) < -band:
        raise StageAdapterError("process progress must be monotone")

    def evaluate(time: Seconds) -> KinematicState:
        tau = _time(time, seconds) / seconds
        progress = start_rate * tau + a * tau**3 + b * tau**4 + c * tau**5
        progress = float(np.clip(progress, 0.0, 1.0))
        speed = (start_rate + 3 * a * tau**2 + 4 * b * tau**3 + 5 * c * tau**4) / seconds
        accel = (6 * a * tau + 12 * b * tau**2 + 20 * c * tau**3) / seconds**2
        position = np.asarray(curve(progress), dtype=float)
        tangent = np.asarray(curve(progress, 1), dtype=float)
        curvature = np.asarray(curve(progress, 2), dtype=float)
        return _arm_with_fixture(
            position, tangent * speed, curvature * speed**2 + tangent * accel, fixture
        )

    return ActualStage(label, seconds, evaluate)


def fixture_stage(
    label: str, chain: Mapping[str, Any], start_time: float, end_time: float
) -> ActualStage:
    """Adapt one window of the unchanged existing nine-axis fixture clock."""
    seconds = _positive_duration(end_time - start_time)
    if not isfinite(start_time) or not isfinite(end_time) or start_time < 0:
        raise StageAdapterError("fixture window must have finite nonnegative times")
    if end_time > float(chain["duration_s"]):
        raise StageAdapterError("fixture window exceeds its source clock")

    def evaluate(time: Seconds) -> KinematicState:
        position, velocity, acceleration = _fixture_evaluate(
            chain, start_time + _time(time, seconds)
        )
        return _kinematics(position, velocity, acceleration)

    return ActualStage(label, seconds, evaluate)


def weld_stage(
    label: str,
    duration: float,
    curve: Callable[..., Any],
    start_time: float,
    fixture: Sequence[float],
) -> ActualStage:
    """Adapt one modeled window of an existing generated weld curve."""
    seconds = _positive_duration(duration)
    if not isfinite(start_time) or start_time < 0:
        raise StageAdapterError("weld curve start time must be finite and nonnegative")

    def evaluate(time: Seconds) -> KinematicState:
        at = start_time + _time(time, seconds)
        return _arm_with_fixture(curve(at), curve(at, 1), curve(at, 2), fixture)

    return ActualStage(label, seconds, evaluate)


# Binary64 allowance for independently evaluating the same modeled curve.
CURVE_PARITY_ULPS = 32768


def check_stage_samples(
    stage: ActualStage, times: Any, positions: Any, velocities: Any, accelerations: Any
) -> None:
    """Compare endpoint and interior samples with the original curve evaluator."""
    times_array = np.asarray(times, dtype=float)
    arrays = [np.asarray(value, dtype=float) for value in (positions, velocities, accelerations)]
    if (
        times_array.ndim != 1
        or len(times_array) < 2
        or not np.isfinite(times_array).all()
        or any(value.shape != (len(times_array), 9) for value in arrays)
        or any(not np.isfinite(value).all() for value in arrays)
    ):
        raise StageAdapterError("finite nine-axis sampled curves required")
    clock_band = (
        CURVE_PARITY_ULPS
        * np.finfo(float).eps
        * max(1.0, abs(float(times_array[0])), abs(float(times_array[-1])), stage.duration)
    )
    if abs(float(times_array[-1] - times_array[0]) - stage.duration) > clock_band:
        raise StageAdapterError(f"{stage.label}: sampled clock does not match stage duration")
    for index in {0, len(times_array) // 2, len(times_array) - 1}:
        time = (
            stage.duration
            if index == len(times_array) - 1
            else Seconds(float(times_array[index] - times_array[0]))
        )
        state = stage.evaluate(time)
        for name, actual, expected in zip(
            ("position", "velocity", "acceleration"), state, arrays, strict=True
        ):
            if actual is None:
                raise StageAdapterError(f"{stage.label}: {name} is unknown")
            reference = expected[index]
            band = (
                CURVE_PARITY_ULPS * np.finfo(float).eps * max(1.0, float(np.max(np.abs(reference))))
            )
            if np.max(np.abs(np.asarray(actual) - reference)) > band:
                raise StageAdapterError(f"{stage.label}: {name} sampled mismatch")
