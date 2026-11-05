"""Actual nine-axis stage joins without inferred derivatives or event time.

Axis order is track metres, six arm radians, positioner radians, tailstock
metres. Velocity and acceleration use those respective units per second and
per second squared. An evaluator describes a modeled curve; it is not evidence
of physical controller motion.
"""

from collections.abc import Callable, Sequence
from math import isfinite
from typing import NamedTuple, NewType

Seconds = NewType("Seconds", float)
Position9 = NewType("Position9", tuple[float, ...])
Velocity9 = NewType("Velocity9", tuple[float, ...])
Acceleration9 = NewType("Acceleration9", tuple[float, ...])


class JoinContractError(RuntimeError):
    """An actual curve, stationary event, or requested join is unsupported."""


class KinematicState(NamedTuple):
    """A nine-axis state and independently known or unknown derivatives."""

    position: Position9
    velocity: Velocity9 | None
    acceleration: Acceleration9 | None


class ActualStage(NamedTuple):
    """Positive-duration stage evaluated against its own modeled seconds."""

    label: str
    duration: Seconds
    evaluate: Callable[[Seconds], KinematicState]


class StationaryEvent(NamedTuple):
    """Named zero-time event at one nine-axis position, with no implied rest."""

    label: str
    position: Position9


class JoinDiagnostic(NamedTuple):
    """Actual endpoint differences; events preserve their given order."""

    left: str | None
    right: str | None
    events: tuple[str, ...]
    position_delta: tuple[float, ...]
    position_residuals: tuple[tuple[float, ...], ...]
    velocity_delta: tuple[float, ...] | None
    acceleration_delta: tuple[float, ...] | None


def _vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    if len(values) != 9 or any(not isfinite(value) for value in values):
        raise JoinContractError(f"{name} requires nine finite axis values")
    return tuple(float(value) for value in values)


def _validated_state(value: KinematicState) -> KinematicState:
    position = Position9(_vector(value.position, "position"))
    velocity = None if value.velocity is None else Velocity9(_vector(value.velocity, "velocity"))
    acceleration = (
        None
        if value.acceleration is None
        else Acceleration9(_vector(value.acceleration, "acceleration"))
    )
    return KinematicState(position, velocity, acceleration)


def _state(stage: ActualStage, time: float) -> KinematicState:
    return _validated_state(stage.evaluate(Seconds(time)))


def _difference(right: Sequence[float], left: Sequence[float]) -> tuple[float, ...]:
    return tuple(b - a for a, b in zip(left, right, strict=True))


def diagnostic_joins(
    stages: Sequence[ActualStage | StationaryEvent],
    *,
    initial_state: KinematicState | None = None,
    final_state: KinematicState | None = None,
) -> list[JoinDiagnostic]:
    """Evaluate boundaries, retain zero-time events, and report measured gaps.

    Endpoint events have an unknown derivative on their open side. Events
    between two motions do not themselves impose a stop. Consecutive zero-time
    events must have the same nine-axis state, since they have no intervening
    motion or time.
    """
    if not stages:
        raise JoinContractError("at least one stage or event is required")
    seen: set[str] = set()
    boundaries: list[tuple[ActualStage, KinematicState, KinematicState]] = []
    positions: dict[str, Position9] = {}
    for item in stages:
        if not isinstance(item, ActualStage | StationaryEvent):
            raise JoinContractError("unsupported stage representation")
        if not item.label or item.label in seen:
            raise JoinContractError("stage and event labels must be unique and nonempty")
        seen.add(item.label)
        if isinstance(item, StationaryEvent):
            positions[item.label] = Position9(_vector(item.position, "event position"))
        else:
            if not isfinite(item.duration) or item.duration <= 0:
                raise JoinContractError("motion stage duration must be finite and positive")
            if not callable(item.evaluate):
                raise JoinContractError("actual curve evaluator required")
            boundaries.append((item, _state(item, 0.0), _state(item, item.duration)))

    joins: list[JoinDiagnostic] = []
    left: tuple[str | None, KinematicState] | None = (
        None if initial_state is None else (None, _validated_state(initial_state))
    )
    pending: list[StationaryEvent] = []
    stage_bounds = {item.label: (first, last) for item, first, last in boundaries}

    def emit(right: tuple[str | None, KinematicState] | None) -> None:
        if left is None and not pending:
            return
        if left is None and right is None:
            raise JoinContractError("events without motion have no join to diagnose")
        event_positions = [positions[event.label] for event in pending]
        for first, second in zip(event_positions[:-1], event_positions[1:], strict=True):
            if first != second:
                raise JoinContractError("consecutive zero-time events change nine-axis state")
        points: list[Sequence[float]] = []
        if left is not None:
            points.append(left[1].position)
        points.extend(event_positions)
        if right is not None:
            points.append(right[1].position)
        residuals = tuple(_difference(b, a) for a, b in zip(points[:-1], points[1:], strict=True))
        if not residuals:
            raise JoinContractError("join requires two positions")
        velocity = (
            None
            if left is None
            or right is None
            or left[1].velocity is None
            or right[1].velocity is None
            else _difference(right[1].velocity, left[1].velocity)
        )
        acceleration = (
            None
            if left is None
            or right is None
            or left[1].acceleration is None
            or right[1].acceleration is None
            else _difference(right[1].acceleration, left[1].acceleration)
        )
        joins.append(
            JoinDiagnostic(
                None if left is None else left[0],
                None if right is None else right[0],
                tuple(event.label for event in pending),
                _difference(points[-1], points[0]),
                residuals,
                velocity,
                acceleration,
            )
        )

    for item in stages:
        if isinstance(item, StationaryEvent):
            pending.append(item)
            continue
        first, last = stage_bounds[item.label]
        emit((item.label, first))
        left = item.label, last
        pending = []
    if final_state is not None:
        emit((None, _validated_state(final_state)))
    elif pending:
        emit(None)
    return joins


def require_join_continuity(
    joins: Sequence[JoinDiagnostic],
    position_tolerances: Sequence[float],
    velocity_tolerances: Sequence[float],
    acceleration_tolerances: Sequence[float],
) -> None:
    """Reject unknown or discontinuous joins using explicit per-axis bands.

    Tolerances use each axis's native position, velocity and acceleration
    units. This gate makes no choice about manufacturing or controller limits.
    """
    if not joins:
        raise JoinContractError("no joins to validate")
    bands = (
        _vector(position_tolerances, "position tolerances"),
        _vector(velocity_tolerances, "velocity tolerances"),
        _vector(acceleration_tolerances, "acceleration tolerances"),
    )
    if any(value < 0 for band in bands for value in band):
        raise JoinContractError("continuity tolerances must be nonnegative")
    for join in joins:
        if join.velocity_delta is None or join.acceleration_delta is None:
            raise JoinContractError(f"{join.left} to {join.right}: unknown derivatives")
        for name, deltas, band in (
            ("position", join.position_residuals, bands[0]),
            ("velocity", (join.velocity_delta,), bands[1]),
            ("acceleration", (join.acceleration_delta,), bands[2]),
        ):
            if any(
                abs(delta) > allowed
                for row in deltas
                for delta, allowed in zip(row, band, strict=True)
            ):
                raise JoinContractError(f"{join.left} to {join.right}: {name} discontinuity")
