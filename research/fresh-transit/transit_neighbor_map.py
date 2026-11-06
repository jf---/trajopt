"""Precompute shared-transit neighbors without repeated whole-tour scans."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from actual_stage_contract import ActualStage, Seconds, StationaryEvent
from event_aware_spline_boundaries import STATIONARY_EVENT, STATIONARY_ROUNDOFF_ULPS


class TransitNeighborError(RuntimeError):
    """Source stages, stationary events or actual boundaries are inconsistent."""


def _same_state(left: Any, right: Any) -> bool:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return bool(np.all(np.abs(a - b) <= STATIONARY_ROUNDOFF_ULPS * np.finfo(float).eps * scale))


def transit_neighbor_map(
    rows: Sequence[Mapping[str, Any]],
    timing: Sequence[Mapping[str, Any]],
    actual: Mapping[int, ActualStage | StationaryEvent],
) -> dict[int, tuple[ActualStage | None, ActualStage | None, tuple[int, ...]]]:
    """Validate ordered nine-axis events once and map each transit's two sides.

    Zero-time events preserve their ordered positions and never imply a stop.
    The result carries actual neighboring curves, not inferred derivatives.
    """
    if not rows or len(rows) != len(timing):
        raise TransitNeighborError("matching nonempty stages and clocks required")
    values: list[Any] = []
    for index, (row, clock) in enumerate(zip(rows, timing, strict=True)):
        q = np.asarray(row["states"], dtype=float)
        duration = float(clock["duration_s"])
        event = clock.get("clock_kind") == STATIONARY_EVENT
        if (
            row["stage"] != clock.get("stage")
            or q.ndim != 2
            or q.shape[1:] != (9,)
            or len(q) < 2
            or not np.isfinite(q).all()
            or not np.isfinite(duration)
            or duration < 0
        ):
            raise TransitNeighborError("invalid nine-axis stage or clock")
        if duration == 0:
            if not event or any(not _same_state(q[0], point) for point in q):
                raise TransitNeighborError("zero-time event must remain stationary")
        elif event:
            raise TransitNeighborError("stationary event must have zero duration")
        if index and not _same_state(values[-1][-1], q[0]):
            raise TransitNeighborError(f"{row['stage']}: nine-axis source boundary mismatch")
        values.append(q)
        if row["stage"].startswith("transit:"):
            if duration <= 0 or any(not _same_state(q[0, 7:], point[7:]) for point in q):
                raise TransitNeighborError("transit must have positive duration and fixed fixture")
        else:
            stage = actual.get(index)
            if stage is None or stage.label != row["stage"]:
                raise TransitNeighborError(f"{row['stage']}: actual stage missing")
            first = (
                stage.position
                if isinstance(stage, StationaryEvent)
                else stage.evaluate(Seconds(0)).position
            )
            last = (
                stage.position
                if isinstance(stage, StationaryEvent)
                else stage.evaluate(stage.duration).position
            )
            if not _same_state(first, q[0]) or not _same_state(last, q[-1]):
                raise TransitNeighborError(f"{row['stage']}: actual endpoint mismatch")
    result: dict[int, tuple[ActualStage | None, ActualStage | None, tuple[int, ...]]] = {}
    for i, row in enumerate(rows):
        if not row["stage"].startswith("transit:"):
            continue
        events = []
        neighbors: list[ActualStage | None] = []
        for direction in (-1, 1):
            j = i + direction
            while 0 <= j < len(rows) and float(timing[j]["duration_s"]) == 0:
                events.append(j)
                j += direction
            if j < 0 or j >= len(rows):
                neighbors.append(None)
            else:
                adjacent = actual.get(j)
                if not isinstance(adjacent, ActualStage):
                    raise TransitNeighborError(f"{row['stage']}: missing adjacent actual motion")
                neighbors.append(adjacent)
        result[i] = (neighbors[0], neighbors[1], tuple(sorted(events)))
    return result
