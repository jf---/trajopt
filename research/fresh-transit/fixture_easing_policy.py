"""Opt-in retiming of ordered, collinear fixture chains on the same geometry."""

import copy
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from actual_stage_adapters import fixture_stage
from actual_stage_contract import ActualStage, Seconds
from event_aware_spline_boundaries import (
    STATIONARY_ROUNDOFF_ULPS,
    modeled_transit_boundaries,
    require_fixed_fixture_boundary_velocities,
)
from fixture_chain_clock import FixtureClockError, fixture_chain_clock

_fixture_clock: Callable[[list[dict[str, Any]], float], dict[str, Any]] = fixture_chain_clock


class FixtureEasingError(RuntimeError):
    """The fixture geometry or modeled clock cannot use this easing policy."""


def ease_fixture_chains(
    segments: list[dict[str, Any]], clock: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, ActualStage]]:
    """Ease each contiguous fixture chord; preserve all nine-axis targets in order.

    Each chain retains its total modeled duration. The existing fixture model
    only accepts a monotone straight chord; unsupported geometry fails loudly.
    Fixture speed and acceleration limits and controller timing are not assumed.
    """
    original = clock.get("stages", ())
    if len(segments) != len(original) or any(
        stage["stage"] != entry.get("stage")
        for stage, entry in zip(segments, original, strict=True)
    ):
        raise FixtureEasingError("fixture stage and clock mismatch")
    revised = copy.deepcopy(clock)
    timing = revised["stages"]
    overrides: dict[str, ActualStage] = {}
    index = 0
    while index < len(segments):
        if not segments[index]["stage"].startswith("fixture:"):
            index += 1
            continue
        end = index + 1
        while end < len(segments) and segments[end]["stage"].startswith("fixture:"):
            end += 1
        group = segments[index:end]
        durations = [float(entry["duration_s"]) for entry in timing[index:end]]
        if not all(np.isfinite(value) and value > 0 for value in durations):
            raise FixtureEasingError("fixture chain requires positive modeled durations")
        try:
            chain = _fixture_clock(group, sum(durations))
        except FixtureClockError as exc:
            raise FixtureEasingError(
                f"fixture chain {group[0]['stage']} is unsupported: {exc}"
            ) from exc
        times = chain["stage_times_s"]
        for j, stage in enumerate(group):
            start, finish = float(times[j]), float(times[j + 1])
            timing[index + j]["duration_s"] = finish - start
            timing[index + j]["clock_kind"] = "modeled_fixture_chord_easing"
            overrides[stage["stage"]] = fixture_stage(stage["stage"], chain, start, finish)
        index = end

    node_times = [0.0]
    for segment, entry in zip(segments, timing, strict=True):
        duration = float(entry["duration_s"])
        count = len(segment["states"]) - 1
        if count <= 0 or not np.isfinite(duration) or duration < 0:
            raise FixtureEasingError("valid stage samples and modeled durations required")
        entry["start_index"] = len(node_times) - 1
        if duration > 0:
            begin = node_times[-1]
            node_times.extend(float(begin + duration * j / count) for j in range(1, count + 1))
        entry["end_index"] = len(node_times) - 1
    revised["node_times_s"] = node_times
    return revised, overrides


def eased_boundary_resolver(overrides: Mapping[str, ActualStage]) -> Callable[..., Any]:
    """Use explicit modeled fixture derivatives at adjacent transit boundaries."""

    def resolve(
        segments: Any, timing: Any, index: int, *, start_velocity: Any, end_velocity: Any
    ) -> Any:
        left, right, events = modeled_transit_boundaries(
            segments,
            timing,
            index,
            interpolation="joint_linear_uniform",
            start_velocity=start_velocity,
            end_velocity=end_velocity,
        )
        values = [left, right]
        for side, direction in enumerate((-1, 1)):
            neighbor = index + direction
            while 0 <= neighbor < len(segments) and float(timing[neighbor]["duration_s"]) == 0:
                neighbor += direction
            if not 0 <= neighbor < len(segments):
                continue
            source = segments[neighbor]
            curve = overrides.get(source["stage"])
            if curve is None:
                continue
            at_end = direction < 0
            state = curve.evaluate(curve.duration if at_end else Seconds(0.0))
            expected = np.asarray(source["states"][-1 if at_end else 0], dtype=float)
            position = np.asarray(state.position, dtype=float)
            band = (
                STATIONARY_ROUNDOFF_ULPS
                * np.finfo(float).eps
                * np.maximum(1.0, np.maximum(np.abs(expected), np.abs(position)))
            )
            if np.any(np.abs(position - expected) > band) or state.velocity is None:
                raise FixtureEasingError(f"{source['stage']}: invalid actual fixture boundary")
            values[side] = np.asarray(state.velocity, dtype=float)
        require_fixed_fixture_boundary_velocities(values[0], values[1])
        return values[0], values[1], events

    return resolve
