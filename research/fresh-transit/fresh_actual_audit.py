"""Build an opt-in actual-curve audit for a freshly verified staged tour."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from actual_stage_adapters import transit_stage
from actual_stage_contract import (
    ActualStage,
    JoinContractError,
    KinematicState,
    Seconds,
    StationaryEvent,
    diagnostic_joins,
    require_join_continuity,
)
from verified_linear_stages import actual_linear_stage


def audit_stages(
    segments: Sequence[dict[str, Any]],
    clock: dict[str, Any],
    transits: Sequence[dict[str, Any]],
    make_spline: Callable[[dict[str, Any]], Any],
    *,
    initial_state: KinematicState | None = None,
    final_state: KinematicState | None = None,
) -> dict[str, Any]:
    """Diagnose modeled joins without inventing physical boundary derivatives.

    Non-transit stages retain the verified nine-axis joint-linear geometry and
    clock. Transit rows describe the actual native quintic. The caller remains
    responsible for independent collision, process and joint-limit checks.
    """
    timing = clock.get("stages", ())
    if not segments or len(segments) != len(timing):
        raise JoinContractError("matching nonempty stages and modeled clock required")
    labels = [stage["stage"] for stage in segments]
    if len(set(labels)) != len(labels) or any(
        label != entry.get("stage") for label, entry in zip(labels, timing, strict=True)
    ):
        raise JoinContractError("stage labels or modeled clocks disagree")
    expected = {label for label in labels if label.startswith("transit:")}
    rows = {row["stage"]: row for row in transits}
    if len(rows) != len(transits) or set(rows) != expected:
        raise JoinContractError("exactly one actual curve is required for each transit")
    stages: list[ActualStage | StationaryEvent] = []
    corners = []
    for source, entry in zip(segments, timing, strict=True):
        label = source["stage"]
        duration = float(entry["duration_s"])
        stage: ActualStage | StationaryEvent
        if label in rows:
            row = rows[label]
            if float(row["duration_s"]) != duration:
                raise JoinContractError(f"{label}: transit clock mismatch")
            stage = transit_stage(label, duration, make_spline(row), row["nodes"], row["fixture"])
        else:
            stage = actual_linear_stage(label, source["states"], duration)
            if isinstance(stage, ActualStage):
                count = len(source["states"]) - 1
                knot_times = np.linspace(0.0, duration, count + 1)
                for index, time in enumerate(knot_times[1:-1], 1):
                    state = stage.evaluate(Seconds(float(time)))
                    if state.velocity is None or state.acceleration is None:
                        corners.append({"stage": label, "node": index, "modeled_time_s": time})
        stages.append(stage)
    first, last = stages[0], stages[-1]
    first_position = (
        first.position
        if isinstance(first, StationaryEvent)
        else first.evaluate(Seconds(0.0)).position
    )
    last_position = (
        last.position
        if isinstance(last, StationaryEvent)
        else last.evaluate(last.duration).position
    )
    actual_initial = initial_state or KinematicState(first_position, None, None)
    actual_final = final_state or KinematicState(last_position, None, None)
    joins = diagnostic_joins(stages, initial_state=actual_initial, final_state=actual_final)
    boundary_status = (
        "supplied"
        if initial_state is not None
        and final_state is not None
        and initial_state.velocity is not None
        and initial_state.acceleration is not None
        and final_state.velocity is not None
        and final_state.acceleration is not None
        else "unknown"
    )
    missing = sum(join.velocity_delta is None or join.acceleration_delta is None for join in joins)
    return {
        "stage_count": len(stages),
        "joins": joins,
        "unresolved_corners": corners,
        "unknown_join_derivatives": missing,
        "outer_boundary_derivatives": boundary_status,
        "smoothness_gate": "unresolved" if corners or missing else "not_evaluated",
        "physical_acceptance": False,
    }


def require_smoothness(
    report: dict[str, Any],
    position_tolerances: Sequence[float],
    velocity_tolerances: Sequence[float],
    acceleration_tolerances: Sequence[float],
) -> None:
    """Require explicit tolerances and resolved modeled derivatives at all joins."""
    if report["unresolved_corners"] or report["unknown_join_derivatives"]:
        raise JoinContractError("actual trajectory has unresolved derivatives")
    require_join_continuity(
        report["joins"], position_tolerances, velocity_tolerances, acceleration_tolerances
    )
