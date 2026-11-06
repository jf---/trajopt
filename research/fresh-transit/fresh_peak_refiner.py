"""Apply a shared actual-time peak-jerk profile to each generated transit."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class FreshPeakRefinementError(RuntimeError):
    """A generated transit is missing or the common refinement failed."""


def refine_all_peak_transits(
    env: Any,
    info: Any,
    names: Sequence[str],
    stages: Sequence[dict[str, Any]],
    clock: dict[str, Any],
    speeds: Sequence[float],
    *,
    solver: Callable[..., list[dict[str, Any]]],
    start_velocity: Any,
    end_velocity: Any,
) -> list[dict[str, Any]]:
    """Apply the same solver to each transit with its original neighbors."""
    timing = clock.get("stages", ())
    if len(stages) != len(timing) or any(
        stage["stage"] != entry.get("stage") for stage, entry in zip(stages, timing, strict=True)
    ):
        raise FreshPeakRefinementError("stage and clock mismatch")
    labels = [stage["stage"] for stage in stages]
    if len(labels) != len(set(labels)):
        raise FreshPeakRefinementError("stage labels must be unique")
    results = []
    for index, stage in enumerate(stages):
        if not stage["stage"].startswith("transit:"):
            continue
        selected = [
            {
                **item,
                "stage": item["stage"]
                if j == index or not item["stage"].startswith("transit:")
                else "other-" + item["stage"],
            }
            for j, item in enumerate(stages)
        ]
        selected_clock = [
            {**entry, "stage": selected[j]["stage"]} for j, entry in enumerate(timing)
        ]

        def accelerations(selected_index: int) -> tuple[Any, Any]:
            # On each positive-duration neighboring joint-linear segment, the
            # one-sided modeled acceleration is zero; tour edges remain free.
            before = any(float(entry["duration_s"]) > 0 for entry in timing[:selected_index])
            after = any(float(entry["duration_s"]) > 0 for entry in timing[selected_index + 1 :])
            return np.zeros(9) if before else None, np.zeros(9) if after else None

        result = solver(
            env,
            info,
            names,
            selected,
            {**clock, "stages": selected_clock},
            speeds,
            start_velocity=start_velocity,
            end_velocity=end_velocity,
            acceleration_resolver=accelerations,
        )
        if len(result) != 1 or result[0].get("stage") != stage["stage"]:
            raise FreshPeakRefinementError(f"{stage['stage']}: expected one matching curve")
        results.extend(result)
    return results
