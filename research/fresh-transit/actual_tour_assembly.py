"""Assemble explicit modeled stage replacements without implying continuity."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from actual_stage_contract import ActualStage, Seconds, StationaryEvent
from verified_linear_stages import actual_linear_stage


class TourAssemblyError(RuntimeError):
    """Stage, replacement or clock does not match the generated tour."""


def assemble_actual_tour(
    segments: Sequence[dict[str, Any]],
    clock: Mapping[str, Any],
    replacements: Mapping[str, ActualStage],
) -> tuple[list[ActualStage | StationaryEvent], list[dict[str, Any]]]:
    """Preserve source order and report corners of unchanged joint-linear stages.

    Replacements use their own explicit modeled durations and must retain the
    stage label. This function never asserts boundary continuity or source
    geometry; consumers must evaluate both. Axis order is track m, six arm
    rad, positioner rad, tailstock m.
    """
    timing = clock.get("stages", ())
    labels = [segment["stage"] for segment in segments]
    if not labels or len(labels) != len(timing) or len(set(labels)) != len(labels):
        raise TourAssemblyError("unique nonempty stages and matching clock required")
    if any(label != entry.get("stage") for label, entry in zip(labels, timing, strict=True)):
        raise TourAssemblyError("stage clock labels disagree")
    if set(replacements) - set(labels):
        raise TourAssemblyError("unknown replacement stage")
    stages: list[ActualStage | StationaryEvent] = []
    corners = []
    for source, entry in zip(segments, timing, strict=True):
        label = source["stage"]
        stage: ActualStage | StationaryEvent
        if label in replacements:
            stage = replacements[label]
            if stage.label != label:
                raise TourAssemblyError(f"{label}: replacement label mismatch")
        else:
            stage = actual_linear_stage(label, source["states"], float(entry["duration_s"]))
            if isinstance(stage, ActualStage):
                times = np.linspace(0.0, stage.duration, len(source["states"]))
                for i, time in enumerate(times[1:-1], 1):
                    state = stage.evaluate(Seconds(float(time)))
                    if state.velocity is None or state.acceleration is None:
                        corners.append({"stage": label, "node": i, "modeled_time_s": float(time)})
        stages.append(stage)
    return stages, corners
