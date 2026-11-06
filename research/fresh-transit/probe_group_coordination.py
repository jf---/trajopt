"""Explicit fixed-path probe join feasibility experiment."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from actual_stage_adapters import process_stage
from actual_stage_contract import ActualStage
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

# Max derivative of smoothstep 10u^3 - 15u^4 + 6u^5 on [0, 1].
STOP_TO_STOP_DURATION_FACTOR = 15.0 / 8.0
PROBE_PHASES = ("approach", "search", "retract", "withdraw")


class ProbeGroupError(RuntimeError):
    """The modeled group is not a continuous fixed-fixture probe sequence."""


def build_probe_group(
    rows: Sequence[dict[str, Any]], durations: Sequence[float]
) -> tuple[list[ActualStage], list[Any]]:
    """Use explicit modeled stops inside the probe and moving outer endpoints.

    This is a fixed-generated-path ablation, not evidence that every stop is
    physically necessary or that the modeled clock is the controller clock.
    Axis order is track m, six arm rad, positioner rad, tailstock m.
    """
    if len(rows) != len(PROBE_PHASES) or len(durations) != len(rows):
        raise ProbeGroupError("four ordered probe phases and durations required")
    if tuple(row["stage"].split(":")[0] for row in rows) != PROBE_PHASES:
        raise ProbeGroupError("expected approach, search, retract, withdrawal")
    indices = [row["stage"].split(":") for row in rows]
    if (
        any(len(item) != 2 or not item[1].isdigit() for item in indices)
        or len({item[1] for item in indices}) != 1
    ):
        raise ProbeGroupError("all phases must identify the same probe")
    positions = [np.asarray(row["states"], dtype=float) for row in rows]
    if any(
        q.ndim != 2 or q.shape[1:] != (9,) or len(q) < 2 or not np.isfinite(q).all()
        for q in positions
    ):
        raise ProbeGroupError("finite nine-axis probe knots required")
    if any(
        not np.array_equal(q[0, 7:], q[-1, 7:]) or not np.all(q[:, 7:] == q[0, 7:])
        for q in positions
    ):
        raise ProbeGroupError("probe fixture must remain fixed")
    if any(
        not np.array_equal(a[-1], b[0]) for a, b in zip(positions[:-1], positions[1:], strict=True)
    ):
        raise ProbeGroupError("probe phase endpoints must match")
    if any(not np.isfinite(t) or t <= 0 for t in durations):
        raise ProbeGroupError("positive modeled process durations required")
    curves = [CubicSpline(np.linspace(0, 1, len(q)), q[:, :7], axis=0) for q in positions]
    stages = [
        process_stage(
            str(row["stage"]),
            float(duration) * STOP_TO_STOP_DURATION_FACTOR,
            curve,
            tuple(map(float, q[0, 7:])),
            1.0 if i == 0 else 0.0,
            1.0 if i == len(rows) - 1 else 0.0,
        )
        for i, (row, duration, curve, q) in enumerate(
            zip(rows, durations, curves, positions, strict=True)
        )
    ]
    return stages, curves
