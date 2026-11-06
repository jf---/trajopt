"""Reconstruct stage-local generated weld curves without refitting a cluster."""

from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

CLOCK_ULPS = 256  # Binary64 accumulation roundoff in modeled seconds.


class StageLocalWeldError(RuntimeError):
    """Stage records or their modeled clocks are inconsistent."""


def stage_local_curves(
    records: list[dict[str, Any]], durations_s: list[float], *, clock_start_s: float = 0.0
) -> list[Any]:
    """Reconstruct each cubic on its original shared modeled clock.

    Each curve has track metres followed by six arm radians. The caller retains
    the fixture separately and must check the actual q/qd/qdd joins and scene.
    """
    durations = np.asarray(durations_s, dtype=float)
    if (
        len(records) == 0
        or not np.isfinite(clock_start_s)
        or durations.shape != (len(records),)
        or not np.isfinite(durations).all()
        or np.any(durations <= 0)
    ):
        raise StageLocalWeldError("positive finite clock required for every stage")
    edges = clock_start_s + np.r_[0.0, np.cumsum(durations)]
    curves = []
    for i, record in enumerate(records):
        knots = np.asarray(record["knots_s"], dtype=float)
        nodes = np.asarray(record["nodes"], dtype=float)
        v0, v1 = np.asarray(record["v0"], dtype=float), np.asarray(record["v1"], dtype=float)
        band = CLOCK_ULPS * np.finfo(float).eps * max(1.0, abs(edges[i]), abs(edges[i + 1]))
        if (
            knots.ndim != 1
            or len(knots) < 2
            or nodes.shape != (len(knots), 7)
            or v0.shape != (7,)
            or v1.shape != (7,)
            or not np.isfinite(knots).all()
            or not np.isfinite(nodes).all()
            or not np.isfinite(v0).all()
            or not np.isfinite(v1).all()
            or np.any(np.diff(knots) <= 0)
            or abs(knots[0] - edges[i]) > band
            or abs(knots[-1] - edges[i + 1]) > band
        ):
            raise StageLocalWeldError(f"invalid actual curve or modeled clock for stage {i}")
        curves.append(CubicSpline(knots, nodes, axis=0, bc_type=((1, v0), (1, v1))))
    return curves


def optimization_window(stage_kinds: list[str], mode: str) -> tuple[int, int]:
    """Select an entire cluster or its consecutive source-bead stages."""
    if (
        len(stage_kinds) < 3
        or stage_kinds[0] != "generated_lead"
        or stage_kinds[-1] != "generated_lead"
        or any(kind != "source_bead" for kind in stage_kinds[1:-1])
    ):
        raise StageLocalWeldError("ordered generated lead, bead and lead stages required")
    if mode == "full":
        return 0, len(stage_kinds)
    if mode == "bead":
        return 1, len(stage_kinds) - 1
    raise StageLocalWeldError("unknown optimization window mode")
