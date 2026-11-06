"""Source workpiece orientation targets for exploratory weld curves."""

from collections.abc import Callable
from typing import Any

import numpy as np
from rotation_targets import interpolate_rotations  # type: ignore[import-untyped]


class SourceOrientationError(RuntimeError):
    """The source orientation stages or modeled clock are invalid."""


def build_source_bead_orientation(
    rotations: Any, durations_s: Any, lead_orientation: Callable[[int, float], Any]
) -> Callable[[int, float], Any]:
    """Use source shortest-arc orientation on bead stages; retain generated leads.

    Rotations are workpiece-frame matrices; durations and sample times are
    modeled seconds. A fitted joint curve can still deviate between samples.
    """
    matrices = np.asarray(rotations, dtype=float)
    durations = np.asarray(durations_s, dtype=float)
    if (
        durations.ndim != 1
        or len(durations) < 3
        or matrices.shape != (len(durations) + 1, 3, 3)
        or not np.isfinite(matrices).all()
        or not np.isfinite(durations).all()
        or np.any(durations <= 0.0)
    ):
        raise SourceOrientationError(
            "finite workpiece rotations and positive modeled clocks required"
        )

    def orientation(stage: int, time_s: float) -> Any:
        if (
            not isinstance(stage, int)
            or stage < 0
            or stage >= len(durations)
            or not np.isfinite(time_s)
            or time_s < 0.0
            or time_s > durations[stage]
        ):
            raise SourceOrientationError("sample outside modeled weld stage")
        if stage in (0, len(durations) - 1):
            return lead_orientation(stage, time_s)
        return interpolate_rotations(
            matrices[stage], matrices[stage + 1], [time_s / durations[stage]]
        )[0]

    return orientation
