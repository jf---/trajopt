"""Position collocation witnesses on an actual modeled spline clock."""

from typing import Any

import numpy as np
from actual_stage_contract import Seconds


class PositionWitnessError(RuntimeError):
    """Position witness clock is invalid."""


def merge_position_witness_times(base: Any, witnesses: Any, duration_s: Seconds) -> list[float]:
    """Retain original samples and add in-domain extrema on a clock in seconds."""
    original = np.asarray(base, dtype=float)
    additions = np.asarray(witnesses, dtype=float)
    duration = float(duration_s)
    if (
        original.ndim != 1
        or additions.ndim != 1
        or len(original) == 0
        or not np.isfinite(original).all()
        or not np.isfinite(additions).all()
        or not np.isfinite(duration)
        or duration <= 0.0
        or np.any(original < 0.0)
        or np.any(original > duration)
        or np.any(additions < 0.0)
        or np.any(additions > duration)
    ):
        raise PositionWitnessError("finite in-domain position sample times required")
    return [float(value) for value in np.unique(np.r_[original, additions])]
