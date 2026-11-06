"""Shared modeled-time collocation for generated weld process legs."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class WeldProgressGridError(RuntimeError):
    """Modeled process durations or the common witness count are invalid."""


CLOCK_ENDPOINT_ULPS = 256  # Binary64 accumulation/subtraction roundoff on modeled seconds.


def local_process_times(
    times_s: Sequence[float], start_s: float, duration_s: float
) -> NDArray[np.float64]:
    """Convert global modeled times using canonical exact stage endpoints."""
    times = np.asarray(times_s, dtype=float)
    band = (
        CLOCK_ENDPOINT_ULPS
        * np.finfo(float).eps
        * max(1.0, abs(start_s), abs(start_s + duration_s))
    )
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or not np.isfinite(start_s)
        or not np.isfinite(duration_s)
        or duration_s <= 0
        or np.any(np.diff(times) <= 0)
        or abs(times[0] - start_s) > band
        or abs(times[-1] - (start_s + duration_s)) > band
    ):
        raise WeldProgressGridError("finite ordered times must span the supplied modeled stage")
    local = times - start_s
    local[0], local[-1] = 0.0, duration_s
    if np.any(local[1:-1] <= 0) or np.any(local[1:-1] >= duration_s):
        raise WeldProgressGridError("interior sample lies outside the modeled stage")
    return local


def witness_grid(
    durations_s: Sequence[float], count: int
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Return ordered stage boundaries and per-stage seconds, including endpoints."""
    durations = np.asarray(durations_s, dtype=float)
    if (
        durations.ndim != 1
        or len(durations) == 0
        or not np.isfinite(durations).all()
        or np.any(durations <= 0)
        or count < 2
    ):
        raise WeldProgressGridError("positive finite durations and at least two witnesses required")
    edges = np.r_[0.0, np.cumsum(durations)]
    return edges, [np.linspace(edges[i], edges[i + 1], count) for i in range(len(durations))]


def process_knot_witnesses(
    samples_s: Sequence[float], knots_s: Sequence[float]
) -> NDArray[np.float64]:
    """Add existing curve knot times to process checks without adding variables."""
    samples, knots = np.asarray(samples_s, dtype=float), np.asarray(knots_s, dtype=float)
    if (
        samples.ndim != 1
        or knots.ndim != 1
        or len(samples) < 2
        or len(knots) < 2
        or not np.isfinite(samples).all()
        or not np.isfinite(knots).all()
        or np.any(np.diff(samples) <= 0)
        or np.any(np.diff(knots) <= 0)
        or knots[0] < samples[0]
        or knots[-1] > samples[-1]
    ):
        raise WeldProgressGridError("ordered knot and sample times must share a stage")
    return np.unique(np.r_[samples, knots])
