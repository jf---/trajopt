"""Allocate a fixed spline-knot budget across explicit process stages."""

import numpy as np


class KnotClockError(RuntimeError):
    """A fixed budget cannot represent the ordered positive-duration stages."""


def stage_aligned_knots(edges, count):
    """Include every stage boundary without increasing the caller's knot count."""
    times = np.asarray(edges, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < len(times)
    ):
        raise KnotClockError(
            "strictly increasing stage boundaries and sufficient integer knots required"
        )
    durations = np.diff(times)
    extra = count - len(times)
    ideal = extra * durations / durations.sum()
    allocated = np.floor(ideal).astype(int) + 1
    left = count - 1 - int(allocated.sum())
    for index in np.argsort(-(ideal - np.floor(ideal)), kind="stable")[:left]:
        allocated[index] += 1
    points = [float(times[0])]
    for i, intervals in enumerate(allocated):
        points.extend(np.linspace(times[i], times[i + 1], int(intervals) + 1)[1:].tolist())
    return np.asarray(points)
