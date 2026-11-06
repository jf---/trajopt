"""Numerical extrema of actual quintic velocity on a modeled interval."""

from typing import Any

import numpy as np
from actual_stage_contract import Seconds
from scipy.interpolate import PPoly  # type: ignore[import-untyped]


class VelocityExtremaError(RuntimeError):
    """The actual spline or requested modeled interval is invalid."""


def velocity_extrema(curve: Any, start_s: Seconds, end_s: Seconds) -> list[dict[str, float]]:
    """Evaluate velocity at interval boundaries and acceleration roots.

    Times are modeled seconds; output velocity units follow the axis (track and
    tailstock metres/s; arm and positioner radians/s). Root finding is numerical,
    not an interval-arithmetic proof.
    """
    knots = np.asarray(curve.t, dtype=float)
    coefficients = np.asarray(curve.c, dtype=float)
    degree = int(curve.k)
    start, end = float(start_s), float(end_s)
    if (
        degree != 5
        or coefficients.ndim != 2
        or coefficients.shape[1] == 0
        or not np.isfinite(coefficients).all()
        or not np.isfinite(knots).all()
        or len(knots) <= 2 * degree + 1
        or np.any(np.diff(knots) < 0)
        or not np.isfinite(start)
        or not np.isfinite(end)
        or start >= end
        or start < knots[degree]
        or end > knots[-degree - 1]
    ):
        raise VelocityExtremaError("finite quintic and in-domain positive interval required")
    breaks = knots[(knots >= start) & (knots <= end)]
    result = []
    for axis in range(coefficients.shape[1]):
        polynomial = PPoly.from_spline((knots, coefficients[:, axis], degree))
        roots = np.asarray(polynomial.derivative(2).roots(extrapolate=False), dtype=float)
        roots = roots[np.isfinite(roots) & (roots >= start) & (roots <= end)]
        times = np.unique(np.r_[start, end, breaks, roots])
        values = np.asarray(polynomial.derivative()(times), dtype=float)
        if not np.isfinite(values).all():
            raise VelocityExtremaError("non-finite spline velocity at candidate extremum")
        low, high = int(np.argmin(values)), int(np.argmax(values))
        result.append(
            {
                "minimum": float(values[low]),
                "minimum_time_s": float(times[low]),
                "maximum": float(values[high]),
                "maximum_time_s": float(times[high]),
            }
        )
    return result
