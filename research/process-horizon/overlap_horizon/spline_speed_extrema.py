"""Numerical velocity extrema of an actual vector-valued quintic spline."""

import numpy as np
from scipy.interpolate import BSpline, PPoly


class SplineSpeedError(RuntimeError):
    """The provided spline is not a finite, positive-duration vector quintic."""


def speed_extrema(curve):
    """Find axiswise absolute speed maxima at piece boundaries and acceleration roots.

    Polynomial roots are numerical, so this is not an interval proof. The input
    clock is seconds; axis units follow the spline positions per second.
    """
    if not isinstance(curve, BSpline):
        raise SplineSpeedError("a vector-valued BSpline is required")
    knots = np.asarray(curve.t, dtype=float)
    coefficients = np.asarray(curve.c, dtype=float)
    if (
        curve.k != 5
        or coefficients.ndim != 2
        or coefficients.shape[1] == 0
        or not np.isfinite(coefficients).all()
        or not np.isfinite(knots).all()
        or len(knots) <= 2 * curve.k + 1
        or np.any(np.diff(knots) < 0)
        or knots[curve.k] >= knots[-curve.k - 1]
    ):
        raise SplineSpeedError("finite vector quintic with positive duration required")
    first, last = knots[curve.k], knots[-curve.k - 1]
    breaks = knots[(knots >= first) & (knots <= last)]
    extrema = []
    for axis in range(coefficients.shape[1]):
        poly = PPoly.from_spline((knots, coefficients[:, axis], curve.k))
        roots = np.asarray(poly.derivative(2).roots(extrapolate=False), dtype=float)
        roots = roots[np.isfinite(roots) & (roots >= first) & (roots <= last)]
        candidates = np.unique(np.r_[first, last, breaks, roots])
        values = np.asarray(poly.derivative()(candidates), dtype=float)
        if not np.isfinite(values).all():
            raise SplineSpeedError("nonfinite velocity at polynomial extrema")
        index = int(np.argmax(np.abs(values)))
        extrema.append(
            {
                "absolute_speed": float(abs(values[index])),
                "signed_speed": float(values[index]),
                "time_s": float(candidates[index]),
            }
        )
    return extrema
