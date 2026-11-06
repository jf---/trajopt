"""Numerical actual position extrema of vector-valued cubic process curves."""

from typing import Any

import numpy as np
from scipy.interpolate import PPoly  # type: ignore[import-untyped]


class CubicExtremaError(RuntimeError):
    """A modeled cubic has an invalid domain or coefficient layout."""


def cubic_position_extrema(curve: Any) -> list[dict[str, float]]:
    """Evaluate each axis at all piece boundaries and derivative roots."""
    knots = np.asarray(curve.x, dtype=float)
    coefficients = np.asarray(curve.c, dtype=float)
    if (
        knots.ndim != 1
        or len(knots) < 2
        or not np.isfinite(knots).all()
        or np.any(np.diff(knots) <= 0)
        or coefficients.ndim != 3
        or coefficients.shape[0] != 4
        or coefficients.shape[1] != len(knots) - 1
        or coefficients.shape[2] == 0
        or not np.isfinite(coefficients).all()
    ):
        raise CubicExtremaError("finite vector cubic with increasing time required")
    results = []
    for axis in range(coefficients.shape[2]):
        polynomial = PPoly(coefficients[:, :, axis], knots, extrapolate=False)
        roots = np.asarray(polynomial.derivative().roots(extrapolate=False), dtype=float)
        roots = roots[np.isfinite(roots) & (roots >= knots[0]) & (roots <= knots[-1])]
        candidates = np.unique(np.r_[knots, roots])
        values = np.asarray(polynomial(candidates), dtype=float)
        if not np.isfinite(values).all():
            raise CubicExtremaError("nonfinite process curve extrema")
        low, high = int(np.argmin(values)), int(np.argmax(values))
        results.append(
            {
                "minimum": float(values[low]),
                "minimum_time_s": float(candidates[low]),
                "maximum": float(values[high]),
                "maximum_time_s": float(candidates[high]),
            }
        )
    return results
