"""Numerical extrema of actual stage-local cubic velocity."""

from typing import Any

import numpy as np
from scipy.interpolate import PPoly  # type: ignore[import-untyped]


class CubicVelocityExtremaError(RuntimeError):
    """The modeled cubic clock or coefficient layout is invalid."""


def cubic_velocity_extrema(curve: Any) -> list[dict[str, float]]:
    """Evaluate each axis at stage boundaries and acceleration roots.

    Modeled times are seconds; velocity units follow each axis. Numerical
    polynomial roots are not an interval-arithmetic certificate.
    """
    knots, coefficients = np.asarray(curve.x, dtype=float), np.asarray(curve.c, dtype=float)
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
        raise CubicVelocityExtremaError("finite vector cubic with increasing clock required")
    results = []
    for axis in range(coefficients.shape[2]):
        polynomial = PPoly(coefficients[:, :, axis], knots, extrapolate=False)
        roots = np.asarray(polynomial.derivative(2).roots(extrapolate=False), dtype=float)
        roots = roots[np.isfinite(roots) & (roots >= knots[0]) & (roots <= knots[-1])]
        candidates = np.unique(np.r_[knots, roots])
        values = np.asarray(polynomial.derivative()(candidates), dtype=float)
        if not np.isfinite(values).all():
            raise CubicVelocityExtremaError("nonfinite process velocity extrema")
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
