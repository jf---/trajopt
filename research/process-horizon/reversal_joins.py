"""Identify process-line reversals without introducing stops elsewhere."""

import numpy as np

# Binary64 roundoff allowance for coincident points and opposite directions.
_GEOMETRY_ROUNDOFF_ULPS = 128


class ReversalGeometryError(RuntimeError):
    """Process line coordinates are nonfinite, degenerate, or malformed."""


def opposing_joins(legs):
    """Return indices of legs followed by a coincident, oppositely directed leg."""
    segments = []
    epsilon = _GEOMETRY_ROUNDOFF_ULPS * np.finfo(float).eps
    for start, end in legs:
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if (
            a.shape != (3,)
            or b.shape != (3,)
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            raise ReversalGeometryError("finite 3D process line endpoints required")
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length == 0:
            raise ReversalGeometryError("process line must have nonzero length")
        segments.append((a, b, direction / length))
    result = []
    for index, (left, right) in enumerate(zip(segments, segments[1:])):
        scale = max(1.0, float(np.linalg.norm(left[1])), float(np.linalg.norm(right[0])))
        if (
            np.linalg.norm(left[1] - right[0]) <= epsilon * scale
            and np.linalg.norm(left[2] + right[2]) <= epsilon
        ):
            result.append(index)
    return result
