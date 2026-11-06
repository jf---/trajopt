"""Validate new generated transit endpoints against the regenerated lead."""

from typing import Any

import numpy as np


class WeldTransitError(RuntimeError):
    """A generated route does not meet its nine-axis connection contract."""


def attach_fixture_to_route(label: str, route: Any, first: Any, last: Any) -> dict[str, Any]:
    """Carry the fixed fixture along a checked seven-axis generated route.

    Axis order is track m, six arm rad, positioner rad, tailstock m. The
    endpoints must be the actual regenerated nine-axis boundary states.
    """
    points, a, b = (
        np.asarray(route, dtype=float),
        np.asarray(first, dtype=float),
        np.asarray(last, dtype=float),
    )
    if (
        not label.startswith("transit:")
        or points.ndim != 2
        or points.shape[1:] != (7,)
        or len(points) < 2
    ):
        raise WeldTransitError("named seven-axis transit route required")
    if a.shape != (9,) or b.shape != (9,) or not all(np.isfinite(x).all() for x in (points, a, b)):
        raise WeldTransitError("finite nine-axis endpoints and route required")
    if not np.array_equal(a[7:], b[7:]):
        raise WeldTransitError("transit fixture changes")
    if not np.array_equal(points[0], a[:7]) or not np.array_equal(points[-1], b[:7]):
        raise WeldTransitError("generated route endpoint differs from the actual new endpoint")
    return {
        "stage": label,
        "states": np.column_stack((points, np.tile(a[7:], (len(points), 1)))).tolist(),
    }
