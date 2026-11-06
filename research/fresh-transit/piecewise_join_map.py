"""Combine two stage-local spline derivative maps into one shared join."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


class PiecewiseJoinError(RuntimeError):
    """The stage derivative maps cannot form a single shared join."""


def combine_join_maps(left: Any, right: Any) -> NDArray[np.float64]:
    """Return left minus right for the same modeled stage boundary."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[0] != 1
        or b.shape[0] != 1
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise PiecewiseJoinError("one finite derivative-map row per stage required")
    return np.hstack((a, -b))
