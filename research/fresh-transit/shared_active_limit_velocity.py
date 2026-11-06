"""Shared velocities at active joint-position limits."""

from typing import Any

import numpy as np

# Numerical recognition of an existing bound state, not a physical tolerance.
ACTIVE_LIMIT_ULPS = 128


class SharedLimitError(RuntimeError):
    """Shared boundary position or limits are invalid."""


def shared_active_limit_velocity(position: Any, velocity: Any, limits: Any) -> Any:
    """Return shared C1 velocity, zero only on axes at an active position bound.

    For a two-sided motion that remains within a bound, an active axis must
    have zero velocity at the join. Other axes retain their supplied modeled
    velocity. Position is never clipped or altered.
    """
    q = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    bounds = np.asarray(limits, dtype=float)
    if (
        q.ndim != 1
        or v.shape != q.shape
        or bounds.shape != (len(q), 2)
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise SharedLimitError("finite position, velocity and ordered bounds required")
    lower_band = (
        ACTIVE_LIMIT_ULPS
        * np.finfo(float).eps
        * np.maximum(1.0, np.maximum(np.abs(q), np.abs(bounds[:, 0])))
    )
    upper_band = (
        ACTIVE_LIMIT_ULPS
        * np.finfo(float).eps
        * np.maximum(1.0, np.maximum(np.abs(q), np.abs(bounds[:, 1])))
    )
    if np.any(q < bounds[:, 0] - lower_band) or np.any(q > bounds[:, 1] + upper_band):
        raise SharedLimitError("shared position lies outside a joint bound")
    result = v.copy()
    result[(q - bounds[:, 0] <= lower_band) | (bounds[:, 1] - q <= upper_band)] = 0.0
    return result
