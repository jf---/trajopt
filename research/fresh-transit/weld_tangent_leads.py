"""Exploratory tangent leads adjoining an unchanged modeled weld."""

from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline  # type: ignore[import-untyped]


class WeldLeadError(RuntimeError):
    """The modeled bead or derivative-constrained lead is invalid."""


def tangent_lead_targets(first: Any, last: Any, length_m: float, count: int) -> tuple[Any, Any]:
    """Extend the workpiece-frame bead direction with constant endpoint orientations.

    These generated leads are a geometric hypothesis, not evidence of a valid
    torch approach, collision clearance or process acceptance. Position uses
    metres. The target bead itself is not changed.
    """
    a, b = np.asarray(first, dtype=float), np.asarray(last, dtype=float)
    if (
        a.shape != (4, 4)
        or b.shape != (4, 4)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise WeldLeadError("finite workpiece-frame endpoint transforms required")
    if not np.isfinite(length_m) or length_m <= 0 or count < 2:
        raise WeldLeadError("positive lead length and at least two samples required")
    direction = b[:3, 3] - a[:3, 3]
    length = float(np.linalg.norm(direction))
    if length == 0:
        raise WeldLeadError("nonzero bead length required")
    unit = direction / length
    progress = np.linspace(0.0, 1.0, count)
    before, after = np.tile(a, (count, 1, 1)), np.tile(b, (count, 1, 1))
    before[:, :3, 3] = a[:3, 3] - (1.0 - progress[:, None]) * length_m * unit
    after[:, :3, 3] = b[:3, 3] + progress[:, None] * length_m * unit
    return before, after


def fit_lead(times: Any, positions: Any, start: tuple[Any, Any], end: tuple[Any, Any]) -> Any:
    """Interpolate generated joint samples while matching supplied qdot and qddot.

    Axis order is track m and six arm rad; time is modeled seconds. The output
    must still be checked between nodes against Cartesian targets and the scene.
    """
    t, q = np.asarray(times, dtype=float), np.asarray(positions, dtype=float)
    if t.ndim != 1 or len(t) < 4 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise WeldLeadError("at least four finite, increasing times required")
    if q.shape != (len(t), 7) or not np.isfinite(q).all():
        raise WeldLeadError("seven-axis finite generated samples required")
    boundaries = []
    for velocity, acceleration in (start, end):
        v, a = np.broadcast_to(velocity, (7,)).copy(), np.broadcast_to(acceleration, (7,)).copy()
        if not np.isfinite(v).all() or not np.isfinite(a).all():
            raise WeldLeadError("finite boundary derivatives required")
        boundaries.append([(1, v), (2, a)])
    return make_interp_spline(t, q, k=5, axis=0, bc_type=tuple(boundaries))
