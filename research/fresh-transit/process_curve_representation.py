"""Opt-in C2 joint curve through freshly generated process knots."""

from math import isfinite
from typing import Any

import numpy as np
from actual_stage_adapters import weld_stage
from actual_stage_contract import ActualStage
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]


class ProcessCurveError(RuntimeError):
    """The generated process states cannot supply this modeled curve."""


def build_process_curve(label: str, states: Any, duration: float) -> tuple[ActualStage, Any]:
    """Interpolate generated knots without assuming a stop at either end.

    The nine-axis inputs are track m, six arm rad, positioner rad, tailstock m.
    The first seven axes vary and the fixture axes remain fixed. The original
    modeled duration defines the clock. Matching knots is not evidence that
    the interpolated Cartesian process geometry or collision checks pass.
    """
    q = np.asarray(states, dtype=float)
    if q.ndim != 2 or q.shape[1:] != (9,) or len(q) < 2 or not np.isfinite(q).all():
        raise ProcessCurveError("finite nine-axis generated knots required")
    if not isfinite(duration) or duration <= 0:
        raise ProcessCurveError("positive modeled duration required")
    if not np.all(q[:, 7:] == q[0, 7:]):
        raise ProcessCurveError("process fixture must remain fixed")
    curve = CubicSpline(np.linspace(0.0, duration, len(q)), q[:, :7], axis=0)
    stage = weld_stage(label, duration, curve, 0.0, tuple(map(float, q[0, 7:])))
    return stage, curve
