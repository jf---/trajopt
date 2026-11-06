"""Independently sample retimed fixture curves for collision and axis bounds."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from actual_stage_contract import ActualStage, Seconds
from fresh_validation import inspect_sampled_curve

_inspect_curve: Callable[[Any, Any, Any, float], dict[str, Any]] = inspect_sampled_curve

# Same per-curve independent sampling density as the native transit verifier.
FIXTURE_SAMPLES = 1025


class FixtureCurveCheckError(RuntimeError):
    """A fixture actual curve is incomplete or cannot be verified."""


def check_fixture_curves(
    curves: Mapping[str, ActualStage], checker: Any, limits: Any, feasibility_band: float
) -> list[dict[str, Any]]:
    """Check all actual nine-axis fixture states and retain modeled dynamics."""
    results = []
    for label, stage in curves.items():
        if stage.label != label:
            raise FixtureCurveCheckError("fixture curve label mismatch")
        states = [
            stage.evaluate(Seconds(float(t)))
            for t in np.linspace(0.0, stage.duration, FIXTURE_SAMPLES)
        ]
        if any(state.velocity is None or state.acceleration is None for state in states):
            raise FixtureCurveCheckError(f"{label}: modeled derivatives are unknown")
        positions = np.asarray([state.position for state in states])
        velocity = np.asarray([state.velocity for state in states])
        acceleration = np.asarray([state.acceleration for state in states])
        checked = _inspect_curve(checker, positions, limits, feasibility_band)
        results.append(
            {
                "stage": label,
                **checked,
                "peak_velocity_by_axis": np.max(np.abs(velocity), axis=0).tolist(),
                "peak_acceleration_by_axis": np.max(np.abs(acceleration), axis=0).tolist(),
                "duration_s": float(stage.duration),
            }
        )
    return results
