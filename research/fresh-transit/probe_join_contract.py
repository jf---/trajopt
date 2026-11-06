"""Separate shared modeled probe boundaries from numerical curve evaluation."""

from typing import Any

import numpy as np

# Binary64 evaluation allowance for the existing quintic interpolation machinery.
EVALUATION_ROUNDOFF_ULPS = 32768


class ProbeJoinError(RuntimeError):
    """Canonical shared derivatives or independent evaluation differ materially."""


def inspect_probe_joins(
    phases: list[dict[str, Any]], joins: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate canonical q/qd/qdd and record scale-aware evaluation residuals.

    Seven-axis positions use track metres and arm radians, with modeled seconds
    for derivatives. The numerical band scales with knot spacing and coefficient
    magnitude; it is an arithmetic allowance, not a physical tolerance.
    """
    if len(phases) < 2 or len(joins) != len(phases) - 1:
        raise ProbeJoinError("matching ordered process phases and joins required")
    result = []
    for left, right, join in zip(phases[:-1], phases[1:], joins, strict=True):
        left_q, right_q = (
            np.asarray(left["ik_states"], dtype=float),
            np.asarray(right["ik_states"], dtype=float),
        )
        if (
            left_q.ndim != 2
            or right_q.ndim != 2
            or left_q.shape[1:] != (7,)
            or right_q.shape[1:] != (7,)
            or min(len(left_q), len(right_q)) < 2
        ):
            raise ProbeJoinError("finite seven-axis process knots required")
        if (
            not np.array_equal(left_q[-1], right_q[0])
            or not np.array_equal(np.asarray(left["v1"]), np.asarray(right["v0"]))
            or not np.array_equal(np.asarray(left["a1"]), np.asarray(right["a0"]))
        ):
            raise ProbeJoinError("canonical q, qd or qdd boundary differs")
        spacing = min(
            float(left["duration_s"]) / (len(left_q) - 1),
            float(right["duration_s"]) / (len(right_q) - 1),
        )
        if (
            not np.isfinite(spacing)
            or spacing <= 0
            or not np.isfinite(left_q).all()
            or not np.isfinite(right_q).all()
        ):
            raise ProbeJoinError("finite process knots and positive modeled spacing required")
        velocity = max(float(np.max(np.abs(left["v1"]))), float(np.max(np.abs(right["v0"]))))
        acceleration = max(float(np.max(np.abs(left["a1"]))), float(np.max(np.abs(right["a0"]))))
        scale = max(
            1.0,
            float(np.max(np.abs(left_q))),
            float(np.max(np.abs(right_q))),
            velocity * spacing,
            acceleration * spacing**2,
        )
        bands = [
            EVALUATION_ROUNDOFF_ULPS * np.finfo(float).eps * scale / min(1.0, spacing) ** order
            for order in range(3)
        ]
        fields = ("position_delta", "velocity_delta", "acceleration_delta")
        raw = [join.get(field) for field in fields]
        if any(
            value is None
            or not np.isfinite(np.asarray(value)).all()
            or np.max(np.abs(value)) > band
            for value, band in zip(raw, bands, strict=True)
        ):
            raise ProbeJoinError(
                "independent boundary evaluation exceeds conditioned roundoff band"
            )
        result.append(
            {
                "left": left["stage"],
                "right": right["stage"],
                "canonical_shared": True,
                "raw_position_delta": raw[0],
                "raw_velocity_delta": raw[1],
                "raw_acceleration_delta": raw[2],
                "evaluation_bands": bands,
            }
        )
    return result
