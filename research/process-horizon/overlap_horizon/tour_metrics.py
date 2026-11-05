"""Summarize finite-sampled whole-tour dynamics by axis kind."""

import numpy as np


class TourMetricsError(RuntimeError):
    """The verifier output has no valid nine-axis stage metrics."""


def summarize_tour(report):
    """Report peaks and sampled jerk integrals; no axis limits are inferred."""
    stages = report["stages"]
    if not stages:
        raise TourMetricsError("stage metrics are required")
    arrays = {}
    for field in (
        "max_velocity_by_axis",
        "max_acceleration_by_axis",
        "max_jerk_by_axis",
        "integrated_squared_jerk_estimate_by_axis",
    ):
        values = np.asarray([stage[field] for stage in stages], dtype=float)
        if values.shape != (len(stages), 9) or not np.isfinite(values).all() or np.any(values < 0):
            raise TourMetricsError(f"invalid nine-axis metrics: {field}")
        arrays[field] = values
    result = {}
    for metric, field in [
        ("speed", "max_velocity_by_axis"),
        ("acceleration", "max_acceleration_by_axis"),
        ("jerk", "max_jerk_by_axis"),
    ]:
        result[f"track_peak_{metric}"] = float(arrays[field][:, 0].max())
        result[f"arm_peak_{metric}"] = float(arrays[field][:, 1:7].max())
    integral = arrays["integrated_squared_jerk_estimate_by_axis"]
    result["track_integrated_squared_jerk"] = float(integral[:, 0].sum())
    result["arm_integrated_squared_jerk_sum"] = float(integral[:, 1:7].sum())
    geometry = [
        stage["process_geometry"] for stage in stages if stage.get("process_geometry") is not None
    ]
    if not geometry:
        raise TourMetricsError("process geometry checks are required")
    for key in (
        "max_position_departure_m",
        "max_orientation_departure_rad",
        "max_backward_distance_m",
    ):
        result[f"process_{key}"] = max(float(x[key]) for x in geometry)
    return result
