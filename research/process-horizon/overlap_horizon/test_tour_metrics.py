"""Whole-tour aggregation preserves axis identity and sampled nature."""

import pytest
from tour_metrics import TourMetricsError, summarize_tour


def test_summarize_tour_separates_track_and_robot_joint_dynamics():
    stage = {
        "max_velocity_by_axis": [2.0] + [3.0] * 6 + [0.0] * 2,
        "max_acceleration_by_axis": [4.0] + [5.0] * 6 + [0.0] * 2,
        "max_jerk_by_axis": [6.0] + [7.0] * 6 + [0.0] * 2,
        "integrated_squared_jerk_estimate_by_axis": [8.0] + [9.0] * 6 + [0.0] * 2,
        "process_geometry": {
            "max_position_departure_m": 0.001,
            "max_orientation_departure_rad": 0.002,
            "max_backward_distance_m": 0.0,
        },
    }
    result = summarize_tour({"stages": [stage, stage]})
    assert result["track_peak_jerk"] == 6.0
    assert result["arm_peak_jerk"] == 7.0
    assert result["track_integrated_squared_jerk"] == 16.0
    assert result["arm_integrated_squared_jerk_sum"] == 108.0
    assert result["process_max_position_departure_m"] == 0.001


def test_summarize_rejects_missing_stages():
    with pytest.raises(TourMetricsError):
        summarize_tour({"stages": []})
