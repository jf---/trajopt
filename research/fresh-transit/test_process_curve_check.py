"""Actual process-curve checks inspect Cartesian geometry and sampled collisions."""

from types import SimpleNamespace

import numpy as np
import pytest
from fresh_validation import CurveCollisionError
from process_curve_check import inspect_process_curve


def source(deviation=False):
    q = np.zeros((5, 9))
    q[:, 0] = np.linspace(0, 1, 5)
    if deviation:
        q[:, 1] = [0, 0.02, 0.04, 0.02, 0]
    return q


def pose(q):
    out = np.eye(4)
    out[:2, 3] = q[:2]
    return out


def test_reports_real_cartesian_error_and_modeled_feed():
    checker = SimpleNamespace(contacts=lambda _q: [])
    result = inspect_process_curve(
        "search:0", source(), 1.0, checker, np.tile([-2.0, 2.0], (9, 1)), np.ones(7), pose, 1e-4
    )
    assert result["sample_count"] == 1025
    assert result["fitted"]["max_position_departure_m"] == pytest.approx(0, abs=1e-10)
    assert result["fitted"]["max_backward_distance_m"] == pytest.approx(0, abs=1e-10)
    assert result["fitted"]["commanded_feed_m_s"] == pytest.approx(1.0)
    drift = inspect_process_curve(
        "search:0", source(True), 1.0, checker, np.tile([-2.0, 2.0], (9, 1)), np.ones(7), pose, 1e-4
    )
    assert drift["fitted"]["max_position_departure_m"] >= 0.04


def test_actual_spline_collision_is_not_hidden_by_safe_endpoints():
    checker = SimpleNamespace(
        contacts=lambda q: [SimpleNamespace(distance_m=-0.01)] if 0.4 < q[0] < 0.6 else []
    )
    with pytest.raises(CurveCollisionError):
        inspect_process_curve(
            "search:0", source(), 1.0, checker, np.tile([-2.0, 2.0], (9, 1)), np.ones(7), pose, 1e-4
        )


def test_axis_speed_result_reports_a_real_excess():
    checker = SimpleNamespace(contacts=lambda _q: [])
    speeds = np.ones(7)
    speeds[0] = 0.5
    result = inspect_process_curve(
        "search:0", source(), 1.0, checker, np.tile([-2.0, 2.0], (9, 1)), speeds, pose, 1e-4
    )
    assert not result["sampled_sourced_speeds_pass"]
    assert result["sampled_speed_ratio_by_axis"][0] == pytest.approx(2.0)
