"""Actual sampled process geometry is checked in the workpiece frame."""

import numpy as np
import pytest
from process_geometry import ProcessGeometryError, assess_process_geometry


def test_reports_order_and_departure_in_meters():
    start = np.eye(4)
    end = np.eye(4)
    end[0, 3] = 1.0
    poses = np.repeat(np.eye(4)[None, :, :], 4, axis=0)
    poses[:, 0, 3] = [0.0, 0.7, 0.5, 1.0]
    poses[1, 1, 3] = 0.02
    result = assess_process_geometry(start, end, poses)
    assert result["max_position_departure_m"] == pytest.approx(0.02)
    assert result["max_backward_distance_m"] == pytest.approx(0.2)
    assert result["progress_min"] == pytest.approx(0.0)
    assert result["progress_max"] == pytest.approx(1.0)
    assert result["max_orientation_departure_rad"] == pytest.approx(0.0)


def test_zero_length_process_line_fails_loudly():
    with pytest.raises(ProcessGeometryError):
        assess_process_geometry(np.eye(4), np.eye(4), np.repeat(np.eye(4)[None, :, :], 2, axis=0))
