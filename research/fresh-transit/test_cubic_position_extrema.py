"""Actual cubic extrema expose limit overshoot between interpolation nodes."""

import numpy as np
import pytest
from cubic_position_extrema import cubic_position_extrema
from scipy.interpolate import CubicSpline


def test_interior_extremum_is_not_hidden_by_safe_knots():
    t = np.array([0.0, 0.4, 0.6, 1.0])
    q = np.zeros((4, 7))
    q[:, 0] = t * (1 - t)
    extrema = cubic_position_extrema(CubicSpline(t, q, axis=0))
    assert np.max(q[:, 0]) == pytest.approx(0.24)
    assert extrema[0]["maximum"] == pytest.approx(0.25)
    assert extrema[0]["maximum_time_s"] == pytest.approx(0.5)


def test_each_axis_has_own_actual_extrema():
    t = np.array([0.0, 0.4, 0.6, 1.0])
    q = np.tile(t[:, None], (1, 7))
    q[:, 2] = -q[:, 2]
    extrema = cubic_position_extrema(CubicSpline(t, q, axis=0))
    assert len(extrema) == 7
    assert extrema[2]["minimum"] == pytest.approx(-1.0)
    assert extrema[1]["maximum"] == pytest.approx(1.0)
