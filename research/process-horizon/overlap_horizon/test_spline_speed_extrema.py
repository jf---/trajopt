"""Numerical extrema must cover interior peaks and both endpoints per axis."""

import numpy as np
import pytest
from scipy.interpolate import make_interp_spline
from spline_speed_extrema import SplineSpeedError, speed_extrema


def test_quintic_velocity_extrema_include_unsampled_interior_peak():
    t = np.linspace(0.0, 1.0, 7)
    x = t - (t - 0.5) ** 3 / 3.0
    y = t**5
    data = np.column_stack((x, y))
    v0 = [0.75, 0.0]
    a0 = [1.0, 0.0]
    v1 = [0.75, 5.0]
    a1 = [-1.0, 20.0]
    spline = make_interp_spline(t, data, k=5, bc_type=([(1, v0), (2, a0)], [(1, v1), (2, a1)]))
    extrema = speed_extrema(spline)
    assert extrema[0]["absolute_speed"] == pytest.approx(1.0)
    assert extrema[0]["time_s"] == pytest.approx(0.5)
    assert extrema[1]["absolute_speed"] == pytest.approx(5.0)
    assert extrema[1]["time_s"] == pytest.approx(1.0)


def test_invalid_curve_is_rejected():
    with pytest.raises(SplineSpeedError):
        speed_extrema(object())
