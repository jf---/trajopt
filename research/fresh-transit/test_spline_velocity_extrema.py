import numpy as np
import pytest
from scipy.interpolate import make_interp_spline
from spline_velocity_extrema import velocity_extrema


def test_velocity_extrema_include_between_node_roots_and_interval_endpoints():
    t = np.linspace(0.0, 1.0, 6)
    p = t**3 - 1.5 * t**2
    curve = make_interp_spline(t, np.column_stack([p, -p]), k=5)
    result = velocity_extrema(curve, 0.1, 0.9)
    assert result[0]["minimum"] == pytest.approx(-0.75)
    assert result[0]["minimum_time_s"] == pytest.approx(0.5)
    assert result[1]["maximum"] == pytest.approx(0.75)
    assert result[1]["maximum_time_s"] == pytest.approx(0.5)
    assert result[0]["maximum"] == pytest.approx(-0.27)
