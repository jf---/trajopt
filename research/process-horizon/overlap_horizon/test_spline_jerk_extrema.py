"""Peak jerk witnesses include interior fourth-derivative roots."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy.interpolate import make_interp_spline
from spline_speed_extrema import jerk_extrema


def test_interior_jerk_peak_on_nonuniform_clock():
    times = np.array([0.0, 0.11, 0.27, 0.46, 0.69, 0.86, 1.0])
    polynomial = Polynomial([0.0, 0.0, 0.0, 0.91 / 6.0, 0.6 / 24.0, -1.0 / 60.0])
    nodes = polynomial(times)[:, None]
    v0, v1 = polynomial.deriv()(0), polynomial.deriv()(1)
    a0, a1 = polynomial.deriv(2)(0), polynomial.deriv(2)(1)
    curve = make_interp_spline(
        times, nodes, k=5, bc_type=([(1, [v0]), (2, [a0])], [(1, [v1]), (2, [a1])])
    )
    peak = jerk_extrema(curve)[0]
    assert peak["absolute_jerk"] == pytest.approx(1.0)
    assert peak["time_s"] == pytest.approx(0.3)
