"""Sample-independent cubic velocity extrema include acceleration roots."""

import numpy as np
from cubic_velocity_extrema import cubic_velocity_extrema
from scipy.interpolate import CubicSpline


def test_cubic_velocity_extremum_between_knots():
    times = np.array([0.0, 2.0])
    # q(t) = t^3 - 3t^2 has v(t) = 3t^2 - 6t, whose minimum is -3 at t=1.
    q = times**3 - 3 * times**2
    curve = CubicSpline(
        times, q[:, None], axis=0, bc_type=((1, np.array([0.0])), (1, np.array([0.0])))
    )
    extrema = cubic_velocity_extrema(curve)[0]
    assert abs(extrema["minimum"] + 3) < 1e-10
    assert abs(extrema["minimum_time_s"] - 1) < 1e-10
