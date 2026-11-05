"""The numerical speed witnesses must evaluate on the native actual spline."""

import _quintic_spline as native
import numpy as np
from scipy.interpolate import make_interp_spline
from spline_speed_extrema import speed_extrema

# Roundoff budget for independently factored quintic interpolation.
PARITY_BAND = 4096 * np.finfo(float).eps


def test_native_and_piecewise_polynomial_agree_at_speed_witnesses():
    times = np.array([0.0, 0.11, 0.27, 0.46, 0.69, 0.86, 1.0])
    nodes = np.column_stack((times**3 - times, 0.4 * times**5 + times**2))
    v0 = np.array([-1.0, 0.0])
    a0 = np.array([0.0, 2.0])
    v1 = np.array([2.0, 4.0])
    a1 = np.array([6.0, 10.0])
    curve = make_interp_spline(times, nodes, k=5, bc_type=([(1, v0), (2, a0)], [(1, v1), (2, a1)]))
    extrema = speed_extrema(curve)
    witnesses = np.array([item["time_s"] for item in extrema])
    model = native.QuinticSplineModel.build(times.tolist(), v0, a0, v1, a1)
    _, velocities = model.evaluate(witnesses, nodes, 1)
    np.testing.assert_allclose(velocities, curve(witnesses, 1), rtol=PARITY_BAND, atol=PARITY_BAND)
    for axis, item in enumerate(extrema):
        np.testing.assert_allclose(
            abs(velocities[axis, axis]), item["absolute_speed"], rtol=PARITY_BAND, atol=PARITY_BAND
        )
