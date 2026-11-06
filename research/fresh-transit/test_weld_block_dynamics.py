"""Exact cubic-segment jerk integrals and derivative peaks on the modeled clock."""

import numpy as np
from scipy.interpolate import CubicSpline
from weld_block_dynamics import block_dynamics


def test_known_cubic_has_correct_peak_and_jerk_integral():
    times = np.array([0.0, 1.0, 2.0])
    values = np.column_stack((times**3, 2 * times**3))
    curve = CubicSpline(times, values, axis=0, bc_type=((1, [0.0, 0.0]), (1, [12.0, 24.0])))
    result = block_dynamics([curve], np.array([2.0, 4.0]))
    np.testing.assert_allclose(result["peak_velocity"], [12.0, 24.0])
    np.testing.assert_allclose(result["peak_acceleration"], [12.0, 24.0])
    np.testing.assert_allclose(result["peak_jerk"], [6.0, 12.0])
    np.testing.assert_allclose(result["integrated_jerk_squared"], [72.0, 288.0])
    np.testing.assert_allclose(result["integrated_speed_normalized_jerk_squared"], [18.0, 18.0])
