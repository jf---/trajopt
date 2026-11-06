"""Free modeled endpoint derivatives preserve the same native curve."""

import numpy as np
from _quintic_spline import QuinticSplineModel


def test_free_start_velocity_and_acceleration_match_fixed_curve():
    times = np.array([0.0, 0.2, 0.65, 1.1, 1.7, 2.3, 3.0])
    knots = np.array(
        [[0.1, -0.2], [-0.2, 0.3], [0.4, 0.2], [0.25, -0.1], [0.7, 0.5], [0.3, 0.8], [0.9, 0.2]]
    )
    v0, a0, v1, a1 = (
        np.array([0.35, -0.1]),
        np.array([-0.18, 0.23]),
        np.array([-0.22, 0.15]),
        np.array([0.27, -0.11]),
    )
    free = QuinticSplineModel.build_variable_boundary(times.tolist(), 2, None, None, v1, a1)
    fixed = QuinticSplineModel.build(times.tolist(), v0, a0, v1, a1)
    samples = np.linspace(0.0, 3.0, 31)
    actual = free.evaluate(samples, np.vstack([knots, v0, a0]), 3)
    expected = fixed.evaluate(samples, knots, 3)
    for observed, target in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, target, atol=1e-9, rtol=1e-9)
