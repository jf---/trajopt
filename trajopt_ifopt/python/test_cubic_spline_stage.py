"""Native stage-local cubic maps reproduce an independent clamped interpolation."""

import _quintic_spline as native
import numpy as np
from scipy.interpolate import CubicSpline


def test_nonuniform_stage_reconstructs_cubic_and_analytic_jacobian():
    times = np.array([0.0, 0.19, 0.54, 1.3, 2.1])
    positions = np.array([[0.0, 0.2], [0.3, -0.4], [0.1, 0.6], [0.8, 0.0], [0.2, 0.5]])
    v0, v1 = np.array([0.5, -0.1]), np.array([-0.2, 0.4])
    reference = CubicSpline(times, positions, axis=0, bc_type=((1, v0), (1, v1)))
    stage = native.CubicSplineStage.build(times.tolist(), 2)
    variables = np.vstack((positions, v0, v1))
    samples = np.unique(np.r_[np.linspace(0.0, 2.1, 301), times])
    for order in range(4):
        mapping = stage.map(samples.tolist(), order)
        np.testing.assert_allclose(
            mapping.basis @ variables + mapping.offset,
            reference(samples, order),
            rtol=1e-10,
            atol=1e-10,
        )
        step = 1e-6
        for index in range(len(variables)):
            plus, minus = variables.copy(), variables.copy()
            plus[index, 0] += step
            minus[index, 0] -= step
            numeric = (
                (mapping.basis @ plus + mapping.offset) - (mapping.basis @ minus + mapping.offset)
            )[:, 0] / (2 * step)
            np.testing.assert_allclose(numeric, mapping.basis[:, index], rtol=1e-6, atol=1e-7)
