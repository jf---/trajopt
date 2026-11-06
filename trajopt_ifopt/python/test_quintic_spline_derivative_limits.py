"""Native derivative bounds must reflect actual spline velocity and variable axes."""

import numpy as np
from _quintic_spline import QuinticSplineDerivativeLimits, QuinticSplineModel
from tesseract_robotics import trajopt_ifopt as ti


def test_sampled_velocity_bounds_use_native_derivative_map():
    times = np.array([0.0, 0.2, 0.5, 0.9, 1.4, 2.0])
    samples = np.array([0.1, 0.7, 1.8])
    q = np.column_stack((times, -times))
    model = QuinticSplineModel.build(
        times.tolist(), np.array([1.0, -1.0]), np.zeros(2), np.array([1.0, -1.0]), np.zeros(2)
    )
    nodes = []
    for i, state in enumerate(q):
        node = ti.Node(f"velocity-{i}")
        node.addVar("position", ["track", "positioner"], state, [ti.Bounds(-10.0, 10.0)] * 2)
        nodes.append(node)
    variables = ti.NodesVariables("velocity-test", nodes)
    knots = [node.getVar("position") for node in variables.getNodes()]
    bounds = [
        bound for _ in samples for bound in (ti.Bounds(-2.0, 2.0), ti.Bounds(-float("inf"), 0.0))
    ]
    term = QuinticSplineDerivativeLimits(model, knots, samples.tolist(), bounds)
    term.linkWithVariables(variables)
    expected = model.evaluate(samples, q, 1)[1].reshape(-1)
    assert np.allclose(term.getValues(), np.r_[expected[::2], expected])
    assert np.allclose(expected, np.tile([1.0, -1.0], len(samples)))
