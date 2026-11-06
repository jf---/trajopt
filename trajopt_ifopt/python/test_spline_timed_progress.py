"""Clock-tracking residual and analytic Jacobian match actual spline progress."""

import _quintic_spline as native
import numpy as np
import pytest
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics.tesseract_common import GeneralResourceLocator, Isometry3d
from tesseract_robotics.tesseract_environment import Environment
from test_spline_cartesian_bindings import SRDF, URDF


def test_timed_progress_matches_projection_and_finite_difference():
    env = Environment()
    assert env.init(URDF, SRDF, GeneralResourceLocator())
    nodes = []
    for index in range(7):
        node = ti.Node(f"knot{index}")
        node.addVar("position", ["x", "y", "yaw"], [index / 6, 0.2, 0.3], [ti.Bounds(-3, 3)] * 3)
        nodes.append(node)
    variables = ti.NodesVariables("knots", nodes)
    knots = [node.getVar("position") for node in variables.getNodes()]
    end = np.eye(4)
    end[0, 3] = 1.0
    info = ti.CartLineInfo(
        env.getJointGroup("moving"), "tcp", "base", Isometry3d.Identity(), Isometry3d(end)
    )
    model = native.QuinticSplineModel.build(
        np.linspace(0, 3, 7).tolist(),
        np.array([1 / 3, 0.0, 0.0]),
        np.zeros(3),
        np.array([1 / 3, 0.0, 0.0]),
        np.zeros(3),
    )
    mapping = model.map([0.0, 1.5, 3.0], 0)
    term = native.SplineTimedProgress(
        info, knots, mapping.basis, mapping.offset, np.array([0.0, 0.4, 1.0]), np.ones(3)
    )
    term.linkWithVariables(variables)
    np.testing.assert_allclose(term.getValues(), [0, 0.1, 0], atol=1e-10, rtol=0)
    jacobian = term.getJacobian().toarray()
    original = np.asarray(variables.getValues()).copy()
    step = 1e-6
    for index in range(original.size):
        trial = original.copy()
        trial[index] += step
        variables.setVariables(trial)
        plus = term.getValues()
        trial[index] -= 2 * step
        variables.setVariables(trial)
        minus = term.getValues()
        np.testing.assert_allclose(
            (plus - minus) / (2 * step), jacobian[:, index], atol=1e-9, rtol=0
        )
    variables.setVariables(original)
    with pytest.raises(ValueError, match="targets or coefficients"):
        native.SplineTimedProgress(
            info, knots, mapping.basis, mapping.offset, np.zeros(2), np.ones(3)
        )
