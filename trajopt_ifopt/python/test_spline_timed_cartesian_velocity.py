"""The native modeled-velocity residual reaches a real Ifopt cost consumer."""

import _quintic_spline as native
import numpy as np
import pytest
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp
from tesseract_robotics.tesseract_common import GeneralResourceLocator, Isometry3d
from tesseract_robotics.tesseract_environment import Environment
from test_spline_cartesian_bindings import SRDF, URDF


def test_timed_velocity_binding_and_squared_cost_use_workpiece_frame():
    env = Environment()
    assert env.init(URDF, SRDF, GeneralResourceLocator())
    nodes = []
    for index in range(7):
        node = ti.Node(f"knot{index}")
        node.addVar("position", ["x", "y", "yaw"], [index / 6, 0.2, 0.3], [ti.Bounds(-3, 3)] * 3)
        nodes.append(node)
    variables = ti.NodesVariables("knots", nodes)
    knots = [node.getVar("position") for node in variables.getNodes()]
    first, last = np.eye(4), np.eye(4)
    first[:2, :2] = [[0, -1], [1, 0]]
    last[:2, :2] = [[0, 1], [-1, 0]]
    last[:3, 3] = [0.4, 0.3, 0.0]
    info = ti.CartLineInfo(
        env.getJointGroup("moving"), "tcp", "base", Isometry3d(first), Isometry3d(last)
    )
    model = native.QuinticSplineModel.build(
        np.linspace(0, 3, 7).tolist(),
        np.array([1 / 3, 0, 0]),
        np.zeros(3),
        np.array([1 / 3, 0, 0]),
        np.zeros(3),
    )
    mapping = model.map([0.0, 1.5], 0)
    term = native.SplineTimedCartesianVelocity(
        info,
        first,
        last,
        knots,
        mapping.basis[0],
        mapping.offset[0],
        mapping.basis[1],
        mapping.offset[1],
        1.5,
        0.5,
        3.0,
        2.0,
    )
    term.linkWithVariables(variables)
    expected = np.array([0.1, -0.3, 0.0]) / (1.5 * 0.5)
    np.testing.assert_allclose(term.getValues(), expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(term.getCoefficients(), [1.0] * 3)
    problem = sqp.TrajOptQPProblem(variables)
    problem.addCostSet(term, sqp.CostPenaltyType.SQUARED)
    problem.setup()
    np.testing.assert_allclose(problem.getTotalExactCost(), expected @ expected, atol=1e-10)
    with pytest.raises(ValueError):
        native.SplineTimedCartesianVelocity(
            info,
            first,
            last,
            knots,
            mapping.basis[0],
            mapping.offset[0],
            mapping.basis[1],
            mapping.offset[1],
            0.0,
            0.5,
            3.0,
            2.0,
        )
