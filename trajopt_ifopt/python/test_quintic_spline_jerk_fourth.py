"""Native fourth-power jerk cost crosses the actual Ifopt/PIQP boundary."""

import numpy as np
import pytest
from _quintic_spline import (
    QuinticSplineError,
    QuinticSplineJerkCost,
    QuinticSplineJerkFourthCost,
    QuinticSplineModel,
)
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp

# Numerical integration and factorization comparison, not a physical limit.
INTEGRATION_RELATIVE_BAND = 1e-8


def test_fourth_cost_matches_reference_and_solves_native_piqp():
    times = np.array([0.0, 0.11, 0.27, 0.46, 0.69, 0.86, 1.0])
    points = np.array([[0.0], [0.2], [0.1], [0.7], [0.3], [0.8], [1.0]])
    zero = np.zeros(1)
    model = QuinticSplineModel.build(times.tolist(), zero, zero, zero, zero)
    nodes = []
    for index, q in enumerate(points):
        node = ti.Node(f"fourth-{index}")
        bounds = [ti.Bounds(float(q[0]), float(q[0]))] if index in (0, 6) else [ti.Bounds(-2, 2)]
        node.addVar("position", ["arm"], q, bounds)
        nodes.append(node)
    variables = ti.NodesVariables("fourth-knots", nodes)
    positions = [node.getVar("position") for node in variables.getNodes()]
    risk = QuinticSplineJerkFourthCost(model, positions, np.array([0.01]))
    regular = QuinticSplineJerkCost(model, positions, np.array([0.1]))
    with pytest.raises(QuinticSplineError):
        QuinticSplineJerkFourthCost(model, positions[:-1], np.array([0.01]))
    with pytest.raises(QuinticSplineError):
        QuinticSplineJerkFourthCost(model, positions, np.array([-0.01]))
    problem = sqp.TrajOptQPProblem(variables)
    problem.addCostSet(regular, sqp.CostPenaltyType.SQUARED)
    problem.addCostSet(risk, sqp.CostPenaltyType.SQUARED)
    problem.setup()
    reference = make_interp_spline(
        times, points, k=5, bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)])
    )
    numerical = (
        sum(
            quad(lambda t: float(reference(t, nu=3)[0] ** 4), a, b)[0]
            for a, b in zip(times[:-1], times[1:], strict=True)
        )
        * 0.01
    )
    actual = float(np.dot(risk.getCoefficients(), risk.getValues() ** 2))
    assert actual == pytest.approx(numerical, rel=INTEGRATION_RELATIVE_BAND)
    before = actual + float(np.dot(regular.getCoefficients(), regular.getValues() ** 2))
    solver = sqp.TrustRegionSQPSolver(sqp.PIQPSolver())
    solver.solve(problem)
    after = float(
        np.dot(risk.getCoefficients(), risk.getValues() ** 2)
        + np.dot(regular.getCoefficients(), regular.getValues() ** 2)
    )
    assert solver.getStatus() == sqp.SQPStatus.NLP_CONVERGED
    assert np.isfinite(after) and after < before
