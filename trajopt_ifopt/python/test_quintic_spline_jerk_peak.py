"""Native signed peak epigraph crosses the Ifopt/PIQP boundary."""

import numpy as np
from _quintic_spline import (
    QuinticSplineJerkCost,
    QuinticSplineJerkPeakConstraint,
    QuinticSplineModel,
)
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp

# Existing SQP feasibility setting; dimensionless normalized jerk.
SQP_FEASIBILITY_BAND = 1e-4


def test_native_peak_epigraph_solves_without_a_fixed_jerk_ceiling():
    times = np.array([0.0, 0.11, 0.27, 0.46, 0.69, 0.86, 1.0])
    points = np.array([[0.0], [0.2], [0.1], [0.7], [0.3], [0.8], [1.0]])
    zero = np.zeros(1)
    model = QuinticSplineModel.build(times.tolist(), zero, zero, zero, zero)
    initial_jerk = model.evaluate(times, points, 3)[3]
    nodes = []
    for index, q in enumerate(points):
        node = ti.Node(f"peak-knot-{index}")
        bounds = [ti.Bounds(float(q[0]), float(q[0]))] if index in (0, 6) else [ti.Bounds(-2, 2)]
        node.addVar("position", ["arm"], q, bounds)
        nodes.append(node)
    peak_node = ti.Node("peak-variable")
    peak_node.addVar(
        "peak", ["arm"], [float(np.max(np.abs(initial_jerk)))], [ti.Bounds(0.0, float("inf"))]
    )
    variables = ti.NodesVariables("peak-knots", [*nodes, peak_node])
    all_nodes = variables.getNodes()
    positions = [node.getVar("position") for node in all_nodes[:-1]]
    peak = all_nodes[-1].getVar("peak")
    epigraph = QuinticSplineJerkPeakConstraint(model, positions, peak, times.tolist(), np.ones(1))
    regular = QuinticSplineJerkCost(model, positions, np.array([0.1]))
    risk = ti.JointPosConstraint(np.zeros(1), peak, np.array([0.1]), "soft-peak")
    problem = sqp.TrajOptQPProblem(variables)
    problem.addConstraintSet(epigraph)
    problem.addCostSet(regular, sqp.CostPenaltyType.SQUARED)
    problem.addCostSet(risk, sqp.CostPenaltyType.SQUARED)
    problem.setup()
    solver = sqp.TrustRegionSQPSolver(sqp.PIQPSolver())
    solver.solve(problem)
    assert solver.getStatus() == sqp.SQPStatus.NLP_CONVERGED
    assert float(peak.value()[0]) >= 0.0
    assert np.max(epigraph.getValues()) <= SQP_FEASIBILITY_BAND
