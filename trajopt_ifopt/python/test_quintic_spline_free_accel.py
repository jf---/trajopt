"""Free endpoint acceleration crosses the native model and Ifopt boundary."""

import numpy as np
from _quintic_spline import QuinticSplineJerkCost, QuinticSplineModel
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp


def test_free_start_acceleration_matches_fixed_and_optimizes():
    times = np.array([0.0, 0.11, 0.27, 0.46, 0.69, 0.86, 1.0])
    q = np.array([[0.0], [0.2], [0.1], [0.7], [0.3], [0.8], [1.0]])
    zero = np.zeros(1)
    accel = np.array([0.6])
    fixed = QuinticSplineModel.build(times.tolist(), zero, accel, zero, zero)
    free = QuinticSplineModel.build_variable_acceleration(times.tolist(), zero, None, zero, zero)
    variables = np.vstack((q, accel))
    assert free.knot_count == 7 and free.variable_count == 8
    for order in range(4):
        assert np.allclose(
            free.evaluate(times, variables, order)[order],
            fixed.evaluate(times, q, order)[order],
            atol=1e-9,
            rtol=1e-9,
        )
    nodes = []
    for i, value in enumerate(variables):
        node = ti.Node(f"variable-{i}")
        bounds = (
            [ti.Bounds(float(value[0]), float(value[0]))] if i in (0, 6) else [ti.Bounds(-10, 10)]
        )
        node.addVar("position", ["joint"], value, bounds)
        nodes.append(node)
    packed = ti.NodesVariables("free-accel", nodes)
    positions = [x.getVar("position") for x in packed.getNodes()]
    cost = QuinticSplineJerkCost(free, positions, np.ones(1))
    problem = sqp.TrajOptQPProblem(packed)
    problem.addCostSet(cost, sqp.CostPenaltyType.SQUARED)
    problem.setup()
    solver = sqp.TrustRegionSQPSolver(sqp.PIQPSolver())
    solver.solve(problem)
    assert solver.getStatus() == sqp.SQPStatus.NLP_CONVERGED
    assert np.isfinite(positions[-1].value()[0])
