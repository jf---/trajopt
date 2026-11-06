"""Trace records distinguish individual dynamic collision row sets."""

import _quintic_spline as native
import numpy as np
from piecewise_sqp_diagnostic import collision_row_counts
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp


class CollisionRows:
    def __init__(self, name, rows):
        self.name = name
        self.rows = rows

    def getName(self):
        return self.name

    def getRows(self):
        return self.rows


def test_collision_rows_are_recorded_without_updating_constraints():
    assert collision_row_counts([CollisionRows("a", 3), CollisionRows("b", 0)]) == {
        "a": 3,
        "b": 0,
    }


def test_native_qp_statistics_are_available_at_callback():
    nodes = ti.createNodesVariables(
        "trajectory",
        ["joint"],
        [np.array([0.0]), np.array([0.5])],
        ti.toBounds(-np.ones(1), np.ones(1)),
    )
    variables = [node.getVar("joints") for node in nodes.getNodes()]
    problem = sqp.TrajOptQPProblem(nodes)
    problem.addCostSet(
        ti.JointVelConstraint(np.zeros(1), variables, np.ones(1), "velocity"),
        sqp.CostPenaltyType.SQUARED,
    )
    problem.setup()

    class Observer(sqp.SQPCallback):
        def __init__(self):
            super().__init__()
            self.stats = []

        def execute(self, current_problem, result):
            self.stats.append(native.qp_diagnostic(current_problem, result.new_var_vals))
            return True

    observer = Observer()
    solver = sqp.TrustRegionSQPSolver(sqp.PIQPSolver())
    solver.registerCallback(observer)
    solver.solve(problem)
    assert observer.stats
    assert all(item["qp_columns"] >= 2 and item["qp_rows"] >= 2 for item in observer.stats)
    assert all(
        item["hessian_max_abs"] > 0 and item["constraint_max_abs"] > 0 for item in observer.stats
    )
    assert all(item["primal_max_violation"] >= 0 for item in observer.stats)
