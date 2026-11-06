"""Matched PIQP cost-scaling replay exercises one captured QP."""

import _qp_scale_replay as replay
import numpy as np
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp


def test_matched_piqp_scale_cost_replay_uses_same_qp_and_box():
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

    class ReplayObserver(sqp.SQPCallback):
        def __init__(self):
            super().__init__()
            self.comparisons = []

        def execute(self, current_problem, result):
            if not self.comparisons:
                self.comparisons.append(
                    replay.compare(current_problem, result.best_var_vals, result.box_size)
                )
            return True

    observer = ReplayObserver()
    solver = sqp.TrustRegionSQPSolver(sqp.PIQPSolver())
    solver.registerCallback(observer)
    solver.solve(problem)
    assert len(observer.comparisons) == 1
    default, scaled = observer.comparisons[0]
    assert not default["scale_cost"] and scaled["scale_cost"]
    assert default["status"] == scaled["status"] == "solved"
    assert default["max_constraint_violation"] < 1e-4
    assert scaled["max_constraint_violation"] < 1e-4
    assert default["max_trust_box_violation"] < 1e-4
    assert scaled["max_trust_box_violation"] < 1e-4
    for item in (default, scaled):
        np.testing.assert_allclose(item["objective"], item["piqp_objective"], atol=1e-4, rtol=0)
    np.testing.assert_allclose(default["objective"], scaled["objective"], atol=1e-4, rtol=0)
    np.testing.assert_allclose(scaled["objective_delta_from_default"], 0.0, atol=1e-4, rtol=0)
