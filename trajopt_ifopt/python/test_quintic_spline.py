"""Native quintic parity, Ifopt consumer and GIL-release contracts."""

import sys
import threading

import numpy as np
import pytest
from _quintic_spline import (
    QuinticSplineError,
    QuinticSplineJerkCost,
    QuinticSplineModel,
    QuinticSplinePositionLimits,
)
from scipy.interpolate import make_interp_spline
from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as sqp

TIMES = [0.0, 0.2, 0.65, 1.1, 1.7, 2.3, 3.0]
POSITIONS = np.array(
    [[0.1, -0.3], [-0.2, 0.4], [0.4, 0.2], [0.25, -0.1], [0.7, 0.6], [0.3, 0.8], [0.9, 0.2]]
)
V0 = np.array([0.35, -0.4])
A0 = np.array([-0.18, 0.12])
V1 = np.array([-0.22, 0.32])
A1 = np.array([0.27, -0.14])


def test_native_maps_match_scipy_with_nonuniform_clock():
    native = QuinticSplineModel.build(TIMES, V0, A0, V1, A1)
    reference = make_interp_spline(
        TIMES, POSITIONS, k=5, bc_type=([(1, V0), (2, A0)], [(1, V1), (2, A1)])
    )
    samples = np.linspace(0, 3, 71)
    for order in range(4):
        mapping = native.map(samples.tolist(), order)
        np.testing.assert_allclose(
            mapping.basis @ POSITIONS + mapping.offset,
            reference(samples, nu=order),
            rtol=2e-9,
            atol=2e-9,
        )
    with pytest.raises(QuinticSplineError):
        native.map([-0.1], 0)


def test_native_ifopt_cost_and_position_constraints_solve_with_piqp():
    model = QuinticSplineModel.build(TIMES, V0, A0, V1, A1)
    names = ["track", "arm"]
    nodes = []
    for i, position in enumerate(POSITIONS):
        node = ti.Node(f"knot-{i}")
        bounds = (
            [ti.Bounds(float(v), float(v)) for v in position]
            if i in (0, 6)
            else [ti.Bounds(-3, 3)] * 2
        )
        node.addVar("position", names, position, bounds)
        nodes.append(node)
    variables = ti.NodesVariables("knots", nodes)
    positions = [node.getVar("position") for node in variables.getNodes()]
    cost = QuinticSplineJerkCost(model, positions, np.array([2.0, 3.0]))
    limits = QuinticSplinePositionLimits(
        model, positions, np.linspace(0, 3, 15).tolist(), [ti.Bounds(-3, 3)] * 2
    )
    assert type(cost).__module__ == "_quintic_spline"
    assert type(limits).__module__ == "_quintic_spline"
    with pytest.raises(QuinticSplineError):
        QuinticSplineJerkCost(model, positions[:-1], np.array([2.0, 3.0]))
    with pytest.raises(QuinticSplineError):
        QuinticSplineJerkCost(model, positions, np.array([2.0, -1.0]))
    with pytest.raises(QuinticSplineError):
        QuinticSplinePositionLimits(model, positions, [0.5, 0.5], [ti.Bounds(-3, 3)] * 2)
    problem = sqp.TrajOptQPProblem(variables)
    problem.addCostSet(cost, sqp.CostPenaltyType.SQUARED)
    problem.addConstraintSet(limits)
    problem.setup()
    before = float(np.dot(cost.getCoefficients(), np.square(cost.getValues())))
    solver = sqp.TrustRegionSQPSolver(sqp.PIQPSolver())
    solver.solve(problem)
    assert solver.getStatus() == sqp.SQPStatus.NLP_CONVERGED
    after = float(np.dot(cost.getCoefficients(), np.square(cost.getValues())))
    assert np.isfinite(after) and after < before
    np.testing.assert_allclose(
        variables.getValues().reshape(7, 2)[[0, -1]], POSITIONS[[0, -1]], atol=1e-6, rtol=0
    )
    assert all(
        b.getLower() <= value <= b.getUpper()
        for b, value in zip(limits.getBounds(), limits.getValues(), strict=True)
    )


def test_native_map_releases_gil_while_evaluating():
    model = QuinticSplineModel.build(TIMES, V0, A0, V1, A1)
    start = threading.Event()
    progressed = threading.Event()

    def worker():
        start.wait()
        progressed.set()

    thread = threading.Thread(target=worker)
    thread.start()
    old = sys.getswitchinterval()
    sys.setswitchinterval(1.0)
    try:
        start.set()
        result = model.map([1.1] * 200_000, 3)
        progressed_during_call = progressed.is_set()
    finally:
        sys.setswitchinterval(old)
        thread.join(timeout=5)
    assert result.basis.shape == (200_000, 7)
    assert progressed_during_call
