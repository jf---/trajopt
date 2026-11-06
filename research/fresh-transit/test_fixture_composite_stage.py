import numpy as np
import pytest
from _quintic_spline import MonotoneFixtureAxis
from actual_stage_contract import Seconds, diagnostic_joins
from fixture_composite_stage import fixture_composite_stage
from fixture_multichord import build_fixture_multichord


def test_composite_stage_preserves_full_nine_axis_join_derivatives():
    q = np.zeros((3, 9))
    q[:, 0] = [0.0, 0.8, 1.6]
    q[:, 7] = [0.0, 0.4, 0.8]
    rows = [
        {"stage": f"fixture:{i}", "states": [a, (a + b) / 2, b]}
        for i, (a, b) in enumerate(zip(q[:-1], q[1:]))
    ]
    boundary = np.zeros(9)
    boundary[0] = 0.5
    model, nodes, times = build_fixture_multichord(
        rows, [2.0, 2.0], boundary, np.zeros(9), boundary, np.zeros(9)
    )
    pos = MonotoneFixtureAxis.build(times, q[:, 7].tolist())
    tail = MonotoneFixtureAxis.build(times, q[:, 8].tolist())
    stages = [
        fixture_composite_stage(rows[i]["stage"], model, nodes, (pos, tail), times[i], times[i + 1])
        for i in range(2)
    ]
    join = diagnostic_joins(stages)[0]
    assert join.position_delta == pytest.approx(np.zeros(9))
    assert join.velocity_delta == pytest.approx(np.zeros(9))
    assert join.acceleration_delta == pytest.approx(np.zeros(9))
    assert stages[0].evaluate(Seconds(2.0)).velocity[7] > 0.0
    assert stages[0].evaluate(Seconds(2.0)).velocity[0] != 0.0
