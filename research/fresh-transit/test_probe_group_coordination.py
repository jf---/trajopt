"""Probe feasibility diagnostic preserves the modeled nine-axis path."""

import numpy as np
import pytest
from actual_stage_contract import Seconds, diagnostic_joins
from probe_group_coordination import ProbeGroupError, build_probe_group


def states(start, end):
    q = np.zeros((5, 9))
    q[:, 0] = np.linspace(start, end, 5)
    return q.tolist()


def test_group_matches_position_velocity_and_acceleration_at_probe_events():
    rows = [
        {"stage": label, "states": states(a, b)}
        for label, a, b in zip(
            ("approach:0", "search:0", "retract:0", "withdraw:0"),
            (0, 1, 2, 1),
            (1, 2, 1, 0),
            strict=True,
        )
    ]
    stages, _ = build_probe_group(rows, [1.0] * 4)
    for join in diagnostic_joins(stages):
        assert np.max(np.abs(join.position_delta)) == pytest.approx(0)
        assert np.max(np.abs(join.velocity_delta)) == pytest.approx(0)
        assert np.max(np.abs(join.acceleration_delta)) == pytest.approx(0)
    assert stages[0].evaluate(Seconds(0)).velocity[0] > 0
    assert stages[-1].evaluate(stages[-1].duration).velocity[0] < 0


def test_group_refuses_a_discontinuous_or_non_probe_sequence():
    rows = [{"stage": "weld:0", "states": states(0, 1)}]
    with pytest.raises(ProbeGroupError):
        build_probe_group(rows, [1.0])


def test_probe_groups_cannot_mix_probe_indices():
    rows = [
        {"stage": label, "states": states(a, b)}
        for label, a, b in zip(
            ("approach:0", "search:0", "retract:1", "withdraw:0"),
            (0, 1, 2, 1),
            (1, 2, 1, 0),
            strict=True,
        )
    ]
    with pytest.raises(ProbeGroupError, match="same probe"):
        build_probe_group(rows, [1.0] * 4)
