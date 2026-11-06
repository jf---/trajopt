"""Checked generated weld batches must reconstruct their original actual curves."""

import numpy as np
import pytest
from weld_batch_adapter import (
    WeldBatchAdapterError,
    adapt_weld_batch,
    canonicalize_weld_transit_endpoints,
    weld_neighbor_indices,
)


def fixture():
    rows = [
        {"stage": "transit:0", "states": [[0.0] * 9, [0.0] * 9]},
        {"stage": "weld:1", "states": [[0.0] * 9, [0.5] * 7 + [0.0, 0.0]]},
        {"stage": "weld:2", "states": [[0.5] * 7 + [0.0, 0.0], [1.0] * 7 + [0.0, 0.0]]},
        {"stage": "transit:3", "states": [[1.0] * 7 + [0.0, 0.0], [1.0] * 7 + [0.0, 0.0]]},
    ]
    clock = {"stages": [{"stage": row["stage"], "duration_s": 1.0} for row in rows]}
    bead = {
        "source_segment": 4,
        "status": "sampled_exploratory_weld_only",
        "stages": [
            {
                "stage": "weld:1",
                "sampled_speeds_pass": True,
                "collision_and_position": {"sample_count": 1025, "penetrating_samples": 0},
            },
            {
                "stage": "weld:2",
                "sampled_speeds_pass": True,
                "collision_and_position": {"sample_count": 1025, "penetrating_samples": 0},
            },
        ],
        "times_s": [0.0, 1.0, 2.0],
        "joint_states_7": [[0.0] * 7, [0.5] * 7, [1.0] * 7],
        "fixture": [0.0, 0.0],
    }
    return rows, clock, {"tour": 16, "beads": [bead]}


def test_checked_group_preserves_actual_shared_state_and_derivatives():
    rows, clock, batch = fixture()
    stages, unresolved = adapt_weld_batch(rows, clock, batch, 16)
    assert unresolved == []
    assert list(stages) == [1, 2]
    left, right = stages[1][0].evaluate(1.0), stages[2][0].evaluate(0.0)
    np.testing.assert_allclose(left.position, right.position)
    np.testing.assert_allclose(left.velocity, right.velocity)
    np.testing.assert_allclose(left.acceleration, right.acceleration)
    assert weld_neighbor_indices(rows, stages) == {0, 3}
    changed = canonicalize_weld_transit_endpoints(rows, stages)
    assert {item["transit"] for item in changed} == {"transit:0", "transit:3"}


def test_failed_group_is_explicitly_unresolved():
    rows, clock, batch = fixture()
    batch["beads"][0]["status"] = "sampled_bead_order_unresolved"
    stages, unresolved = adapt_weld_batch(rows, clock, batch, 16)
    assert stages == {}
    assert unresolved == [{"source_segment": 4, "status": "sampled_bead_order_unresolved"}]


def test_changed_generated_boundary_fails_loudly():
    rows, clock, batch = fixture()
    batch["beads"][0]["joint_states_7"][0][0] = 0.1
    with pytest.raises(WeldBatchAdapterError, match="boundary"):
        adapt_weld_batch(rows, clock, batch, 16)
