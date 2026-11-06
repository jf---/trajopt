"""Canonical shared process boundaries and independent evaluation are distinct."""

import numpy as np
import pytest
from probe_join_contract import ProbeJoinError, inspect_probe_joins


def phase(stage, start, end):
    return {
        "stage": stage,
        "duration_s": 1.0,
        "ik_states": [np.full(7, value).tolist() for value in np.linspace(start, end, 17)],
        "v0": [0.0] * 7,
        "v1": [0.0] * 7,
        "a0": [0.0] * 7,
        "a1": [0.0] * 7,
    }


def test_shared_canonical_boundary_accepts_conditioned_evaluation_noise():
    phases = [phase("approach:0", 0.0, 1.0), phase("search:0", 1.0, 2.0)]
    joins = [
        {
            "position_delta": [0.0] * 9,
            "velocity_delta": [0.0] * 9,
            "acceleration_delta": [1e-11] * 7 + [0.0, 0.0],
        }
    ]
    result = inspect_probe_joins(phases, joins)
    assert result[0]["raw_acceleration_delta"] == joins[0]["acceleration_delta"]
    assert result[0]["evaluation_bands"][2] > 1e-11


def test_different_canonical_derivative_fails_even_if_evaluation_is_close():
    phases = [phase("approach:0", 0.0, 1.0), phase("search:0", 1.0, 2.0)]
    phases[1]["a0"][0] = 1e-5
    joins = [
        {"position_delta": [0.0] * 9, "velocity_delta": [0.0] * 9, "acceleration_delta": [0.0] * 9}
    ]
    with pytest.raises(ProbeJoinError, match="canonical"):
        inspect_probe_joins(phases, joins)
