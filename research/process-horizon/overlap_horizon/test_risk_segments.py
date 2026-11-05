"""Generated macro splicing preserves the source and fixes transit endpoints."""

import numpy as np
import pytest
from risk_segments import RiskSegmentsError, build_risk_segments


def test_macro_and_affected_transits_are_copied_and_connected():
    fixture = [0.1, 1.1]
    path = np.zeros((65, 7))
    path[:, 0] = 2.6
    path[:, 1] = np.linspace(0.0, 1.0, 65)
    names = [
        "fixture:0",
        "transit:1",
        "approach:3",
        "search:3",
        "retract:3",
        "withdraw:3",
        "transit:2",
        "weld:3",
    ]
    old = [{"stage": x, "states": [[0.0] * 7 + fixture, [0.0] * 7 + fixture]} for x in names]
    candidate = {
        "macros": [{"id": "3:0:0", "path": path.tolist()}],
        "transits": [{"solution": {"stage": "transit:1"}}, {"solution": {"stage": "transit:2"}}],
    }
    actual = build_risk_segments(old, candidate)
    assert actual[2]["states"][0] == [2.6, 0.0] + [0.0] * 5 + fixture
    assert actual[1]["states"][-1] == actual[2]["states"][0]
    assert actual[-2]["states"][0] == actual[-3]["states"][-1]
    assert old[2]["states"][0][0] == 0.0


def test_invalid_or_unmatched_candidate_fails():
    with pytest.raises(RiskSegmentsError):
        build_risk_segments([], {"macros": [{"id": "1:0:0", "path": []}], "transits": []})


def test_duplicate_generated_probe_is_rejected():
    path = np.zeros((65, 7)).tolist()
    stages = [
        {"stage": f"{phase}:3", "states": [[0.0] * 9, [0.0] * 9]}
        for phase in ("approach", "search", "retract", "withdraw")
    ]
    candidate = {
        "macros": [{"id": "3:0:0", "path": path}, {"id": "3:1:1", "path": path}],
        "transits": [],
    }
    with pytest.raises(RiskSegmentsError):
        build_risk_segments(stages, candidate)
