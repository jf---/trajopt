"""Opt-in transit timing changes preserve process and fixture clocks."""

import pytest
from verify_duration_policy import TransitDurationError, apply_transit_durations


def test_only_explicit_transit_duration_changes_and_input_stays_unmodified():
    clock = {
        "stages": [
            {"stage": "transit:0", "duration_s": 1.0},
            {"stage": "search:0", "duration_s": 2.0},
            {"stage": "fixture:1", "duration_s": 3.0},
        ]
    }
    actual = apply_transit_durations(clock, {"transit:0": 1.25})
    assert [stage["duration_s"] for stage in actual["stages"]] == [1.25, 2.0, 3.0]
    assert clock["stages"][0]["duration_s"] == 1.0


def test_unknown_process_and_nonpositive_overrides_fail():
    clock = {
        "stages": [
            {"stage": "transit:0", "duration_s": 1.0},
            {"stage": "search:0", "duration_s": 2.0},
        ]
    }
    for overrides in ({"transit:unknown": 2.0}, {"search:0": 3.0}, {"transit:0": 0.0}):
        with pytest.raises(TransitDurationError):
            apply_transit_durations(clock, overrides)
