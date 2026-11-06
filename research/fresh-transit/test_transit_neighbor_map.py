"""Shared transit boundaries preserve ordered stationary events."""

import pytest
from event_aware_spline_boundaries import STATIONARY_EVENT
from transit_neighbor_map import TransitNeighborError, transit_neighbor_map
from verified_linear_stages import actual_linear_stage


def state(track):
    return [track] + [0.0] * 8


def stage(label, a, b):
    return {"stage": label, "states": [state(a), state(b)]}


def test_adjacent_events_remain_ordered_without_implying_rest():
    rows = [
        stage("approach:0", 0, 1),
        stage("event:0", 1, 1),
        stage("event:1", 1, 1),
        stage("transit:0", 1, 2),
        stage("weld:0", 2, 3),
    ]
    timing = [
        {
            "stage": row["stage"],
            "duration_s": 0.0 if row["stage"].startswith("event:") else 1.0,
            "clock_kind": STATIONARY_EVENT if row["stage"].startswith("event:") else "modeled",
        }
        for row in rows
    ]
    actual = {
        i: actual_linear_stage(row["stage"], row["states"], timing[i]["duration_s"])
        for i, row in enumerate(rows)
        if not row["stage"].startswith("transit:")
    }
    left, right, events = transit_neighbor_map(rows, timing, actual)[3]
    assert (left.label, right.label) == ("approach:0", "weld:0")
    assert events == (1, 2)


def test_event_that_changes_state_is_rejected():
    rows = [stage("event:0", 0, 1)]
    timing = [{"stage": "event:0", "duration_s": 0.0, "clock_kind": STATIONARY_EVENT}]
    with pytest.raises(TransitNeighborError, match="event"):
        transit_neighbor_map(rows, timing, {})
