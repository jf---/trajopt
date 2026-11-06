"""Actual-curve assembly preserves stages and exposes unresolved joins."""

import numpy as np
import pytest
from actual_stage_contract import ActualStage, KinematicState, Seconds, diagnostic_joins
from actual_tour_assembly import TourAssemblyError, assemble_actual_tour


def source(label, start, finish):
    points = np.zeros((3, 9))
    points[:, 0] = [start, (start + finish) / 2, finish]
    return {"stage": label, "states": points.tolist()}


def supplied(label, start, finish, duration):
    def evaluate(t):
        q = (start + (finish - start) * float(t) / duration,) + (0.0,) * 8
        v = ((finish - start) / duration,) + (0.0,) * 8
        return KinematicState(q, v, (0.0,) * 9)

    return ActualStage(label, Seconds(duration), evaluate)


def test_preserves_stage_order_and_reports_real_discontinuities():
    segments = [source("first", 0, 1), source("second", 1, 2)]
    clock = {
        "stages": [{"stage": "first", "duration_s": 1.0}, {"stage": "second", "duration_s": 1.0}]
    }
    stages, corners = assemble_actual_tour(
        segments, clock, {"second": supplied("second", 1.25, 3, 2)}
    )
    assert [stage.label for stage in stages] == ["first", "second"]
    assert stages[1].duration == 2
    assert corners == []
    assert diagnostic_joins(stages)[0].position_delta[0] == pytest.approx(0.25)


def test_rejects_unknown_or_mismatched_overrides():
    segments = [source("first", 0, 1)]
    clock = {"stages": [{"stage": "first", "duration_s": 1.0}]}
    with pytest.raises(TourAssemblyError, match="unknown"):
        assemble_actual_tour(segments, clock, {"other": supplied("other", 0, 1, 1)})
    with pytest.raises(TourAssemblyError, match="label"):
        assemble_actual_tour(segments, clock, {"first": supplied("other", 0, 1, 1)})
