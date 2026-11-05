"""Actual-stage adapters reproduce analytical source curves and derivatives."""

import numpy as np
import pytest
from actual_stage_adapters import (
    StageAdapterError,
    fixture_stage,
    process_stage,
    transit_stage,
    weld_stage,
)
from actual_stage_contract import diagnostic_joins

FIXTURE = (0.7, 1.1)
ZERO7 = np.zeros(7)


class LinearNative:
    def evaluate(self, times, nodes, derivative):
        t = float(times[0])
        return np.array([nodes[0] + t * nodes[1]]), np.array([nodes[1]]), np.array([ZERO7])


class LineCurve:
    def __call__(self, time, derivative=0):
        if derivative == 0:
            return np.arange(7, dtype=float) + time
        return np.ones(7) if derivative == 1 else np.zeros(7)


def test_native_transit_preserves_fixture_and_actual_derivatives():
    a = transit_stage("transit:0", 2.0, LinearNative(), np.array([ZERO7, np.ones(7)]), FIXTURE)
    state = a.evaluate(1.0)
    assert state.position == pytest.approx((1.0,) * 7 + FIXTURE)
    assert state.velocity == pytest.approx((1.0,) * 7 + (0.0, 0.0))
    assert state.acceleration == pytest.approx((0.0,) * 9)
    with pytest.raises(StageAdapterError):
        a.evaluate(3.0)


def test_process_uses_actual_progress_chain_rule_without_implied_rest():
    a = process_stage("approach:0", 2.0, LineCurve(), FIXTURE, 1.0, 0.0)
    begin, end = a.evaluate(0.0), a.evaluate(2.0)
    assert begin.position[:7] == pytest.approx(tuple(range(7)))
    assert begin.velocity[:7] == pytest.approx((0.5,) * 7)
    assert end.position[:7] == pytest.approx(tuple(x + 1 for x in range(7)))
    assert end.velocity == pytest.approx((0.0,) * 9)
    assert begin.acceleration == pytest.approx((0.0,) * 9)


def test_fixture_progress_retains_all_nine_axes_and_event_time():
    start = [0.0] * 9
    end = [0.0] * 9
    end[0], end[7] = 2.0, 1.0
    chain = {"start": start, "end": end, "duration_s": 4.0}
    first = fixture_stage("fixture:0", chain, 0.0, 2.0)
    second = fixture_stage("fixture:1", chain, 2.0, 4.0)
    state = first.evaluate(2.0)
    assert state.position[0] == pytest.approx(1.0)
    assert state.position[7] == pytest.approx(0.5)
    assert state.velocity[0] > 0.0 and state.velocity[7] > 0.0
    join = diagnostic_joins([first, second])[0]
    assert join.position_delta == pytest.approx((0.0,) * 9)
    assert join.velocity_delta == pytest.approx((0.0,) * 9)
    assert join.acceleration_delta == pytest.approx((0.0,) * 9)


def test_weld_curve_uses_original_offset_time_and_fixed_fixture():
    weld = weld_stage("weld:0", 2.0, LineCurve(), 3.0, FIXTURE)
    state = weld.evaluate(1.0)
    assert state.position[:7] == pytest.approx(tuple(x + 4 for x in range(7)))
    assert state.position[7:] == FIXTURE
    assert state.velocity[:7] == pytest.approx((1.0,) * 7)
    assert state.acceleration == pytest.approx((0.0,) * 9)


def test_process_rejects_nonmonotone_progress_instead_of_fabricating_derivatives():
    with pytest.raises(StageAdapterError, match="progress"):
        process_stage("search:0", 1.0, LineCurve(), FIXTURE, 10.0, 0.0)


def test_actual_sample_comparison_catches_interior_mismatch():
    from actual_stage_adapters import check_stage_samples

    stage = weld_stage("weld:0", 2.0, LineCurve(), 3.0, FIXTURE)
    times = np.array([0.0, 1.0, 2.0])
    states = [stage.evaluate(float(t)) for t in times]
    positions = np.array([s.position for s in states])
    velocities = np.array([s.velocity for s in states])
    accelerations = np.array([s.acceleration for s in states])
    check_stage_samples(stage, times, positions, velocities, accelerations)
    positions[1, 0] += 0.1
    with pytest.raises(StageAdapterError, match="position"):
        check_stage_samples(stage, times, positions, velocities, accelerations)


def test_source_clock_endpoint_roundoff_uses_declared_stage_duration():
    from actual_stage_adapters import check_stage_samples

    stage = weld_stage("weld:0", 0.1, LineCurve(), 0.3, FIXTURE)
    times = np.array([0.3, 0.35, 0.4])
    values = [stage.evaluate(t) for t in (0.0, 0.05, 0.1)]
    check_stage_samples(
        stage,
        times,
        [x.position for x in values],
        [x.velocity for x in values],
        [x.acceleration for x in values],
    )
