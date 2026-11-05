import numpy as np
import pytest
from event_aware_spline_boundaries import (
    BoundaryTimingError,
    FixtureBoundaryVelocityError,
    MissingBoundaryVelocityError,
)
from fresh_boundaries import modeled_boundaries


def make_stage(label, a, b):
    return {"stage": label, "states": [a.tolist(), b.tolist()]}


def test_preserves_stationary_event_and_uses_neighbor_slopes():
    a = np.zeros(9)
    b = a.copy()
    b[0] = 1
    c = b.copy()
    c[1] = 2
    stages = [
        make_stage("approach:0", a, b),
        make_stage("weld:1", b, b),
        make_stage("transit:2", b, c),
    ]
    clock = [
        {"stage": "approach:0", "duration_s": 2},
        {"stage": "weld:1", "duration_s": 0, "clock_kind": "source_zero_time_stationary_event"},
        {"stage": "transit:2", "duration_s": 1},
    ]
    left, right, events = modeled_boundaries(
        stages, clock, 2, start_velocity=None, end_velocity=np.zeros(9)
    )
    assert left[0] == 0.5 and right[1] == 0 and events == (1,)


def test_requires_tour_boundary_velocity():
    a = np.zeros(9)
    b = a.copy()
    b[0] = 1
    with pytest.raises(MissingBoundaryVelocityError):
        modeled_boundaries(
            [make_stage("transit:0", a, b)],
            [{"stage": "transit:0", "duration_s": 1}],
            0,
            start_velocity=None,
            end_velocity=np.zeros(9),
        )


def test_rejects_zero_time_fixture_change():
    a = np.zeros(9)
    b = a.copy()
    b[7] = 1
    stages = [make_stage("weld:1", a, b), make_stage("transit:2", b, b)]
    clock = [
        {"stage": "weld:1", "duration_s": 0, "clock_kind": "source_zero_time_stationary_event"},
        {"stage": "transit:2", "duration_s": 1},
    ]
    with pytest.raises(BoundaryTimingError):
        modeled_boundaries(stages, clock, 1, start_velocity=np.zeros(9), end_velocity=np.zeros(9))


def test_rejects_moving_fixture_velocity_at_transit():
    a = np.zeros(9)
    b = a.copy()
    b[7] = 1
    stages = [make_stage("fixture:0", a, b), make_stage("transit:1", b, b)]
    clock = [{"stage": "fixture:0", "duration_s": 1}, {"stage": "transit:1", "duration_s": 1}]
    with pytest.raises(FixtureBoundaryVelocityError):
        modeled_boundaries(stages, clock, 1, start_velocity=np.zeros(9), end_velocity=np.zeros(9))
