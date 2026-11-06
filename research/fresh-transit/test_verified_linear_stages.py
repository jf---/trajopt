"""Verified joint-linear stages retain their geometry and report corner derivatives."""

import numpy as np
import pytest
from actual_stage_contract import StationaryEvent
from verified_linear_stages import LinearStageError, actual_linear_stage


def test_interpolates_all_nine_axes_with_actual_modeled_derivatives():
    states = np.zeros((3, 9))
    states[:, 0] = [0.0, 1.0, 2.0]
    states[:, 7] = [0.0, 0.5, 1.0]
    stage = actual_linear_stage("fixture:0", states, 4.0)
    mid = stage.evaluate(1.0)
    assert mid.position[0] == pytest.approx(0.5)
    assert mid.position[7] == pytest.approx(0.25)
    assert mid.velocity[0] == pytest.approx(0.5)
    assert mid.velocity[7] == pytest.approx(0.25)
    assert mid.acceleration == pytest.approx((0.0,) * 9)


def test_direction_change_has_unknown_velocity_and_acceleration_at_the_corner():
    states = np.zeros((3, 9))
    states[:, 0] = [0.0, 1.0, 0.0]
    stage = actual_linear_stage("weld:0", states, 2.0)
    corner = stage.evaluate(1.0)
    assert corner.position[0] == 1.0
    assert corner.velocity is None and corner.acceleration is None
    assert stage.evaluate(0.0).velocity[0] == pytest.approx(1.0)
    assert stage.evaluate(2.0).velocity[0] == pytest.approx(-1.0)


def test_stationary_zero_time_becomes_ordered_event_without_rest():
    states = np.zeros((2, 9))
    event = actual_linear_stage("probe-hit", states, 0.0)
    assert isinstance(event, StationaryEvent)
    assert event.position == (0.0,) * 9
    states[-1, 8] = 1.0
    with pytest.raises(LinearStageError, match="zero-time"):
        actual_linear_stage("impossible", states, 0.0)


def test_invalid_time_and_missing_axis_are_rejected():
    stage = actual_linear_stage("transit:1", np.zeros((2, 9)), 1.0)
    with pytest.raises(LinearStageError, match="outside"):
        stage.evaluate(2.0)
    with pytest.raises(LinearStageError, match="nine"):
        actual_linear_stage("bad", np.zeros((2, 8)), 1.0)


def test_constructed_nonuniform_binary_knots_keep_corner_undefined():
    states = np.zeros((8, 9))
    states[:, 0] = [0, 1, 0, 1, 0, 1, 0, 1]
    stage = actual_linear_stage("probe:0", states, 0.3)
    for t in np.linspace(0.0, 0.3, 8)[1:-1]:
        assert stage.evaluate(float(t)).velocity is None
