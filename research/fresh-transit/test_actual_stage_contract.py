"""Actual-curve join behavior with explicit event and derivative semantics."""

import pytest
from actual_stage_contract import (
    ActualStage,
    JoinContractError,
    KinematicState,
    StationaryEvent,
    diagnostic_joins,
    require_join_continuity,
)

Q0 = (0.0,) * 9
Q1 = (1.0,) + (0.0,) * 8
V = (0.2,) + (0.0,) * 8
A = (0.3,) + (0.0,) * 8
ZERO = (0.0,) * 9


def stage(name, first, last, v=V, a=A):
    def evaluate(t):
        return KinematicState(first if t == 0 else last, v, a)

    return ActualStage(name, 1.0, evaluate)


def test_actual_endpoints_cross_consecutive_ordered_events_without_implied_rest():
    path = [
        stage("first", Q0, Q1),
        StationaryEvent("hit", Q1),
        StationaryEvent("return", Q1),
        stage("next", Q1, Q0),
    ]
    join = diagnostic_joins(path)[0]
    assert join.events == ("hit", "return")
    assert join.position_delta == ZERO
    assert join.velocity_delta == ZERO
    assert join.acceleration_delta == ZERO
    require_join_continuity([join], ZERO, ZERO, ZERO)


def test_unknown_derivatives_are_not_inferred_or_accepted():
    joins = diagnostic_joins([stage("first", Q0, Q1, v=None, a=None), stage("next", Q1, Q0)])
    assert joins[0].velocity_delta is None
    assert joins[0].acceleration_delta is None
    with pytest.raises(JoinContractError, match="unknown"):
        require_join_continuity(joins, ZERO, ZERO, ZERO)


def test_jumps_report_each_axis_and_consumer_uses_explicit_tolerances():
    moving = list(V)
    moving[7] = 0.4
    joins = diagnostic_joins([stage("first", Q0, Q1), stage("next", Q1, Q0, tuple(moving))])
    assert joins[0].velocity_delta[7] == pytest.approx(0.4)
    with pytest.raises(JoinContractError, match="velocity"):
        require_join_continuity(joins, ZERO, ZERO, ZERO)
    band = list(ZERO)
    band[7] = 0.5
    require_join_continuity(joins, ZERO, tuple(band), ZERO)


def test_consecutive_zero_time_events_cannot_change_any_state():
    different = list(Q1)
    different[8] = 1.0
    with pytest.raises(JoinContractError, match="zero-time"):
        diagnostic_joins(
            [
                stage("first", Q0, Q1),
                StationaryEvent("a", Q1),
                StationaryEvent("b", tuple(different)),
                stage("next", Q1, Q0),
            ]
        )


def test_zero_duration_motion_and_invalid_sample_fail_loudly():
    with pytest.raises(JoinContractError, match="positive"):
        diagnostic_joins([ActualStage("bad", 0.0, lambda _: KinematicState(Q0, None, None))])
    with pytest.raises(JoinContractError, match="nine"):
        diagnostic_joins([ActualStage("bad", 1.0, lambda _: KinematicState((0.0,), None, None))])


def test_endpoint_events_are_not_dropped():
    path = [StationaryEvent("start", Q0), stage("first", Q0, Q1), StationaryEvent("end", Q1)]
    joins = diagnostic_joins(path)
    assert [(j.left, j.right, j.events) for j in joins] == [
        (None, "first", ("start",)),
        ("first", None, ("end",)),
    ]
    with pytest.raises(JoinContractError, match="unknown"):
        require_join_continuity(joins, ZERO, ZERO, ZERO)


def test_supplied_outer_boundaries_are_checked_without_assuming_rest():
    path = [StationaryEvent("start", Q0), stage("motion", Q0, Q1), StationaryEvent("end", Q1)]
    joins = diagnostic_joins(
        path,
        initial_state=KinematicState(Q0, V, A),
        final_state=KinematicState(Q1, V, A),
    )
    assert [join.velocity_delta for join in joins] == [ZERO, ZERO]
    assert [join.acceleration_delta for join in joins] == [ZERO, ZERO]
    require_join_continuity(joins, ZERO, ZERO, ZERO)


def test_event_excursion_is_not_hidden_by_matching_outer_states():
    middle = list(Q1)
    middle[7] = 0.3
    joins = diagnostic_joins(
        [stage("a", Q0, Q1), StationaryEvent("fixture", tuple(middle)), stage("b", Q1, Q0)]
    )
    assert joins[0].position_delta == ZERO
    with pytest.raises(JoinContractError, match="position"):
        require_join_continuity(joins, ZERO, ZERO, ZERO)


def test_missing_outer_conditions_do_not_vacuously_pass_continuity():
    joins = diagnostic_joins([stage("only", Q0, Q1)])
    assert joins == []
    with pytest.raises(JoinContractError, match="no joins"):
        require_join_continuity(joins, ZERO, ZERO, ZERO)


def test_supplied_outer_conditions_are_checked_without_events():
    joins = diagnostic_joins(
        [stage("motion", Q0, Q1)],
        initial_state=KinematicState(Q0, V, A),
        final_state=KinematicState(Q1, V, A),
    )
    assert [(join.left, join.right) for join in joins] == [(None, "motion"), ("motion", None)]
    require_join_continuity(joins, ZERO, ZERO, ZERO)
