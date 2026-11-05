import numpy as np
import pytest
from fixture_chain_clock import FixtureClockError, evaluate_fixture_clock, fixture_chain_clock


def stage(label, a, b):
    return {"stage": label, "states": [a.tolist(), b.tolist()]}


def test_retimes_collinear_fixture_chain_without_intermediate_stops():
    q0 = np.zeros(9)
    q0[7] = 2.0
    q1 = q0.copy()
    q1[7] = 1.6
    q2 = q0.copy()
    q2[7] = 0.8
    q3 = q0.copy()
    q3[7] = -0.3
    stages = [stage("fixture:0", q0, q1), stage("fixture:1", q1, q2), stage("fixture:2", q2, q3)]
    clock = fixture_chain_clock(stages, 3.0)
    for i, q in enumerate((q0, q1, q2, q3)):
        state, vel, acc = evaluate_fixture_clock(clock, clock["stage_times_s"][i])
        np.testing.assert_allclose(state, q, atol=1e-13, rtol=0)
        if i in (0, 3):
            np.testing.assert_allclose(vel, 0, atol=1e-13, rtol=0)
            np.testing.assert_allclose(acc, 0, atol=1e-13, rtol=0)
        else:
            assert vel[7] < 0
    values = np.array([evaluate_fixture_clock(clock, t)[0][7] for t in np.linspace(0, 3, 101)])
    assert np.all(np.diff(values) <= 0)
    assert clock["duration_s"] == 3.0


def test_rejects_reverse_or_nonlinear_nine_axis_path():
    q0 = np.zeros(9)
    q1 = q0.copy()
    q1[7] = 1
    q2 = q1.copy()
    q2[7] = 0.5
    with pytest.raises(FixtureClockError, match="monotone"):
        fixture_chain_clock([stage("fixture:0", q0, q1), stage("fixture:1", q1, q2)], 1.0)
    q2 = q1.copy()
    q2[7] = 2
    q1[0] = 0.1
    with pytest.raises(FixtureClockError, match="chord"):
        fixture_chain_clock([stage("fixture:0", q0, q1), stage("fixture:1", q1, q2)], 1.0)


def test_stationary_fixture_stage_is_not_silently_retimed_away():
    a = np.zeros(9)
    b = a.copy()
    b[7] = 1.0
    with pytest.raises(FixtureClockError, match="stationary"):
        fixture_chain_clock([stage("fixture:0", a, a), stage("fixture:1", a, b)], 2.0)
