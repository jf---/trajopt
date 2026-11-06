import numpy as np
import pytest
from fixture_multichord import FixtureMultichordError, build_fixture_multichord


def test_three_corner_chords_keep_ordered_targets_without_stopping_track():
    q = np.zeros((4, 9))
    q[:, 0] = [0.0, 0.8, 1.6, 2.4]
    q[:, 7] = [0.0, 0.4, 0.75, 1.30]
    rows = [
        {"stage": f"fixture:{i}", "states": [a, (a + b) / 2, b]}
        for i, (a, b) in enumerate(zip(q[:-1], q[1:]))
    ]
    v0 = np.zeros(9)
    v1 = np.zeros(9)
    v0[0] = v1[0] = 0.5
    model, nodes, boundaries = build_fixture_multichord(
        rows, [2.0, 2.0, 2.0], v0, np.zeros(9), v1, np.zeros(9)
    )
    position, velocity, acceleration = model.evaluate(np.asarray(boundaries), nodes, 2)
    assert position == pytest.approx(q)
    assert velocity[0, 0] == pytest.approx(0.5)
    assert velocity[-1, 0] == pytest.approx(0.5)
    assert np.all(np.abs(velocity[1:-1, 0]) > 0.0)
    assert np.isfinite(acceleration).all()


def test_zero_duration_or_disconnected_stage_fails():
    q = np.zeros(9)
    r = q.copy()
    r[0] = 1.0
    s = q.copy()
    s[0] = 2.0
    rows = [{"stage": "fixture:0", "states": [q, r]}, {"stage": "fixture:1", "states": [s, q]}]
    with pytest.raises(FixtureMultichordError):
        build_fixture_multichord(rows, [1.0, 1.0], q, q, q, q)
    with pytest.raises(FixtureMultichordError):
        build_fixture_multichord(rows[:1], [0.0], q, q, q, q)
