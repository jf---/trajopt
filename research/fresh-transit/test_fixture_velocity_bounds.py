import numpy as np
import pytest
from fixture_velocity_bounds import FixtureVelocityError, fixture_velocity_bounds


def test_fixture_reversal_intersects_order_constraints_without_restricting_track():
    states = np.zeros((3, 9))
    states[:, 0] = [0.0, 0.8, 0.0]
    states[:, 7] = [0.0, 0.4, 0.2]
    bounds = fixture_velocity_bounds(states, [0.0, 1.0, 2.0], [0.5, 1.0, 1.5], np.ones(7))
    assert bounds[0][0] == pytest.approx((-1.0, 1.0))
    assert bounds[1][0] == pytest.approx((-1.0, 1.0))
    assert bounds[0][7] == (0.0, float("inf"))
    assert bounds[1][7] == (0.0, 0.0)
    assert bounds[2][7] == (-float("inf"), 0.0)
    assert all(row[8] == (0.0, 0.0) for row in bounds)


def test_invalid_stage_times_fail():
    with pytest.raises(FixtureVelocityError):
        fixture_velocity_bounds(np.zeros((2, 9)), [0.0, 0.0], [0.0], np.ones(7))
