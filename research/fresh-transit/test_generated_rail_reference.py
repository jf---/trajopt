import numpy as np
import pytest
from generated_rail_reference import GeneratedRailError, generated_rail_at


def test_moving_and_stationary_generated_rail_at_process_time():
    states = np.zeros((3, 9))
    states[:, 0] = [1.6, 1.2, 0.8]
    assert generated_rail_at(states, 4.0, 1.0) == pytest.approx(1.4)
    assert generated_rail_at(states, 4.0, 4.0) == pytest.approx(0.8)
    states[:, 0] = 0.4
    assert generated_rail_at(states, 4.0, 1.0) == pytest.approx(0.4)


def test_invalid_clock_or_missing_nine_axis_states_fails_loudly():
    with pytest.raises(GeneratedRailError):
        generated_rail_at(np.zeros((2, 9)), 0.0, 0.0)
    with pytest.raises(GeneratedRailError):
        generated_rail_at(np.zeros((2, 9)), 1.0, 1.1)
    with pytest.raises(GeneratedRailError):
        generated_rail_at(np.zeros((1, 7)), 1.0, 0.0)
