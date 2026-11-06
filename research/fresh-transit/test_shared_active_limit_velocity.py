import numpy as np
import pytest
from shared_active_limit_velocity import SharedLimitError, shared_active_limit_velocity


def test_only_active_axes_stop_and_interior_axes_keep_generated_velocity():
    q = np.array([0.0, 0.5, 1.0])
    v = np.array([-0.4, 0.3, 0.6])
    limits = np.array([[0.0, 1.0]] * 3)
    assert shared_active_limit_velocity(q, v, limits) == pytest.approx([0.0, 0.3, 0.0])


def test_state_outside_limit_fails_instead_of_clamping_position():
    with pytest.raises(SharedLimitError):
        shared_active_limit_velocity([-0.01], [-0.4], [[0.0, 1.0]])
