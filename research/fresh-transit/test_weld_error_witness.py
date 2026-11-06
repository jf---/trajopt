import numpy as np
import pytest
from weld_error_witness import WeldWitnessError, select_error_witnesses


def test_normalized_position_and_orientation_select_different_stages():
    stages = [
        (np.array([0.0, 0.1]), np.array([0.0, 0.0]), np.array([False, True])),
        (np.array([0.0, 0.001]), np.array([0.0, 0.4]), np.array([False, True])),
    ]
    assert select_error_witnesses(stages, 2) == [(0, 1), (1, 1)]


def test_excluded_samples_and_zero_errors_do_not_add_witnesses():
    stages = [(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([True, True]))]
    assert select_error_witnesses(stages, 3) == []
    stages = [(np.array([3.0, 1.0]), np.zeros(2), np.array([False, True]))]
    assert select_error_witnesses(stages, 1) == [(0, 1)]


def test_bad_dimensions_fail_loudly():
    with pytest.raises(WeldWitnessError):
        select_error_witnesses([(np.zeros(2), np.zeros(3), np.ones(2, dtype=bool))], 1)
