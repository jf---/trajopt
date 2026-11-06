"""Refinement selects actual process and contact extremes without blanket sampling."""

import numpy as np
import pytest
from weld_actual_witnesses import WeldWitnessError, bracketing_witnesses, select_weld_witnesses


def test_selects_worst_position_and_penetration_across_stages():
    result = select_weld_witnesses(
        [np.array([0.0, 1.0, 2.0]), np.array([2.0, 3.0, 4.0])],
        [np.array([0.0, 0.2, 0.0]), np.array([0.0, 0.5, 0.0])],
        [np.array([0.1, -0.3, 0.2]), np.array([0.1, -0.2, 0.2])],
        [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])],
    )
    assert result == {0: [1.0], 1: [3.0]}


def test_no_extra_collision_witness_when_clear_and_no_duplicates():
    result = select_weld_witnesses(
        [np.array([0.0, 1.0])],
        [np.array([0.0, 0.5])],
        [np.array([1.0, 1.0])],
        [np.array([0.0, 1.0])],
    )
    assert result == {0: [1.0]}
    with pytest.raises(WeldWitnessError):
        select_weld_witnesses(
            [np.array([0.0, 1.0])],
            [np.array([0.0])],
            [np.array([1.0, 1.0])],
            [np.array([0.0, 1.0])],
        )


def test_reverse_progress_adds_both_sides_of_worst_step():
    result = select_weld_witnesses(
        [np.array([0.0, 1.0, 2.0, 3.0])],
        [np.array([0.0, 0.0, 0.0, 1.0])],
        [np.array([0.0, 0.0, 0.0, 0.0])],
        [np.array([0.0, 1.0, 0.4, 2.0])],
    )
    assert result == {0: [1.0, 2.0, 3.0]}


def test_collision_sample_is_bracketed_by_existing_witnesses():
    assert bracketing_witnesses([0.0, 1.0, 2.0], 1.3) == (1.0, 2.0)
    assert bracketing_witnesses([0.0, 1.0, 2.0], 1.0) == (1.0, 1.0)
    with pytest.raises(WeldWitnessError):
        bracketing_witnesses([0.0, 1.0], 1.5)
