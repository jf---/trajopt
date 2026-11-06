import numpy as np
import pytest
from position_witness_times import PositionWitnessError, merge_position_witness_times


def test_actual_extremum_is_added_without_removing_original_collocation():
    actual = merge_position_witness_times([0.033, 0.066], [0.0187, 0.033], 1.0)
    assert actual == pytest.approx([0.0187, 0.033, 0.066])


def test_invalid_or_outside_witness_fails():
    with pytest.raises(PositionWitnessError):
        merge_position_witness_times([0.1], [1.1], 1.0)
    with pytest.raises(PositionWitnessError):
        merge_position_witness_times([0.1], [np.nan], 1.0)
