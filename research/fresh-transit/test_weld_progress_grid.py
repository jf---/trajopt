"""A shared witness policy includes ordered endpoints for each process leg."""

import numpy as np
import pytest
from weld_progress_grid import (
    WeldProgressGridError,
    local_process_times,
    process_knot_witnesses,
    witness_grid,
)


def test_common_witness_grid_keeps_waypoint_boundaries_and_order():
    edges, witnesses = witness_grid([2.0, 4.0, 3.0], 5)
    np.testing.assert_allclose(edges, [0.0, 2.0, 6.0, 9.0])
    assert [len(values) for values in witnesses] == [5, 5, 5]
    for left, right in zip(witnesses[:-1], witnesses[1:], strict=True):
        assert left[-1] == right[0]
        assert np.all(np.diff(left) > 0)


def test_invalid_timing_or_count_fails_loudly():
    with pytest.raises(WeldProgressGridError):
        witness_grid([1.0, 0.0], 5)
    with pytest.raises(WeldProgressGridError):
        witness_grid([1.0], 1)


def test_process_knot_witnesses_include_original_knot_and_existing_sample_times():
    np.testing.assert_array_equal(
        process_knot_witnesses([0.0, 0.5, 1.0], [0.0, 0.3, 1.0]),
        [0.0, 0.3, 0.5, 1.0],
    )
    with pytest.raises(WeldProgressGridError):
        process_knot_witnesses([0.0, 0.5, 1.0], [0.0, 1.5])


def test_known_stage_endpoints_use_the_supplied_duration():
    start, duration = 1.234, 2.3
    actual = local_process_times([start, start + duration / 2, start + duration], start, duration)
    assert actual[0] == 0.0
    assert actual[-1] == duration
    np.testing.assert_allclose(actual[1], duration / 2)
    with pytest.raises(WeldProgressGridError):
        local_process_times([start, start + duration + 0.5], start, duration)
