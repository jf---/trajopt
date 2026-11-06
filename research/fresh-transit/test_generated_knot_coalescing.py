"""Only a routine sample near a retained transition can be coalesced."""

import numpy as np
from generated_knot_coalescing import coalesce_generated_knots


def test_short_routine_sample_removed_but_transition_and_endpoints_remain():
    times = np.array([0.0, 1.0, 1.00001, 2.0, 3.0])
    values = np.column_stack((times, 2 * times))
    keep, removed = coalesce_generated_knots(times, values, np.array([2.0, 3.0]), [1.00001])
    np.testing.assert_array_equal(keep, [0.0, 1.00001, 2.0, 3.0])
    assert removed == [1.0]


def test_protected_or_fast_distinct_samples_are_retained():
    times = np.array([0.0, 1.0, 1.00001, 2.0, 3.0])
    values = np.column_stack((times, 2 * times))
    keep, removed = coalesce_generated_knots(times, values, np.array([2.0, 3.0]), [1.0, 1.00001])
    np.testing.assert_array_equal(keep, times)
    assert removed == []
    values[1, 1] += 0.2
    keep, removed = coalesce_generated_knots(times, values, np.array([2.0, 3.0]), [1.00001])
    np.testing.assert_array_equal(keep, times)
    assert removed == []


def test_explicit_broader_fraction_preserves_protected_and_speed_guards():
    times = np.array([0.0, 1.0, 1.04, 2.0, 3.0])
    values = np.column_stack((times, 2 * times))
    default, removed = coalesce_generated_knots(times, values, [2.0, 3.0], [1.04])
    np.testing.assert_array_equal(default, times)
    assert removed == []
    broader, removed = coalesce_generated_knots(
        times, values, [2.0, 3.0], [1.04], short_interval_fraction=0.05
    )
    np.testing.assert_array_equal(broader, [0.0, 1.04, 2.0, 3.0])
    assert removed == [1.0]
    protected, removed = coalesce_generated_knots(
        times, values, [2.0, 3.0], [1.0, 1.04], short_interval_fraction=0.05
    )
    np.testing.assert_array_equal(protected, times)
    assert removed == []
    values[1, 1] += 0.3
    fast, removed = coalesce_generated_knots(
        times, values, [2.0, 3.0], [1.04], short_interval_fraction=0.05
    )
    np.testing.assert_array_equal(fast, times)
    assert removed == []
