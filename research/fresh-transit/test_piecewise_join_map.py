"""Piecewise process joins constrain actual shared spline derivatives."""

import numpy as np
from piecewise_join_map import combine_join_maps


def test_join_map_is_left_minus_right_for_every_axis():
    left = np.asarray([[1.0, 2.0]])
    right = np.asarray([[3.0, 4.0, 5.0]])
    basis = combine_join_maps(left, right)
    np.testing.assert_array_equal(basis, [[1.0, 2.0, -3.0, -4.0, -5.0]])
    values = np.arange(10, dtype=float).reshape(5, 2)
    np.testing.assert_allclose(basis @ values, left @ values[:2] - right @ values[2:])
