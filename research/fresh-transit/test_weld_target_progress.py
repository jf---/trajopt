"""Actual progress is measured against a 3D target, not one chord alone."""

import numpy as np
from weld_target_progress import measure_target_progress


def test_three_dimensional_corner_is_forward_without_chord_shortcut():
    times = np.arange(5, dtype=float)
    target = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]]
    )
    report = measure_target_progress(target, target, times)
    assert report["nearest_target_backward_steps"] == 0
    assert report["actual_local_tangent_backward_total_m"] == 0
    assert report["max_distance_to_sampled_3d_target_m"] == 0


def test_reverse_is_reported_in_spatial_units():
    times = np.arange(5, dtype=float)
    target = np.column_stack((times, np.zeros((5, 2))))
    actual = target[[0, 1, 3, 2, 4]]
    report = measure_target_progress(actual, target, times)
    assert report["nearest_target_backward_steps"] == 1
    assert report["nearest_target_total_backward_m"] == 1.0
    assert report["actual_local_tangent_backward_total_m"] == 1.0
