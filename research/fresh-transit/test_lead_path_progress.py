"""Curved lead progress follows its target, not the endpoint chord."""

import numpy as np
import pytest
from lead_path_progress import LeadPathProgressError, sampled_path_progress


def test_curved_lead_with_chord_reversal_stays_ordered():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[2.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, -1.0, 0.0]])
    tangents, arc = sampled_path_progress(points, velocities)
    assert np.dot(points[2] - points[1], points[-1] - points[0]) < 0
    assert np.all(np.diff(arc) > 0)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    np.testing.assert_allclose(
        (arc + np.einsum("ij,ij->i", tangents, points - points)) / arc[-1], arc / arc[-1]
    )


def test_zero_target_velocity_is_explicitly_unsupported():
    with pytest.raises(LeadPathProgressError):
        sampled_path_progress(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.zeros((2, 3)))
