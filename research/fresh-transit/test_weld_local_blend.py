"""Local weld blending retains waypoints and shares derivatives."""

import numpy as np
import pytest
from weld_local_blend import WeldBlendError, build_local_weld_blend

# One-sided finite differences have O(h) truncation; scale by sampled magnitude.
FINITE_DIFFERENCE_MULTIPLIER = 32


def test_only_internal_bead_corners_change_and_waypoints_remain():
    points = np.array(
        [[-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [1.0, 2, 0], [1.0, 3, 0]]
    )
    rotations = np.tile(np.eye(3), (len(points), 1, 1))
    position, orientation, info = build_local_weld_blend(points, rotations, [1.0] * 5)
    for i, point in enumerate(points):
        stage = min(i, 4)
        at = 0.0 if i < 5 else 1.0
        assert position(stage, at) == pytest.approx(point)
        assert orientation(stage, at) == pytest.approx(np.eye(3))
    for i in range(1, 5):
        assert position(i - 1, 1.0, 1) == pytest.approx(position(i, 0.0, 1))
        assert position(i - 1, 1.0, 2) == pytest.approx(position(i, 0.0, 2))
    assert info["source_bead_internal_join_count"] == 2
    # A segment away from its local window remains exactly on its source line.
    assert position(1, 0.25) == pytest.approx([0.25, 0, 0])


def test_zero_length_or_reversing_segments_fail_loudly():
    rotations = np.tile(np.eye(3), (4, 1, 1))
    with pytest.raises(WeldBlendError, match="zero"):
        build_local_weld_blend([[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]], rotations, [1, 1, 1])


def test_orientation_has_shared_first_and_second_derivatives_at_source_corner():
    points = np.array([[-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [1.0, 2, 0]])
    rotations = []
    for angle in (0.0, 0.1, 0.3, 0.5, 0.6):
        c, s = np.cos(angle), np.sin(angle)
        rotations.append(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    _, orientation, _ = build_local_weld_blend(points, rotations, [1.0] * 4)
    h = np.finfo(float).eps ** 0.25
    center = orientation(1, 1.0)
    left1, left2 = orientation(1, 1.0 - h), orientation(1, 1.0 - 2 * h)
    right1, right2 = orientation(2, h), orientation(2, 2 * h)
    vleft = (center - left1) / h
    vright = (right1 - center) / h
    aleft = (center - 2 * left1 + left2) / h**2
    aright = (right2 - 2 * right1 + center) / h**2
    band = (
        FINITE_DIFFERENCE_MULTIPLIER * h * max(1.0, np.linalg.norm(aleft), np.linalg.norm(aright))
    )
    assert np.linalg.norm(vleft - vright) <= band
    assert np.linalg.norm(aleft - aright) <= band


def test_single_straight_bead_is_not_spatially_blended():
    points = np.array([[-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    rotations = np.tile(np.eye(3), (4, 1, 1))
    position, _, info = build_local_weld_blend(points, rotations, [1.0] * 3)
    for t in np.linspace(0, 1, 17):
        assert position(1, float(t)) == pytest.approx([t, 0, 0])
    assert info["source_bead_internal_join_count"] == 0
