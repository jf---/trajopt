"""Cartesian tangent classification does not prescribe robot-joint stops."""

import numpy as np
import pytest
from probe_join_geometry import ProbeJoinGeometryError, inspect_probe_join


def pose(x, y=0.0, angle=0.0):
    result = np.eye(4)
    result[:3, 3] = (x, y, 0)
    c, s = np.cos(angle), np.sin(angle)
    result[:2, :2] = ((c, -s), (s, c))
    return result


def test_positive_cartesian_tangent_allows_nonzero_progress():
    report = inspect_probe_join(pose(0), pose(1), pose(3))
    assert report.status == "compatible_with_positive_progress"
    assert report.right_rate_per_left_rate == pytest.approx(0.5)


def test_geometric_corner_and_search_reversal_require_zero_tcp_velocity():
    corner = inspect_probe_join(pose(0), pose(1), pose(1, 1))
    reversal = inspect_probe_join(pose(0), pose(1), pose(0))
    assert corner.status == "zero_tcp_velocity_for_exact_corner"
    assert reversal.status == "zero_tcp_velocity_for_exact_corner"
    assert not hasattr(corner, "joint_velocity")


def test_orientation_change_can_make_collinear_translation_incompatible():
    report = inspect_probe_join(pose(0), pose(1), pose(2, angle=0.2))
    assert report.status == "zero_tcp_velocity_for_exact_corner"


def test_roundoff_scale_orientation_is_retained_and_flagged():
    report = inspect_probe_join(pose(0), pose(1), pose(2, angle=np.finfo(float).eps))
    assert report.status == "compatible_with_positive_progress"
    assert report.rotation_below_roundoff
    assert np.linalg.norm(report.outgoing.rotation) > 0


def test_stationary_leg_is_ambiguous_not_a_whole_robot_stop():
    assert (
        inspect_probe_join(pose(0), pose(0), pose(1)).status
        == "stationary_leg_requires_event_semantics"
    )


def test_malformed_pose_fails_loudly():
    with pytest.raises(ProbeJoinGeometryError):
        inspect_probe_join(np.zeros((3, 3)), pose(1), pose(2))
