"""Shared TCP boundary velocity retains an explicit generated nullspace reference."""

import numpy as np
import pytest
from probe_shared_velocity import ProbeSharedVelocityError, solve_shared_velocity


def full_rank_jacobian():
    matrix = np.zeros((6, 7))
    matrix[:, :6] = np.eye(6)
    return matrix


def test_compatible_shared_twist_keeps_generated_nullspace_velocity():
    reference = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
    target = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = solve_shared_velocity(full_rank_jacobian(), target, reference, np.ones(7), np.ones(6))
    assert result.status == "modeled_candidate"
    assert result.velocity[0] == pytest.approx(0.2)
    assert result.velocity[6] == pytest.approx(0.4)


def test_zero_tcp_twist_does_not_imply_zero_robot_velocity():
    reference = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
    result = solve_shared_velocity(
        full_rank_jacobian(), np.zeros(6), reference, np.ones(7), np.ones(6)
    )
    assert result.status == "modeled_candidate"
    assert result.velocity[6] == pytest.approx(0.4)


def test_zero_tcp_twist_can_keep_coordinated_track_and_arm_motion():
    matrix = np.zeros((6, 7))
    matrix[0, :2] = (1.0, 1.0)
    matrix[1:, 2:] = np.eye(5)
    reference = np.array([0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = solve_shared_velocity(matrix, np.zeros(6), reference, np.ones(7), np.ones(6))
    assert result.status == "modeled_candidate"
    assert result.velocity[:2] == pytest.approx((0.4, -0.4))


def test_rank_deficiency_and_speed_excess_are_loud():
    with pytest.raises(ProbeSharedVelocityError, match="rank"):
        solve_shared_velocity(np.zeros((6, 7)), np.ones(6), np.zeros(7), np.ones(7), np.ones(6))
    result = solve_shared_velocity(
        full_rank_jacobian(),
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(7),
        np.ones(7),
        np.ones(6),
    )
    assert result.status == "candidate_exceeds_sourced_speed"


def test_missing_generated_reference_is_rejected():
    with pytest.raises(ProbeSharedVelocityError, match="generated reference"):
        solve_shared_velocity(full_rank_jacobian(), np.zeros(6), None, np.ones(7), np.ones(6))
