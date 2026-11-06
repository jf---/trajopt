"""Shared TCP boundary velocity retains an explicit generated nullspace reference."""

import numpy as np
import pytest
from probe_shared_velocity import (
    MACHINE_ROUNDOFF_ULPS,
    ProbeSharedVelocityError,
    solve_constrained_shared_acceleration,
    solve_constrained_shared_velocity,
    solve_shared_velocity,
)


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


def test_active_track_limit_keeps_feasible_nullspace_direction():
    matrix = np.zeros((6, 7))
    matrix[0, :2] = (1.0, 1.0)
    matrix[1:, 2:] = np.eye(5)
    limits = np.tile([-1.0, 1.0], (7, 1))
    limits[0] = (0.0, 1.0)
    q = np.zeros(7)
    outward = np.array([-0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    inward = -outward
    stopped = solve_constrained_shared_velocity(
        matrix, np.zeros(6), outward, np.ones(7), np.ones(6), q, limits
    )
    moving = solve_constrained_shared_velocity(
        matrix, np.zeros(6), inward, np.ones(7), np.ones(6), q, limits
    )
    assert stopped.velocity[0] >= -MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps
    assert np.asarray(matrix) @ np.asarray(stopped.velocity) == pytest.approx(np.zeros(6))
    # At an interior join the curve has two sides, so the bounded track must
    # have zero velocity there even if a one-sided motion could point inward.
    assert moving.velocity[:2] == pytest.approx((0.0, 0.0))


def test_interior_track_limit_does_not_force_unrelated_arm_nullspace_to_stop():
    limits = np.tile([-1.0, 1.0], (7, 1))
    limits[0] = (0.0, 1.0)
    reference = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
    result = solve_constrained_shared_velocity(
        full_rank_jacobian(), np.zeros(6), reference, np.ones(7), np.ones(6), np.zeros(7), limits
    )
    assert result.velocity[6] == pytest.approx(0.4)


def test_tangent_cone_infeasibility_fails_without_changing_tcp_twist():
    limits = np.tile([-1.0, 1.0], (7, 1))
    limits[0] = (0.0, 1.0)
    with pytest.raises(ProbeSharedVelocityError, match="feasible"):
        solve_constrained_shared_velocity(
            full_rank_jacobian(),
            np.array([-0.1, 0, 0, 0, 0, 0]),
            np.zeros(7),
            np.ones(7),
            np.ones(6),
            np.zeros(7),
            limits,
        )


def test_shared_acceleration_is_inward_at_active_lower_bound():
    matrix = np.zeros((6, 7))
    matrix[0, :2] = (1.0, 1.0)
    matrix[1:, 2:] = np.eye(5)
    target = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = solve_constrained_shared_acceleration(matrix, target, np.ones(7), np.ones(6), (0,), ())
    assert result.acceleration[0] >= -MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps
    assert matrix @ np.asarray(result.acceleration) == pytest.approx(target)
