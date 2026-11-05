"""Explicit modeled nine-axis transit boundary policy."""

from event_aware_spline_boundaries import (
    modeled_transit_boundaries,
    require_fixed_fixture_boundary_velocities,
)


def modeled_boundaries(stages, clock, index, *, start_velocity, end_velocity):
    """Keep stationary events and reject fixture velocity incompatible with a transit.

    Velocities use track metres/s, six arm radians/s, positioner radians/s and
    tailstock metres/s. Values are modeled derivatives, not controller feedback.
    """
    left, right, events = modeled_transit_boundaries(
        stages,
        clock,
        index,
        interpolation="joint_linear_uniform",
        start_velocity=start_velocity,
        end_velocity=end_velocity,
    )
    require_fixed_fixture_boundary_velocities(left, right)
    return left, right, events
