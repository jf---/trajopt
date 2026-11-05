"""Use explicit eased fixture derivatives at fixed-fixture transit joins."""

import numpy as np
from event_aware_spline_boundaries import (
    modeled_transit_boundaries,
    require_fixed_fixture_boundary_velocities,
)
from fixture_chain_clock import FixtureClockError


def eased_transit_boundaries(
    stages, clock, index, *, start_velocity, end_velocity, fixture_endpoint_velocities
):
    """Use actual modeled fixture endpoint derivatives; retain all zero-time events."""
    left, right, events = modeled_transit_boundaries(
        stages,
        clock,
        index,
        interpolation="joint_linear_uniform",
        start_velocity=start_velocity,
        end_velocity=end_velocity,
    )
    for direction in (-1, 1):
        neighbor = index + direction
        while 0 <= neighbor < len(stages) and float(clock[neighbor]["duration_s"]) == 0:
            neighbor += direction
        if not 0 <= neighbor < len(stages):
            continue
        label = stages[neighbor]["stage"]
        if label in fixture_endpoint_velocities:
            override = np.asarray(
                fixture_endpoint_velocities[label][1 if direction < 0 else 0], dtype=float
            )
            if override.shape != (9,) or not np.isfinite(override).all():
                raise FixtureClockError("fixture endpoint velocity must contain nine finite axes")
            if direction < 0:
                left = override
            else:
                right = override
    require_fixed_fixture_boundary_velocities(left, right)
    return left, right, events
