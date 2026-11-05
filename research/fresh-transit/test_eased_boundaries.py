import numpy as np
import pytest
from eased_boundaries import eased_transit_boundaries
from event_aware_spline_boundaries import FixtureBoundaryVelocityError


def test_opt_in_eased_fixture_uses_actual_shared_endpoint_velocity():
    a = np.zeros(9)
    b = a.copy()
    b[7] = 1.0
    c = b.copy()
    c[0] = 1.0
    stages = [
        {"stage": "fixture:0", "states": [a.tolist(), b.tolist()]},
        {"stage": "transit:1", "states": [b.tolist(), c.tolist()]},
    ]
    clock = [{"stage": "fixture:0", "duration_s": 1.0}, {"stage": "transit:1", "duration_s": 1.0}]
    start, end, events = eased_transit_boundaries(
        stages,
        clock,
        1,
        start_velocity=None,
        end_velocity=np.zeros(9),
        fixture_endpoint_velocities={"fixture:0": (np.zeros(9), np.zeros(9))},
    )
    np.testing.assert_array_equal(start, np.zeros(9))
    np.testing.assert_array_equal(end, np.zeros(9))
    assert not events
    with pytest.raises(FixtureBoundaryVelocityError):
        eased_transit_boundaries(
            stages,
            clock,
            1,
            start_velocity=None,
            end_velocity=np.zeros(9),
            fixture_endpoint_velocities={},
        )
