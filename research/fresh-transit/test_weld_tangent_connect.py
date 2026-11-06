"""Regenerated weld transits must terminate at the new generated lead states."""

import numpy as np
import pytest
from weld_tangent_connect import WeldTransitError, attach_fixture_to_route


def test_generated_route_preserves_new_endpoints_and_fixture():
    start = np.zeros(9)
    finish = np.r_[np.ones(7), 0.3, 1.1]
    start[7:] = finish[7:]
    route = np.stack((start[:7], finish[:7]))
    actual = attach_fixture_to_route("transit:3", route, start, finish)
    np.testing.assert_array_equal(actual["states"], np.stack((start, finish)))


def test_old_endpoint_or_fixture_change_is_rejected():
    start, finish = np.zeros(9), np.ones(9)
    route = np.stack((start[:7], finish[:7]))
    with pytest.raises(WeldTransitError, match="fixture"):
        attach_fixture_to_route("transit:3", route, start, finish)
    finish[7:] = 0
    route[-1, 0] = 0
    with pytest.raises(WeldTransitError, match="endpoint"):
        attach_fixture_to_route("transit:3", route, start, finish)
