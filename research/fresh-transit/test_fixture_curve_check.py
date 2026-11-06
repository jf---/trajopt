"""Fixture easing is checked on its actual nine-axis curve."""

from types import SimpleNamespace

import numpy as np
import pytest
from actual_stage_adapters import fixture_stage
from fixture_curve_check import check_fixture_curves
from fresh_validation import CurveCollisionError


def fixture():
    start = np.zeros(9)
    end = start.copy()
    end[7] = 1.0
    return fixture_stage(
        "fixture:0", {"start": start.tolist(), "end": end.tolist(), "duration_s": 2.0}, 0.0, 2.0
    )


def test_samples_real_eased_positions_with_bounds_and_velocity():
    checker = SimpleNamespace(contacts=lambda _q: [])
    result = check_fixture_curves(
        {"fixture:0": fixture()}, checker, np.tile([-2.0, 2.0], (9, 1)), 1e-4
    )
    assert result[0]["sample_count"] == 1025
    assert result[0]["penetrating_samples"] == 0
    assert result[0]["peak_velocity_by_axis"][7] > 0


def test_collision_between_endpoints_fails_without_fallback():
    checker = SimpleNamespace(
        contacts=lambda q: [SimpleNamespace(distance_m=-0.01)] if 0.4 < q[7] < 0.6 else []
    )
    with pytest.raises(CurveCollisionError):
        check_fixture_curves({"fixture:0": fixture()}, checker, np.tile([-2.0, 2.0], (9, 1)), 1e-4)
