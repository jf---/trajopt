"""Native fixture-axis interpolation preserves targets and compatible motion."""

import numpy as np
import pytest
from _quintic_spline import MonotoneFixtureAxis

MACHINE_ROUNDOFF_ULPS = 128  # Binary64 allowance, not a physical tolerance.


def test_monotone_targets_keep_internal_motion_and_stop_only_at_reversal_or_hold():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    targets = np.array([0.0, 1.0, 1.2, 1.1, 1.1])
    axis = MonotoneFixtureAxis.build(times.tolist(), targets.tolist())
    at_targets = axis.evaluate_many(times.tolist())
    assert at_targets[:, 0] == pytest.approx(targets)
    assert at_targets[1, 1] > 0.0
    assert at_targets[2:, 1] == pytest.approx(np.zeros(3))
    assert at_targets[:, 2] == pytest.approx(np.zeros(5))
    for i in range(4):
        sample = axis.evaluate_many(np.linspace(times[i], times[i + 1], 101).tolist())
        direction = np.sign(targets[i + 1] - targets[i])
        if direction:
            assert np.min(direction * sample[:, 1]) >= -MACHINE_ROUNDOFF_ULPS * np.finfo(float).eps
        else:
            assert np.max(np.abs(sample[:, 1])) == pytest.approx(0.0)
