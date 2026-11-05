"""Contracts for fixed-duration probe process clocks and composed derivatives."""

import numpy as np
import pytest
from progress_clock import ProgressClockError, compose_process, quintic_progress
from scipy.interpolate import CubicSpline


def test_probe_clock_preserves_outer_derivatives_and_direction():
    t = np.linspace(0.0, 2.0, 101)
    s, v, a, _ = quintic_progress(t, 2.0, 0.0, 1.5)
    np.testing.assert_allclose(s[[0, -1]], [0.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(v[[0, -1]], [0.0, 0.75], atol=1e-14)
    np.testing.assert_allclose(a[[0, -1]], [0.0, 0.0], atol=1e-14)
    assert np.min(v) >= 0.0


def test_composed_joint_derivatives_match_independent_differences():
    # Polynomial joint geometry gives closed derivatives without assuming a robot.
    u = np.linspace(0.0, 1.0, 9)
    curve = CubicSpline(u, np.column_stack((u**3, u**2, u)), axis=0)
    t = np.linspace(0.01, 1.99, 5001)
    q, v, a, j = compose_process(curve, t, 2.0, 0.5, 1.5)
    np.testing.assert_allclose(np.gradient(q, t, axis=0)[2:-2], v[2:-2], atol=2e-6)
    np.testing.assert_allclose(np.gradient(v, t, axis=0)[2:-2], a[2:-2], atol=2e-6)
    np.testing.assert_allclose(np.gradient(a, t, axis=0)[2:-2], j[2:-2], atol=2e-5)


def test_clock_rejects_between_sample_reverse_and_bad_inputs():
    with pytest.raises(ProgressClockError, match="revers"):
        quintic_progress(np.array([0.0, 1.0]), 1.0, 5.0, 5.0)
    with pytest.raises(ProgressClockError):
        quintic_progress(np.array([0.0, 1.0]), 0.0, 0.0, 1.0)
    with pytest.raises(ProgressClockError):
        quintic_progress(np.array([-0.1, 1.0]), 1.0, 0.0, 1.0)
