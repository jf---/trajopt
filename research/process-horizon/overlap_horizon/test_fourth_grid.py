"""Actual native derivative peaks and risk integrals at the grid boundary."""

import numpy as np
import pytest
from run_fourth_grid import inspect_transits


def test_quadrature_and_peak_for_cubic_joint_curve():
    t = np.linspace(0.0, 1.0, 7)
    nodes = np.zeros((7, 7))
    nodes[:, 0] = t**3
    v0 = np.zeros(7)
    v1 = np.zeros(7)
    v1[0] = 3.0
    a0 = np.zeros(7)
    a1 = np.zeros(7)
    a1[0] = 6.0
    curve = {
        "stage": "transit:1",
        "knots_s": t.tolist(),
        "nodes": nodes.tolist(),
        "v0": v0.tolist(),
        "v1": v1.tolist(),
        "a0": a0.tolist(),
        "a1": a1.tolist(),
    }
    result = inspect_transits([{"curve": curve}], np.full(7, 10.0), {"transit:1": curve})
    assert result["peak_speed_by_axis"][0] == pytest.approx(3.0)
    assert result["peak_jerk_by_axis"][0] == pytest.approx(6.0)
    assert result["transit_exact_normalized_jerk_squared"] == pytest.approx(0.36)
    assert result["transit_exact_normalized_jerk_fourth"] == pytest.approx(0.1296)
    assert result["max_baseline_knot_position_difference_native_units"] == 0.0
