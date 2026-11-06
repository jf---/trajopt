"""A process-curve fit preserves generated knots and reports actual derivatives."""

import numpy as np
import pytest
from process_curve_representation import ProcessCurveError, build_process_curve


def test_cubic_fit_preserves_nine_axis_knots_and_models_c2_motion():
    t = np.linspace(0.0, 2.0, 5)
    states = np.zeros((5, 9))
    states[:, 0] = t * t
    states[:, 1] = 3 * t
    states[:, 7] = 0.4
    states[:, 8] = 1.1
    stage, curve = build_process_curve("search:0", states, 2.0)
    for i, time in enumerate(t):
        result = stage.evaluate(float(time))
        assert result.position == pytest.approx(states[i])
        assert result.velocity[0] == pytest.approx(2 * time)
        assert result.acceleration[0] == pytest.approx(2.0)
        assert result.position[7:] == pytest.approx((0.4, 1.1))
    assert curve(1.0, 2)[0] == pytest.approx(2.0)


def test_moving_fixture_and_missing_geometry_fail_loudly():
    states = np.zeros((5, 9))
    states[-1, 7] = 1.0
    with pytest.raises(ProcessCurveError, match="fixture"):
        build_process_curve("search:0", states, 1.0)
    with pytest.raises(ProcessCurveError, match="nine"):
        build_process_curve("search:0", np.zeros((2, 8)), 1.0)
    with pytest.raises(ProcessCurveError, match="duration"):
        build_process_curve("search:0", np.zeros((2, 9)), 0.0)
