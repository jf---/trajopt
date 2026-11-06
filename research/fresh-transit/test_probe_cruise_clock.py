"""A modeled SearchL clock ramps smoothly and retains nominal interior feed."""

import numpy as np
import pytest
from probe_cruise_clock import ProbeClockError, search_cruise_clock


def test_preserves_order_endpoints_and_nominal_cruise_rate():
    duration, evaluate = search_cruise_clock(2.0)
    assert duration == pytest.approx(3.0)
    progress, rate, accel, jerk = evaluate(np.linspace(0, duration, 1001))
    assert progress[0] == pytest.approx(0)
    assert progress[-1] == pytest.approx(1)
    assert np.all(np.diff(progress) >= 0)
    assert max(rate) == pytest.approx(0.5)
    assert rate[0] == pytest.approx(0)
    assert rate[-1] == pytest.approx(0)
    assert accel[0] == pytest.approx(0)
    assert accel[-1] == pytest.approx(0)
    assert jerk[0] == pytest.approx(0)
    assert jerk[-1] == pytest.approx(0)
    assert rate[500] == pytest.approx(0.5)


def test_rejects_nonpositive_duration_and_outside_clock():
    with pytest.raises(ProbeClockError):
        search_cruise_clock(0.0)
    _, evaluate = search_cruise_clock(1.0)
    with pytest.raises(ProbeClockError):
        evaluate(np.array([-1.0]))


def test_constructed_clock_endpoint_is_exact_for_source_duration():
    duration, evaluate = search_cruise_clock(1.8999999999999997)
    progress, _, _, _ = evaluate(np.linspace(0.0, duration, 17))
    assert progress[-1] == 1.0
