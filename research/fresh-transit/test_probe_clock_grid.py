"""A shared ramp fraction preserves source cruise and explicit endpoint conditions."""

import numpy as np
import pytest
from probe_clock_variants import search_cruise_clock_with_ramp
from probe_cruise_clock import ProbeClockError, search_cruise_clock


@pytest.mark.parametrize("fraction", [0.25, 0.5])
def test_ramp_fraction_keeps_nominal_cruise_and_distance(fraction):
    duration, clock = search_cruise_clock_with_ramp(2.0, ramp_fraction=fraction)
    assert duration == pytest.approx(2.0 * (1.0 + fraction))
    t = np.linspace(0.0, duration, 1001)
    p, v, a, _ = clock(t)
    assert p[0] == pytest.approx(0.0)
    assert p[-1] == pytest.approx(1.0)
    assert v[0] == pytest.approx(0.0)
    assert v[-1] == pytest.approx(0.0)
    assert a[0] == pytest.approx(0.0)
    assert a[-1] == pytest.approx(0.0)
    assert max(v) == pytest.approx(0.5)
    assert min(v) >= 0
    assert np.all(np.diff(p) >= 0)


def test_half_ramp_reproduces_existing_clock():
    duration, variant = search_cruise_clock_with_ramp(2.0, ramp_fraction=0.5)
    original_duration, original = search_cruise_clock(2.0)
    assert duration == original_duration
    times = np.linspace(0.0, duration, 101)
    for expected, observed in zip(original(times), variant(times), strict=True):
        np.testing.assert_array_equal(expected, observed)


def test_one_sided_progress_peak_is_the_shared_duration_factor():
    # The existing one-sided quintic has p'(u) = 1 + 12u^2 - 28u^3 + 15u^4.
    # Its interior peak is p'(2/5) = 189/125.
    derivative = np.polynomial.Polynomial([1.0, 0.0, 12.0, -28.0, 15.0])
    candidates = [0.0, 1.0, 2.0 / 5.0]
    assert max(float(derivative(x)) for x in candidates) == pytest.approx(189.0 / 125.0)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_ramp_fraction_is_rejected(fraction):
    with pytest.raises(ProbeClockError):
        search_cruise_clock_with_ramp(2.0, ramp_fraction=fraction)
