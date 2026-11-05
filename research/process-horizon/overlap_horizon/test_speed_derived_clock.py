"""Speed-derived transit duration policy, independent of route or robot."""

import pytest
from speed_derived_clock import SpeedClockError, refine_speed_clock


def test_reoptimizes_from_measured_peak_and_records_both_attempts():
    calls = []

    def solve(duration):
        calls.append(duration)
        return {"duration_s": duration, "peak_ratio": 1.2 / duration}

    accepted, attempts = refine_speed_clock(1.0, solve)
    assert calls == pytest.approx([1.0, 1.2])
    assert accepted["peak_ratio"] == pytest.approx(1.0)
    assert len(attempts) == 2


def test_solver_invalid_or_infeasible_clock_fails_loudly():
    with pytest.raises(SpeedClockError):
        refine_speed_clock(0.0, lambda _duration: {})
    with pytest.raises(SpeedClockError):
        refine_speed_clock(
            1.0, lambda duration: {"duration_s": duration, "peak_ratio": float("nan")}
        )
    with pytest.raises(SpeedClockError):
        refine_speed_clock(
            1.0, lambda duration: {"duration_s": duration, "peak_ratio": 2.0}, max_attempts=2
        )
