"""Adaptive peak witnesses cover both signs without a physical jerk ceiling."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from peak_witnesses import PeakWitnessError, select_peak_witnesses
from scipy.interpolate import make_interp_spline


def curve(sign):
    times = np.array([0.0, 0.11, 0.27, 0.46, 0.69, 0.86, 1.0])
    polynomial = sign * Polynomial([0.0, 0.0, 0.0, 0.91 / 6.0, 0.6 / 24.0, -1.0 / 60.0])
    return make_interp_spline(
        times,
        polynomial(times)[:, None],
        k=5,
        bc_type=(
            [(1, [polynomial.deriv()(0)]), (2, [polynomial.deriv(2)(0)])],
            [(1, [polynomial.deriv()(1)]), (2, [polynomial.deriv(2)(1)])],
        ),
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_violating_extrema_adds_a_witness_for_either_sign(sign):
    times, evidence = select_peak_witnesses(
        curve(sign), np.array([0.95]), np.array([1.0]), [0.0, 1.0], 1e-4
    )
    assert times == pytest.approx([0.3])
    assert evidence[0]["normalized_excess"] == pytest.approx(0.05)


def test_no_new_witness_when_peak_is_enclosed_and_stagnation_fails():
    assert (
        select_peak_witnesses(curve(1.0), np.array([1.01]), np.array([1.0]), [0.0, 1.0], 1e-4)[0]
        == []
    )
    witness = select_peak_witnesses(
        curve(1.0), np.array([0.95]), np.array([1.0]), [0.0, 1.0], 1e-4
    )[0][0]
    with pytest.raises(PeakWitnessError):
        select_peak_witnesses(
            curve(1.0), np.array([0.95]), np.array([1.0]), [0.0, witness, 1.0], 1e-4
        )


def test_witness_rounds_warm_start_and_retain_all_witnesses():
    from peak_witnesses import refine_peak_witnesses

    calls = []

    def solve(times, previous):
        calls.append((times.copy(), previous is not None))
        return {"curve": curve(1.0), "peak_epigraph": [0.95 if previous is None else 1.0]}

    result, rounds = refine_peak_witnesses([0.0, 1.0], solve, np.array([1.0]), 1e-4)
    assert result["peak_epigraph"] == [1.0]
    assert len(rounds) == 2
    assert calls[0][1] is False and calls[1][1] is True
    assert calls[1][0] == pytest.approx([0.0, 0.3, 1.0])


def test_witness_budget_failure_keeps_failure_explicit():
    from peak_witnesses import refine_peak_witnesses

    with pytest.raises(PeakWitnessError):
        refine_peak_witnesses(
            [0.0, 1.0],
            lambda _times, _previous: {"curve": curve(1.0), "peak_epigraph": [0.95]},
            np.array([1.0]),
            1e-4,
            max_rounds=1,
        )
