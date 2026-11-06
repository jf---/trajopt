"""A bounded soft-objective result remains explicit and preserves strict mode."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from peak_witnesses import PeakWitnessError, refine_peak_witnesses, refine_peak_witnesses_bounded
from scipy.interpolate import make_interp_spline


def curve():
    times = np.array([0.0, 0.11, 0.27, 0.46, 0.69, 0.86, 1.0])
    p = Polynomial([0.0, 0.0, 0.0, 0.91 / 6.0, 0.6 / 24.0, -1.0 / 60.0])
    return make_interp_spline(
        times,
        p(times)[:, None],
        k=5,
        bc_type=(
            [(1, [p.deriv()(0)]), (2, [p.deriv(2)(0)])],
            [(1, [p.deriv()(1)]), (2, [p.deriv(2)(1)])],
        ),
    )


def test_budgeted_result_keeps_curve_and_marks_objective_incomplete():
    def solve(_times, _previous):
        return {"curve": curve(), "peak_epigraph": [0.95], "marker": "final"}

    result, rounds, status = refine_peak_witnesses_bounded(
        [0.0, 1.0], solve, np.array([1.0]), 1e-4, max_rounds=1
    )
    assert result["marker"] == "final"
    assert status == "budget_exhausted"
    assert rounds[-1]["extrema"][0]["absolute_jerk"] == pytest.approx(1.0)
    assert rounds[-1]["extrema"][0]["normalized_excess"] == pytest.approx(0.05)
    with pytest.raises(PeakWitnessError, match="budget exhausted"):
        refine_peak_witnesses([0.0, 1.0], solve, np.array([1.0]), 1e-4, max_rounds=1)


def test_bounded_result_marks_satisfied_witnesses_complete():
    def solve(_times, _previous):
        return {"curve": curve(), "peak_epigraph": [1.0]}

    _, _, status = refine_peak_witnesses_bounded(
        [0.0, 1.0], solve, np.array([1.0]), 1e-4, max_rounds=1
    )
    assert status == "satisfied"


def test_explicit_bounded_solver_retains_terminal_native_solution(monkeypatch):
    import peak_transit_solver as module

    c = curve()
    times = np.array([0.0, 0.11, 0.27, 0.46, 0.69, 0.86, 1.0])
    nodes = c(times)

    def native(*args, **kwargs):
        return [
            {
                "knots_s": times.tolist(),
                "nodes": nodes.tolist(),
                "v0": c(0, 1).tolist(),
                "a0": c(0, 2).tolist(),
                "v1": c(1, 1).tolist(),
                "a1": c(1, 2).tolist(),
                "peak_epigraph": [0.95],
            }
        ]

    monkeypatch.setattr(module, "refine_transits", native)
    solve = module.make_bounded_peak_transit_solver(0.1, 1.0, max_witness_rounds=1)
    result = solve(
        None, None, (), [{"stage": "transit:1"}], {"stages": [{"duration_s": 1.0}]}, [1.0]
    )[0]
    assert result["peak_witness_status"] == "budget_exhausted"
    assert result["peak_witness_rounds"][-1]["extrema"][0]["absolute_jerk"] == pytest.approx(1.0)
