"""The explicit shared SQP and witness policy reaches each refinement."""

import numpy as np
import peak_transit_solver as module
from numpy.polynomial import Polynomial


def test_explicit_threshold_and_round_budget_reach_native_solver(monkeypatch):
    times = np.linspace(0.0, 1.0, 7)
    p = Polynomial([0.0, 0.0, 0.0, 0.91 / 6.0, 0.6 / 24.0, -1.0 / 60.0])
    calls = []
    witness_calls = []
    original = module.refine_peak_witnesses

    def fake(*_args, **kwargs):
        calls.append(kwargs)
        return [
            {
                "knots_s": times.tolist(),
                "nodes": p(times)[:, None].tolist(),
                "v0": [p.deriv()(0)],
                "a0": [p.deriv(2)(0)],
                "v1": [p.deriv()(1)],
                "a1": [p.deriv(2)(1)],
                "peak_epigraph": [1.0],
            }
        ]

    def witnessed(*args, **kwargs):
        witness_calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "refine_transits", fake)
    monkeypatch.setattr(module, "refine_peak_witnesses", witnessed)
    solve = module.make_peak_transit_solver(
        0.1, 1.0, absolute_merit_improvement=1e-9, max_witness_rounds=12
    )
    solve(None, None, (), [{"stage": "transit:1"}], {"stages": [{"duration_s": 1.0}]}, [1.0])
    assert calls[0]["absolute_merit_improvement"] == 1e-9
    assert witness_calls[0]["max_rounds"] == 12


def test_witness_refinement_warm_starts_the_optimized_accelerations(monkeypatch):
    times = np.linspace(0.0, 1.0, 7)
    p = Polynomial([0.0, 0.0, 0.0, 0.91 / 6.0, 0.6 / 24.0, -1.0 / 60.0])
    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return [
            {
                "knots_s": times.tolist(),
                "nodes": p(times)[:, None].tolist(),
                "v0": [p.deriv()(0)],
                "a0": [p.deriv(2)(0)],
                "v1": [p.deriv()(1)],
                "a1": [p.deriv(2)(1)],
                "peak_epigraph": [0.95 if len(calls) == 1 else 1.0],
            }
        ]

    monkeypatch.setattr(module, "refine_transits", fake)
    solve = module.make_peak_transit_solver(0.1, 1.0)
    solve(None, None, (), [{"stage": "transit:1"}], {"stages": [{"duration_s": 1.0}]}, [1.0])
    assert calls[0]["seed_accelerations"] is None
    assert calls[1]["seed_accelerations"] == ([p.deriv(2)(0)], [p.deriv(2)(1)])
