"""Bounded peak-witness orchestration retains the original zero-weight path."""

import numpy as np
import peak_transit_solver as module
from numpy.polynomial import Polynomial


def test_zero_weight_uses_original_solver_without_extra_arguments(monkeypatch):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return ["baseline"]

    monkeypatch.setattr(module, "refine_transits", fake)
    assert module.make_peak_transit_solver(0.0, 1.0)(None, None, (), [], {}, (), marker=1) == [
        "baseline"
    ]
    assert calls[0][1] == {"marker": 1}


def test_positive_weight_adds_extremum_and_warm_starts(monkeypatch):
    times = np.linspace(0.0, 1.0, 7)
    p = Polynomial([0.0, 0.0, 0.0, 0.91 / 6.0, 0.6 / 24.0, -1.0 / 60.0])
    calls = []

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
                "peak_epigraph": [0.95 if kwargs["seed_positions"] is None else 1.0],
            }
        ]

    monkeypatch.setattr(module, "refine_transits", fake)
    result = module.make_peak_transit_solver(0.1, 1.0)(
        None, None, (), [{"stage": "transit:1"}], {"stages": [{"duration_s": 1.0}]}, [1.0]
    )
    assert len(calls) == 2
    assert calls[0]["seed_positions"] is None
    assert calls[1]["seed_positions"] is not None
    assert len(result[0]["peak_witness_rounds"]) == 2
