import numpy as np
from fresh_joins import modeled_join_diagnostics


def test_reports_actual_spline_vs_process_slopes_in_nine_axes():
    a = np.zeros(9)
    b = a.copy()
    b[0] = 1
    c = b.copy()
    c[1] = 2
    stages = [
        {"stage": "transit:0", "states": [a.tolist(), b.tolist()]},
        {"stage": "approach:0", "states": [b.tolist(), c.tolist()]},
    ]
    clock = {
        "stages": [
            {"stage": "transit:0", "duration_s": 1},
            {"stage": "approach:0", "duration_s": 2},
        ]
    }
    end = np.r_[0.0, 0.5, np.zeros(7)]
    got = modeled_join_diagnostics(stages, clock, {"transit:0": {"end": (b, end)}})
    assert got[0]["velocity_jump_per_s"][1] == 0.5
    assert got[0]["position_jump"] == [0.0] * 9


def test_event_does_not_invent_velocity():
    q = np.zeros(9).tolist()
    stages = [{"stage": "weld:0", "states": [q, q]}, {"stage": "transit:1", "states": [q, q]}]
    clock = {
        "stages": [{"stage": "weld:0", "duration_s": 0}, {"stage": "transit:1", "duration_s": 1}]
    }
    assert modeled_join_diagnostics(stages, clock, {})[0]["velocity_jump_per_s"] is None
