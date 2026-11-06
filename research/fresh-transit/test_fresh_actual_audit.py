"""Fresh stage audit keeps original geometry and refuses unresolved smoothness."""

import numpy as np
import pytest
from actual_stage_contract import JoinContractError
from fresh_actual_audit import audit_stages, require_smoothness


def source():
    def q(x):
        return [x] + [0.0] * 8

    stages = [
        {"stage": "transit:0", "states": [q(0), q(1)]},
        {"stage": "search:0", "states": [q(1), q(2), q(1)]},
        {"stage": "transit:1", "states": [q(1), q(0)]},
    ]
    clock = {"stages": [{"stage": x["stage"], "duration_s": 1.0} for x in stages]}

    def transit(stage, first, last):
        return {
            "stage": stage,
            "duration_s": 1.0,
            "knots_s": [0.0, 1.0],
            "nodes": [q(first)[:7], q(last)[:7]],
            "fixture": [0.0, 0.0],
            "v0": [last - first] + [0.0] * 6,
            "v1": [last - first] + [0.0] * 6,
            "a0": [0.0] * 7,
            "a1": [0.0] * 7,
        }

    return stages, clock, [transit("transit:0", 0, 1), transit("transit:1", 1, 0)]


class LinearModel:
    def evaluate(self, times, nodes, derivative):
        p = np.array(nodes)
        t = float(times[0])
        return np.array([p[0] + t * (p[-1] - p[0])]), np.array([p[-1] - p[0]]), np.zeros((1, 7))


def test_corner_and_unsupplied_tour_boundaries_remain_unresolved():
    stages, clock, transits = source()
    report = audit_stages(stages, clock, transits, lambda _: LinearModel())
    assert report["stage_count"] == 3
    assert report["unresolved_corners"][0]["stage"] == "search:0"
    assert report["outer_boundary_derivatives"] == "unknown"
    assert report["smoothness_gate"] == "unresolved"
    with pytest.raises(JoinContractError, match="unresolved"):
        require_smoothness(report, (0.0,) * 9, (0.0,) * 9, (0.0,) * 9)


def test_incomplete_or_foreign_transit_is_not_silently_omitted():
    stages, clock, transits = source()
    with pytest.raises(JoinContractError, match="transit"):
        audit_stages(stages, clock, transits[:-1], lambda _: LinearModel())
    transits[-1]["stage"] = "transit:99"
    with pytest.raises(JoinContractError, match="transit"):
        audit_stages(stages, clock, transits, lambda _: LinearModel())


def test_supplied_positions_do_not_turn_unknown_derivatives_into_supplied_motion():
    from actual_stage_contract import KinematicState

    stages, clock, transits = source()
    unknown = KinematicState((0.0,) * 9, None, None)
    report = audit_stages(
        stages, clock, transits, lambda _: LinearModel(), initial_state=unknown, final_state=unknown
    )
    assert report["outer_boundary_derivatives"] == "unknown"
    assert report["unknown_join_derivatives"] >= 2
