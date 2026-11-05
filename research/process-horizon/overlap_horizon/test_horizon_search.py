"""Search and weighted objective contracts independent of a robot scene."""

import numpy as np
import pytest
from horizon_search import (
    CandidateInfeasible,
    HorizonSearchError,
    search_shared_rates,
    weighted_jerk,
)


def test_objective_uses_seconds_and_per_joint_speeds():
    t = np.array([0.0, 1.0, 2.0])
    jerk = np.tile([2.0, 4.0], (3, 1))
    assert weighted_jerk(t, jerk, np.array([1.0, 2.0])) == pytest.approx(16.0)


def test_search_keeps_failures_and_uses_shared_full_horizon_objective():
    def candidate(left, right):
        if (left, right) == (0.5, 1.0):
            raise CandidateInfeasible("sampled collision")
        return {"weighted_jerk": (left - 1.0) ** 2 + (right - 0.5) ** 2, "label": [left, right]}

    outcomes, selected = search_shared_rates([0.5, 1.0], candidate)
    assert len(outcomes) == 4
    assert sum(x["status"] == "failed" for x in outcomes) == 1
    assert selected["label"] == [1.0, 0.5]


def test_search_fails_loudly_if_every_candidate_fails_or_cost_is_invalid():
    def fail(_left, _right):
        raise CandidateInfeasible("infeasible")

    with pytest.raises(HorizonSearchError):
        search_shared_rates([1.0], fail)
    with pytest.raises(HorizonSearchError):
        search_shared_rates([1.0], lambda _left, _right: {"weighted_jerk": float("nan")})


def test_process_order_rejects_material_reverse_but_not_float_roundoff():
    from horizon_search import require_ordered_progress

    require_ordered_progress(
        {"progress_min": -1e-15, "progress_max": 1.0 + 1e-15, "max_backward_progress": 1e-15}
    )
    with pytest.raises(CandidateInfeasible, match="progress"):
        require_ordered_progress(
            {"progress_min": 0.0, "progress_max": 1.0, "max_backward_progress": 0.1}
        )
