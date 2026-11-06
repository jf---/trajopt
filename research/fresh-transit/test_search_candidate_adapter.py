import numpy as np
import pytest
from search_candidate_adapter import SearchCandidateError, checked_search_nodes


def example():
    q = np.zeros((6, 7))
    q[:, 0] = np.linspace(0.0, 1.0, 6)
    process = {
        "stage": "search:3",
        "duration_s": 2.0,
        "ik_states": q.tolist(),
        **{key: np.zeros(7).tolist() for key in ("v0", "a0", "v1", "a1")},
    }
    variant = {
        "line_mode": "hard",
        "progress_weight": 1.0,
        "status": "SQPStatus.NLP_CONVERGED",
        "actual_curve": {"penetrating_contacts": 0},
        "candidate_nodes": q.tolist(),
    }
    probe = {"probe": 3, "scene_restored": True, "variants": [variant]}
    evidence = {
        "generated_input_nodes": q.tolist(),
        "boundary_velocity_acceleration": {
            key: np.zeros(7).tolist() for key in ("v0", "a0", "v1", "a1")
        },
        "modeled_duration_s": 2.0,
    }
    return process, probe, evidence


def test_checked_search_retains_anchored_generated_curve():
    process, probe, evidence = example()
    nodes = checked_search_nodes(process, probe, evidence, 3)
    assert nodes == pytest.approx(np.asarray(process["ik_states"]))


def test_rejects_changed_joint_endpoint_and_failed_collision_check():
    process, probe, evidence = example()
    probe["variants"][0]["candidate_nodes"][0][0] = 0.2
    with pytest.raises(SearchCandidateError):
        checked_search_nodes(process, probe, evidence, 3)
    process, probe, evidence = example()
    probe["variants"][0]["actual_curve"]["penetrating_contacts"] = 1
    with pytest.raises(SearchCandidateError):
        checked_search_nodes(process, probe, evidence, 3)
