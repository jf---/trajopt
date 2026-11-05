import numpy as np
import pytest
from fresh_tour_pipeline import FreshTourError, modeled_tour_end_secants, plan_and_refine


def test_tour_end_secants_are_explicit_modeled_velocities():
    a = np.zeros(9)
    b = a.copy()
    b[0] = 2
    c = b.copy()
    c[1] = 3
    stages = [
        {"stage": "transit:0", "states": [a.tolist(), b.tolist()]},
        {"stage": "transit:1", "states": [b.tolist(), c.tolist()]},
    ]
    clock = {"stages": [{"duration_s": 2.0}, {"duration_s": 3.0}]}
    start, end = modeled_tour_end_secants(stages, clock)
    assert start[0] == 1.0 and end[1] == 1.0


def test_planner_failure_does_not_call_verifier_or_refiner():
    def plan(*_args, **_kwargs):
        return {"status": "wall_budget_exhausted"}

    def verify(*_args, **_kwargs):
        raise AssertionError("must not verify failed plan")

    with pytest.raises(FreshTourError, match="wall_budget_exhausted"):
        plan_and_refine(
            {"inputs": ()}, 16, plan, verify, {}, {}, boundary_policy=modeled_tour_end_secants
        )


def test_checked_tour_evaluates_native_spline_and_reports_join(monkeypatch):
    import fresh_tour_pipeline as pipeline

    names = tuple(f"j{i}" for i in range(9))
    q0 = np.zeros(9)
    q1 = q0.copy()
    q1[0] = 0.1
    q2 = q1.copy()
    q2[0] = 0.2
    segments = [
        {"stage": "transit:0", "states": [q0.tolist(), q1.tolist()]},
        {"stage": "weld:1", "states": [q1.tolist(), q2.tolist()]},
    ]
    clock = {
        "stages": [
            {"stage": "transit:0", "duration_s": 1.0},
            {"stage": "weld:1", "duration_s": 1.0},
        ]
    }
    speed = np.ones(7)
    knots = np.linspace(0, 1, 7)
    nodes = np.zeros((7, 7))
    nodes[:, 0] = knots * 0.1
    velocity = np.zeros(7)
    velocity[0] = 0.1
    rows = [
        {
            "stage": "transit:0",
            "knots_s": knots.tolist(),
            "nodes": nodes.tolist(),
            "duration_s": 1.0,
            "v0": velocity.tolist(),
            "v1": velocity.tolist(),
            "fixture": [0.0, 0.0],
        }
    ]
    monkeypatch.setattr(pipeline, "prepare_fresh", lambda *_args: (segments, clock, speed))
    monkeypatch.setattr(pipeline, "refine_transits", lambda *_args, **_kwargs: rows)

    class Checker:
        def contacts(self, _q):
            return []

    monkeypatch.setattr(pipeline.StateChecker, "build", lambda *_args: Checker())

    class Limits:
        joint_limits = np.tile([-10.0, 10.0], (9, 1))

    class Group:
        def getLimits(self):
            return Limits()

    class Env:
        def getJointGroup(self, *_args):
            return Group()

    scene = {"inputs": (Env(), None, None, None, names, None, None, None, None, None, None, None)}
    result = pipeline.refine_checked_tour(
        scene,
        20,
        {},
        {"status": "independently_sampled_tour"},
        {},
        {},
        boundary_policy=lambda _segments, _clock: (np.zeros(9), np.zeros(9)),
    )
    assert result["transits"][0]["actual_curve_check"]["sample_count"] == 1025
    assert result["transits"][0]["sampled_speed_ratio_by_axis"][0] == pytest.approx(0.1)
    assert result["join_diagnostics"][0]["velocity_jump_per_s"][0] == pytest.approx(0.0)
    assert result["process_verification"]["status"] == "independently_sampled_tour"

    # A successful solver status cannot accept a disconnected endpoint.
    rows[0]["nodes"][-1][0] += 0.1
    with pytest.raises(FreshTourError, match="endpoint position differs"):
        pipeline.refine_checked_tour(
            scene,
            20,
            {},
            {"status": "independently_sampled_tour"},
            {},
            {},
            boundary_policy=lambda _segments, _clock: (np.zeros(9), np.zeros(9)),
        )
